import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.atpg.logic_sim_three import print_pi, reset_gates
from src.atpg.podem import (
    SUCCESS,
    get_all_faults,
    initialize,
    podem,
    simple_backtrace,
)
from src.atpg.reconv_podem import PathConsistencySolver
from src.atpg.recursive_reconv_solver import (
    HierarchicalReconvSolver,
    ReconvPairPredictor,
)
from src.ml.core.model import MultiPathTransformer
from src.ml.data.embedding import EmbeddingExtractor
from src.util.struct import Fault, Gate, GateType, LogicValue


def _logic_value_label(value: LogicValue | int) -> str:
    mapping = {
        LogicValue.ZERO: "0",
        LogicValue.ONE: "1",
        LogicValue.XD: "X",
        LogicValue.D: "D",
        LogicValue.DB: "DB",
    }
    try:
        return mapping[LogicValue(value)]
    except Exception:
        return str(value)


def _format_assignment(assignment: Dict[int, LogicValue], limit: int = 20) -> str:
    if not assignment:
        return "{}"

    items = sorted(assignment.items())
    rendered = [f"{node}={_logic_value_label(value)}" for node, value in items[:limit]]
    if len(items) > limit:
        rendered.append(f"... +{len(items) - limit} more")
    return "{" + ", ".join(rendered) + "}"


def _format_paths(paths: List[List[int]]) -> str:
    return " | ".join("->".join(str(node) for node in path) for path in paths)


def _format_pair(pair_info: Dict[str, Any]) -> str:
    return (
        f"start={pair_info['start']} reconv={pair_info['reconv']} "
        f"paths=[{_format_paths(pair_info['paths'])}]"
    )


def _binary_constraint_value(value: LogicValue | int) -> LogicValue | None:
    value = LogicValue(value)
    if value in (LogicValue.ZERO, LogicValue.ONE):
        return value
    if value == LogicValue.D:
        return LogicValue.ONE
    if value == LogicValue.DB:
        return LogicValue.ZERO
    return None


@dataclass
class AiPodemConfig:
    model_path: str
    device: str = "cuda"
    enable_ai_activation: bool = True
    enable_ai_propagation: bool = True
    verbose: bool = False
    no_fallback: bool = False  # If True, never fall back to classic backtrace/PODEM
    candidate_count: int = 8
    candidate_seed_base: int = 20260504
    candidate_temperature: float = 0.7
    enable_symbolic_repair: bool = True


class AIBacktracer:
    """
    Backtrace function that uses HierarchicalReconvSolver to satisfy objectives.
    Falls back to simple_backtrace if AI fails.
    """

    def __init__(
        self, solver: HierarchicalReconvSolver, verbose: bool = False, no_fallback: bool = False
    ):
        self.solver = solver
        self.circuit = solver.circuit
        self.verbose = verbose
        self.no_fallback = no_fallback
        # Precompute PI indices to avoid O(N) iteration in every __call__
        self.pi_indices = [i for i, g in enumerate(self.circuit) if g.type == GateType.INPT]

    def __call__(self, objective: Fault, circuit: List[Gate]) -> Fault:
        # Objective: gate_id, value. Try AI Solve.
        if self.verbose:
            print(f"[AI-BT] Objective: Gate {objective.gate_id} = {objective.value}")
        try:
            # Fast path: skip AI if no reconvergent structure exists
            if hasattr(self.solver, "pair_cache"):
                if objective.gate_id not in self.solver.pair_cache:
                    pairs = self.solver._collect_and_sort_pairs(objective.gate_id)
                    self.solver.pair_cache[objective.gate_id] = pairs
                else:
                    pairs = self.solver.pair_cache[objective.gate_id]

                if not pairs and not self.no_fallback:
                    if self.verbose:
                        print(
                            f"  [AI-BT] No reconv pairs for gate {objective.gate_id}, skipping AI."
                        )
                    return simple_backtrace(objective, circuit)

            # Build constraints from the live PODEM state.  D/DB carry the
            # good-circuit value needed by the reconvergent justification solver.
            current_constraints = {}
            for i, g in enumerate(self.circuit):
                if g is None:
                    continue
                constraint_val = _binary_constraint_value(g.val)
                if constraint_val is not None:
                    current_constraints[i] = constraint_val

            import hashlib

            seed_material = (
                f"{objective.gate_id}:{int(objective.value)}:"
                f"{sorted((int(k), int(v)) for k, v in current_constraints.items())}"
            )
            current_seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
            if self.verbose:
                print(f"  [AI-BT] Constraints: {current_constraints}, Seed: {current_seed}")

            solution = self.solver.solve(
                objective.gate_id,
                objective.value,
                current_constraints,
                seed=current_seed,
            )

            if solution:
                if self.verbose:
                    print(f"  [AI-BT] Solution: {solution}")
                # 1. Try to find a direct PI assignment
                for gid, val in solution.items():
                    if (
                        self.circuit[gid].type == GateType.INPT
                        and self.circuit[gid].val == LogicValue.XD
                    ):
                        if self.verbose:
                            print(f"  [AI-BT] Returning assignment: Gate {gid}={val}")
                        return Fault(gid, val)

                # 2. If no PI, finding an internal node in solution that needs
                # justification and use simple_backtrace to reach a PI from there.
                if self.verbose:
                    print("  [AI-BT] No direct PI found. Looking for intermediate objectives...")
                for gid, val in solution.items():
                    if self.circuit[gid].val == LogicValue.XD:
                        if self.verbose:
                            print(
                                "  [AI-BT] Delegating to simple_backtrace for "
                                f"internal objective: Gate {gid}={val}"
                            )
                        return simple_backtrace(Fault(gid, val), circuit)

                if self.verbose:
                    print("  [AI-BT] Solution found but all nodes already assigned/consistent?")
            else:
                if self.verbose:
                    print("  [AI-BT] No solution from solver.")
        except Exception as e:
            if self.verbose:
                print(f"  [AI-BT] Error: {e}")
                import traceback

                traceback.print_exc()
            pass

        if self.no_fallback:
            if self.verbose:
                print(
                    f"  [AI-BT] AI backtrace failed for gate {objective.gate_id}; "
                    "returning UNTESTABLE decision because fallback is disabled."
                )
            return Fault(-1, -1)
        # Fallback to simple
        if self.verbose:
            print("  [AI-BT] Fallback to simple_backtrace")
        return simple_backtrace(objective, circuit)


class StaticHintBacktracer:
    """Backtrace with AI activation hints, falling back to the classic heuristic.

    The activation solver may return values for internal nodes that are not
    directly applied to the circuit.  This backtracer uses those values only to
    choose among otherwise valid X fanins for the current PODEM objective.
    """

    def __init__(
        self,
        hints: Dict[int, LogicValue],
        verbose: bool = False,
        no_fallback: bool = False,
    ):
        self.hints = {int(k): LogicValue(v) for k, v in hints.items()}
        self.verbose = verbose
        self.no_fallback = no_fallback

    def __call__(self, objective: Fault, circuit: List[Gate]) -> Fault:
        result = self._hinted_backtrace(objective, circuit)
        if result is not None:
            return result
        if self.no_fallback:
            raise RuntimeError(
                f"[AI-HINT] No complete hint path for gate {objective.gate_id}"
                " and fallback is disabled."
            )
        return simple_backtrace(objective, circuit)

    def _hinted_backtrace(self, objective: Fault, circuit: List[Gate]) -> Fault | None:
        current_id = int(objective.gate_id)
        target = LogicValue(objective.value)

        while circuit[current_id].nfi != 0:
            next_choice = self._choose_hinted_fanin(circuit, current_id, target)
            if next_choice is None:
                return None
            current_id, target = next_choice

        if self.verbose:
            print(f"[AI-HINT] Backtrace selected PI {current_id}={_logic_value_label(target)}")
        return Fault(current_id, target)

    def _choose_hinted_fanin(
        self,
        circuit: List[Gate],
        gate_id: int,
        target: LogicValue,
    ) -> tuple[int, LogicValue] | None:
        gate = circuit[gate_id]
        x_fanins = [fin for fin in gate.fin if circuit[fin].val == LogicValue.XD]
        if not x_fanins:
            return None

        required = self._required_fanin_value(gate.type, target)
        if required is None:
            return None

        for fin in x_fanins:
            if self.hints.get(fin) == required:
                if self.verbose:
                    print(
                        f"[AI-HINT] Gate {gate_id}={_logic_value_label(target)} -> "
                        f"{fin}={_logic_value_label(required)}"
                    )
                return fin, required
        return None

    @staticmethod
    def _required_fanin_value(
        gate_type: GateType,
        target: LogicValue,
    ) -> LogicValue | None:
        if gate_type == GateType.BUFF:
            return target
        if gate_type == GateType.NOT:
            return LogicValue.ONE if target == LogicValue.ZERO else LogicValue.ZERO
        if gate_type == GateType.AND:
            return LogicValue.ONE if target == LogicValue.ONE else LogicValue.ZERO
        if gate_type == GateType.NAND:
            return LogicValue.ONE if target == LogicValue.ZERO else LogicValue.ZERO
        if gate_type == GateType.OR:
            return LogicValue.ZERO if target == LogicValue.ZERO else LogicValue.ONE
        if gate_type == GateType.NOR:
            return LogicValue.ZERO if target == LogicValue.ONE else LogicValue.ONE
        return None


def _podem_succeeded(result: int | bool) -> bool:
    """Return True only for an explicit PODEM success status."""
    return result == SUCCESS


def post_process_logic_gates(
    vals: torch.Tensor,  # [P, L] predicted values (0 or 1)
    gate_types: torch.Tensor,  # [P, L] gate type for each position
    mask: torch.Tensor,  # [P, L] valid mask (True for real nodes)
    constraints: Optional[Dict[int, "LogicValue"]] = None,
    node_ids: Optional[torch.Tensor] = None,  # [P, L] for constraint lookup
) -> torch.Tensor:
    """Forward-propagate deterministic gate rules (NOT/BUFF) through paths.

    For each path, iterates from position 0 forward. At each position:
    - NOT gate: force cur = 1 - prev
    - BUFF gate: force cur = prev
    - Others: keep model prediction (AND/OR/NAND/NOR satisfy inequality constraints)

    Also respects any externally provided constraints.

    Returns: corrected vals tensor [P, L]
    """
    corrected = vals.clone()
    P, L = vals.shape

    for p in range(P):
        path_len = mask[p].sum().item()
        if path_len <= 1:
            continue

        # If constraints exist, apply them to the first position
        if constraints is not None and node_ids is not None:
            nid = int(node_ids[p, 0].item())
            if nid in constraints:
                corrected[p, 0] = 0 if constraints[nid] == LogicValue.ZERO else 1

        # Forward propagate
        for pos in range(1, int(path_len)):
            gt = int(gate_types[p, pos].item())
            prev_val = int(corrected[p, pos - 1].item())

            # Apply constraints first if available
            if constraints is not None and node_ids is not None:
                nid = int(node_ids[p, pos].item())
                if nid in constraints:
                    corrected[p, pos] = 0 if constraints[nid] == LogicValue.ZERO else 1
                    continue

            # Deterministic gate rules
            if gt == GateType.NOT:
                corrected[p, pos] = 1 - prev_val
            elif gt == GateType.BUFF:
                corrected[p, pos] = prev_val
            # AND/NAND/OR/NOR: keep model prediction (inequality-based)

    return corrected


class ModelPairPredictor(ReconvPairPredictor):
    def __init__(
        self,
        circuit: List[Gate],
        circuit_path: str,
        config: AiPodemConfig,
        pre_loaded_model=None,
    ):
        self.circuit_path = circuit_path
        self.circuit = circuit
        self.config = config
        self.device = torch.device(config.device)
        if config.verbose:
            print(f"[AI-BT] Using device: {self.device}")

        # Load embeddings — uses disk cache keyed on circuit_path
        self.extractor = EmbeddingExtractor()
        self.struct_emb, _, self.gate_mapping, _ = self.extractor.extract_embeddings(
            circuit_path,
            pre_parsed_circuit=circuit,  # Avoids redundant re-parse inside extractor
        )
        self.struct_emb = self.struct_emb.to(self.device)
        # Map str(id) -> int(aig_id)
        self.gate_mapping = {int(k): int(v) for k, v in self.gate_mapping.items()}
        self.prediction_cache = {}

        # Use pre-loaded model if provided (avoids redundant torch.load per benchmark)
        if pre_loaded_model is not None:
            self.model = pre_loaded_model
        else:
            self.model = self._load_model(config.model_path)

        self.solver = PathConsistencySolver(circuit)

    def _load_model(self, model_path: str):
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
                input_dim = int(
                    cfg.get("input_dim", cfg.get("observed_dim", cfg.get("embedding_dim", 132)))
                )
                if input_dim < 132:
                    input_dim = 132
                model_dim = int(cfg.get("model_dim", 512))
                nhead = int(cfg.get("nhead", 4))
                enc_layers = int(cfg.get("num_encoder_layers", cfg.get("enc_layers", 3)))
                int_layers = int(cfg.get("num_interaction_layers", cfg.get("int_layers", 3)))
                ffn_dim = int(cfg.get("dim_feedforward", cfg.get("ffn_dim", 512)))
                model = MultiPathTransformer(
                    input_dim=input_dim,
                    model_dim=model_dim,
                    nhead=nhead,
                    num_encoder_layers=enc_layers,
                    num_interaction_layers=int_layers,
                    dim_feedforward=ffn_dim,
                ).to(self.device)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                model.eval()
                return model
            except Exception as e:
                print(f"[AI-PODEM] Failed to load model weights: {e}")
        else:
            print(f"[AI-PODEM] Model path not found: {model_path}. Using random weights.")

        model = MultiPathTransformer(
            input_dim=132,
            model_dim=512,
            nhead=4,
            num_encoder_layers=3,
            num_interaction_layers=3,
            dim_feedforward=512,
        ).to(self.device)
        model.eval()
        return model

    def predict(
        self,
        pair_info: Dict[str, Any],
        constraints: Dict[int, LogicValue],
        seed: Optional[int] = None,
    ) -> Tuple[List[Dict[int, LogicValue]], Optional[Dict[str, Any]]]:
        path_nodes = set()
        for p in pair_info["paths"]:
            path_nodes.update(p)

        relevant_constraints = frozenset(
            (nid, val) for nid, val in constraints.items() if nid in path_nodes
        )
        cache_seed = int(seed if seed is not None else self.config.candidate_seed_base)
        cache_key = (pair_info["start"], pair_info["reconv"], relevant_constraints, cache_seed)

        if len(self.prediction_cache) > 500:
            self.prediction_cache.clear()

        if cache_key in self.prediction_cache:
            if self.config.verbose:
                cached_candidates, _ = self.prediction_cache[cache_key]
                print(
                    "[AI-MODEL] Cache hit for "
                    f"{_format_pair(pair_info)} with constraints "
                    f"{_format_assignment(dict(relevant_constraints))}"
                )
                print(
                    f"[AI-MODEL] Cached candidates: "
                    f"{[_format_assignment(candidate) for candidate in cached_candidates]}"
                )
            return self.prediction_cache[cache_key]

        if self.struct_emb is None or self.model is None:
            # Fallback to pure solver if model failed
            res = self._fallback_solve(pair_info, constraints)[0], None
            if self.config.verbose:
                print(
                    f"[AI-MODEL] Model unavailable, using fallback for {_format_pair(pair_info)}"
                )
            self.prediction_cache[cache_key] = res
            return res

        # 1. Prepare Batch for Model

        paths = pair_info["paths"]

        # Optimization: If all nodes in paths already have values, use them directly
        all_constrained = True
        precomputed_assignment = {}
        for p in paths:
            for nid in p:
                if nid in constraints:
                    precomputed_assignment[nid] = constraints[nid]
                else:
                    all_constrained = False
                    break
            if not all_constrained:
                break

        if all_constrained and precomputed_assignment:
            # All gates already have values - skip model, return existing values
            # Need strict type match for return tuple
            res = [precomputed_assignment], None
            if self.config.verbose:
                print(
                    f"[AI-MODEL] Skipping inference for {_format_pair(pair_info)} because all "
                    f"path nodes are constrained: {_format_assignment(precomputed_assignment)}"
                )
            self.prediction_cache[cache_key] = res
            return res

        # Convert path node IDs to AIG IDs to get embeddings
        path_embs_list = []
        gate_types_list = []
        node_ids_list = []  # For saving state

        max_len = max(len(p) for p in paths)

        for p in paths:
            p_emb = []
            p_types = []
            p_ids = []
            for nid in p:
                p_ids.append(nid)
                if nid in self.gate_mapping:
                    aig_id = self.gate_mapping[nid]
                    if aig_id < self.struct_emb.size(0):
                        p_emb.append(self.struct_emb[aig_id].clone())
                    else:
                        p_emb.append(torch.zeros(128, device=self.device))
                else:
                    p_emb.append(torch.zeros(128, device=self.device))

                # Pad to 131 (3 logic dims) matching training encoding:
                #   dim 128: val=0 (ZERO)
                #   dim 129: val=1 (ONE)
                #   dim 130: unknown (default 1.0)
                if p_emb[-1].shape[0] < 131:
                    logic_dims = torch.zeros(131 - p_emb[-1].shape[0], device=self.device)
                    p_emb[-1] = torch.cat([p_emb[-1], logic_dims])

                if nid in constraints:
                    val = constraints[nid]
                    if val == LogicValue.ZERO:
                        p_emb[-1][128] = 1.0  # val=0
                        p_emb[-1][129] = 0.0  # val=1
                        p_emb[-1][130] = 0.0  # unknown
                    elif val == LogicValue.ONE:
                        p_emb[-1][128] = 0.0  # val=0
                        p_emb[-1][129] = 1.0  # val=1
                        p_emb[-1][130] = 0.0  # unknown
                else:
                    # Unknown: [0, 0, 1] — matches training default
                    p_emb[-1][128] = 0.0
                    p_emb[-1][129] = 0.0
                    p_emb[-1][130] = 1.0

                # Pad to 132 (next multiple of 4) to satisfy nhead divisibility
                if p_emb[-1].shape[0] < 132:
                    p_emb[-1] = torch.cat(
                        [p_emb[-1], torch.zeros(132 - p_emb[-1].shape[0], device=self.device)]
                    )

                # Gate Type
                if nid < len(self.circuit):
                    p_types.append(self.circuit[nid].type)
                else:
                    p_types.append(0)  # Unknown

            # Pad sequence
            while len(p_emb) < max_len:
                # Padding slots: struct zeros + unknown logic + zero pad
                pad_emb = torch.zeros(132, device=self.device)
                pad_emb[130] = 1.0  # unknown
                p_emb.append(pad_emb)
                p_types.append(0)
                p_ids.append(0)

            path_embs_list.append(torch.stack(p_emb))
            gate_types_list.append(torch.tensor(p_types, device=self.device))
            node_ids_list.append(torch.tensor(p_ids, device=self.device))

        # Stack to [1, P, L, D]
        batch_embs = torch.stack(path_embs_list).unsqueeze(0)
        batch_types = torch.stack(gate_types_list).unsqueeze(0)
        batch_ids = torch.stack(node_ids_list).unsqueeze(0)

        batch_mask = torch.ones((1, len(paths), max_len), dtype=torch.bool, device=self.device)
        for i, p in enumerate(paths):
            batch_mask[0, i, len(p) :] = False

        # Snapshot for RL (Clone to CPU)
        inputs_snapshot = {
            "node_ids": batch_ids.cpu(),
            "mask_valid": batch_mask.cpu(),
            "gate_types": batch_types.cpu(),
            "files": [self.circuit_path],
        }

        if self.config.verbose:
            print(f"[AI-MODEL] Query: {_format_pair(pair_info)}")
            print(
                "[AI-MODEL] Relevant constraints: "
                f"{_format_assignment(dict(relevant_constraints))}"
            )
            print(
                "[AI-MODEL] Tensor shapes: "
                f"emb={tuple(batch_embs.shape)} mask={tuple(batch_mask.shape)} "
                f"types={tuple(batch_types.shape)}"
            )
            if seed is not None:
                print(f"[AI-MODEL] Seed: {seed}")

        # 2. Run Inference
        with torch.no_grad():
            # Inject noise if seed is provided. Scale could be configurable.
            perturb_scale = 0.5 if seed is not None else 0.0
            logits, solv_logits = self.model(
                batch_embs,
                batch_mask,
                batch_types,
                node_ids=batch_ids,  # Pass Node IDs for embedding
                seed=seed,
                perturb_scale=perturb_scale,
            )

        # 3. Decode deterministic candidate set.
        probs = torch.softmax(logits, dim=-1)  # [1, P, L, 2]
        solv_probs = torch.softmax(solv_logits, dim=-1).squeeze(0)

        candidate_vals = []
        candidate_vals.append(torch.argmax(probs, dim=-1).squeeze(0))
        n_candidates = max(1, int(self.config.candidate_count))
        if n_candidates > 1:
            temp = max(1e-3, float(self.config.candidate_temperature))
            sample_probs = torch.softmax(logits / temp, dim=-1).squeeze(0).detach().cpu()
            flat_probs = sample_probs.reshape(-1, 2)
            for cidx in range(1, n_candidates):
                gen = torch.Generator(device="cpu")
                gen.manual_seed(cache_seed + cidx)
                sampled = torch.multinomial(flat_probs, 1, generator=gen).reshape(
                    len(paths), max_len
                )
                candidate_vals.append(sampled.to(self.device))

        candidate_assignments = []
        seen_assignments = set()
        for vals in candidate_vals:
            vals = post_process_logic_gates(
                vals,
                batch_types.squeeze(0),
                batch_mask.squeeze(0),
                constraints=constraints,
                node_ids=batch_ids.squeeze(0),
            )
            predicted_assignment = {}
            node_confidence = {}

            for i, p in enumerate(paths):
                for j, nid in enumerate(p):
                    if j >= len(p):
                        continue
                    val = int(vals[i, j].item())
                    lv = LogicValue.ZERO if val == 0 else LogicValue.ONE
                    conf = float(probs[0, i, j, val].item())

                    if nid in constraints:
                        lv = constraints[nid]
                        conf = 1.0

                    if nid in predicted_assignment and predicted_assignment[nid] != lv:
                        prev_conf = node_confidence.get(nid, 0.0)
                        if conf <= prev_conf:
                            continue

                    predicted_assignment[nid] = lv
                    node_confidence[nid] = conf

            signature = tuple(sorted((int(k), int(v)) for k, v in predicted_assignment.items()))
            if signature not in seen_assignments:
                seen_assignments.add(signature)
                candidate_assignments.append(predicted_assignment)

        res = self._rank_solutions_with_model(
            pair_info,
            constraints,
            probs,
            paths,
            candidate_assignments,
            inputs_snapshot,
            bench_file=self.circuit_path,
        )
        if self.config.verbose:
            candidates, _ = res
            print(
                "[AI-MODEL] Solvability probs: "
                f"{[round(float(prob), 4) for prob in solv_probs.tolist()]}"
            )
            print(
                f"[AI-MODEL] Raw candidates: "
                f"{[_format_assignment(c) for c in candidate_assignments[:3]]}"
            )
            print(
                f"[AI-MODEL] Ranked candidates: "
                f"{[_format_assignment(candidate) for candidate in candidates]}"
            )
        self.prediction_cache[cache_key] = res
        return res

    def _rank_solutions_with_model(
        self,
        pair_info,
        constraints,
        probs,
        paths,
        predicted_assignments,
        inputs_snapshot,
        bench_file="",
    ):
        ranked = []
        rejected = 0
        for assignment in predicted_assignments:
            violations = self._verify_assignment_logic(assignment, constraints)
            if violations > 0:
                rejected += 1
                continue
            # Prefer compact assignments that satisfy more visible constraints and
            # introduce fewer internal commitments for PODEM to justify.
            n_internal = sum(
                1
                for gid in assignment
                if gid < len(self.circuit) and self.circuit[gid].type != GateType.INPT
            )
            ranked.append((n_internal, assignment))

        ranked.sort(key=lambda x: x[0])
        model_candidates = [assignment for _, assignment in ranked]
        candidates = list(model_candidates)
        repair_candidates = []

        if self.config.enable_symbolic_repair:
            fallback, _ = self._fallback_solve(pair_info, constraints)
            seen = {
                tuple(sorted((int(k), int(v)) for k, v in candidate.items()))
                for candidate in candidates
            }
            for candidate in fallback:
                signature = tuple(sorted((int(k), int(v)) for k, v in candidate.items()))
                if signature not in seen:
                    repair_candidates.append(candidate)
                    candidates.append(candidate)
                    seen.add(signature)
            if self.config.verbose:
                print(
                    f"[AI-MODEL] Rejected {rejected} model candidate(s); added "
                    f"{len(fallback)} symbolic repair candidates for {_format_pair(pair_info)}"
                )
        elif not candidates and self.config.verbose:
            print(
                f"[AI-MODEL] Rejected {rejected} model candidate(s); "
                f"symbolic repair disabled for {_format_pair(pair_info)}"
            )
        elif self.config.verbose:
            print(
                f"[AI-MODEL] Accepted {len(candidates)} model candidate(s), "
                f"rejected {rejected} for {_format_pair(pair_info)}"
            )

        limit = max(1, int(self.config.candidate_count))
        if repair_candidates and len(candidates) > limit:
            model_limit = max(0, limit - len(repair_candidates))
            candidates = model_candidates[:model_limit] + repair_candidates

        return candidates[:limit], inputs_snapshot

    def _verify_assignment_logic(
        self, assignment: Dict[int, LogicValue], constraints: Dict[int, LogicValue] = None
    ) -> int:
        """Verify logical consistency and return the count of violations.

        Returns 0 if all gates are consistent, otherwise the number of
        gates whose predicted value contradicts their Boolean truth table.
        """
        from src.atpg.logic_sim_three import compute_gate_value

        violations = 0
        for nid, val in assignment.items():
            if nid >= len(self.circuit):
                continue
            gate = self.circuit[nid]
            if not gate.fin:
                continue

            full_ctx = {}
            if constraints:
                full_ctx.update(constraints)
            full_ctx.update(assignment)

            # Check if all inputs are present in the full context
            if all(fin in full_ctx for fin in gate.fin):
                original_vals = {fin: self.circuit[fin].val for fin in gate.fin}
                original_gate_val = gate.val

                for fin in gate.fin:
                    self.circuit[fin].val = full_ctx[fin]

                expected_val = compute_gate_value(self.circuit, gate)

                for fin, v in original_vals.items():
                    self.circuit[fin].val = v
                gate.val = original_gate_val

                if expected_val != val:
                    violations += 1
                    if self.config.verbose:
                        print(
                            f"  [AI-BT] Logic Mismatch at Gate "
                            f"{nid} ({gate.type}): "
                            f"Expected {expected_val}, "
                            f"Predicted {val}"
                        )

        return violations

    def _fallback_solve(
        self, pair_info, constraints
    ) -> Tuple[List[Dict[int, LogicValue]], Optional[Dict[str, Any]]]:
        # Try both 0 and 1 for Reconvergence Node
        reconv_node = pair_info["reconv"]
        targets = []
        if reconv_node in constraints:
            targets.append(constraints[reconv_node])
        else:
            targets = [LogicValue.ZERO, LogicValue.ONE]

        # Create minimal snapshot for RL tracking even in fallback
        paths = pair_info.get("paths", [])
        if paths:
            max_len = max(len(p) for p in paths)
            node_ids = torch.zeros(1, len(paths), max_len, dtype=torch.long)
            mask_valid = torch.zeros(1, len(paths), max_len, dtype=torch.bool)
            gate_types = torch.zeros(1, len(paths), max_len, dtype=torch.long)

            for i, p in enumerate(paths):
                for j, nid in enumerate(p):
                    node_ids[0, i, j] = nid
                    mask_valid[0, i, j] = True
                    if nid < len(self.circuit):
                        gate_types[0, i, j] = self.circuit[nid].type

            snapshot = {
                "node_ids": node_ids,
                "mask_valid": mask_valid,
                "gate_types": gate_types,
                "files": [self.circuit_path],
            }
        else:
            snapshot = None

        candidates = []
        for t in targets:
            if self.config.verbose:
                print(
                    f"[AI-MODEL] Fallback solve for {_format_pair(pair_info)} with "
                    f"reconv target {_logic_value_label(t)}"
                )
            res = self.solver.solve(pair_info, t, constraints)
            if res:
                candidates.append(res)
                if self.config.verbose:
                    print(
                        f"[AI-MODEL] Fallback candidate: {_format_assignment(res)}"
                    )
        return candidates, snapshot


def ai_podem(
    circuit: List[Gate],
    fault: Fault,
    total_gates: int,
    model_path: str = "checkpoints/reconv_minimal_model.pt",
    circuit_path: str = "",
    enable_ai_activation: bool = True,
    enable_ai_propagation: bool = False,
    predictor: Optional[ModelPairPredictor] = None,
    solver: Optional[HierarchicalReconvSolver] = None,
    verbose: bool = False,
    seed: Optional[int] = None,
    no_fallback: bool = False,
    max_backtracks: int = 5000,
) -> bool:
    """
    AI-Assisted PODEM with configurable modes.

    Args:
        enable_ai_activation: Use AI Solver to justify fault activation (pre-fill).
        enable_ai_propagation: Use AI Solver for backtracing during propagation.
    """

    # Initialize shared PODEM structures
    initialize(circuit, total_gates)
    reset_gates(circuit, total_gates)

    # Predictor & Solver Setup

    if enable_ai_activation or enable_ai_propagation:
        if solver is None:
            if not circuit_path:
                print("[AI-PODEM] Warning: circuit_path missing, AI might fail.")
            if predictor is None:
                # Create config from args
                config = AiPodemConfig(
                    model_path=model_path,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    enable_ai_activation=enable_ai_activation,
                    enable_ai_propagation=enable_ai_propagation,
                    no_fallback=no_fallback,
                )
                predictor = ModelPairPredictor(circuit, circuit_path, config)
            solver = HierarchicalReconvSolver(circuit, predictor)
        if predictor is not None:
            predictor.config.no_fallback = no_fallback

    # --- Step 1 & 2: AI Justification (Activation) + Hybrid PODEM ---
    result = False
    if enable_ai_activation and solver:
        # Target: Fault Activation (If s-a-0 or D, we want 1. If s-a-1 or DB, we want 0)
        activation_val = (
            LogicValue.ONE if fault.value in [LogicValue.ZERO, LogicValue.D] else LogicValue.ZERO
        )

        # If seed is provided, we just do ONE attempt with that specific seed
        # Otherwise, we rotate through multiple attempts (standard hybrid behavior)
        max_attempts = 1 if seed is not None else 5
        for attempt in range(max_attempts):
            current_seed = seed if seed is not None else (42 + attempt)

            if verbose:
                print(
                    f"[AI-PODEM] Attempt {attempt+1}/{max_attempts}: Justifying Gate "
                    f"{fault.gate_id} @ {activation_val} (Seed: {current_seed})"
                )
            ai_assignment = solver.solve(fault.gate_id, activation_val, seed=current_seed)
            if ai_assignment:
                if verbose:
                    print(
                        f"[AI-PODEM] AI found activation assignment ({len(ai_assignment)} gates)."
                    )
                # Reset and apply to PIs
                reset_gates(circuit, total_gates)
                pi_cnt = 0
                for gid, val in ai_assignment.items():
                    if circuit[gid].type == GateType.INPT:
                        circuit[gid].val = val
                        pi_cnt += 1
                if verbose:
                    print(f"[AI-PODEM] Applied {pi_cnt} PI assignments.")

                # Run PODEM from this starting state
                backtracer = None
                if enable_ai_propagation and solver:
                    backtracer = AIBacktracer(
                        solver,
                        verbose=verbose,
                        no_fallback=no_fallback,
                    )
                elif ai_assignment:
                    backtracer = StaticHintBacktracer(
                        ai_assignment,
                        verbose=verbose,
                        no_fallback=no_fallback,
                    )

                result = mogu_podem_wrapper(
                    circuit,
                    fault,
                    total_gates,
                    backtrace_func=backtracer,
                    max_backtracks=max_backtracks,
                )
                if _podem_succeeded(result):
                    if verbose:
                        print(f"[AI-PODEM] Success on attempt {attempt+1}!")
                        print("Test Pattern:", print_pi(circuit, total_gates))
                    return True
                else:
                    if verbose:
                        print(f"[AI-PODEM] PODEM failed (or hit limit) on attempt {attempt+1}.")
            else:
                if verbose:
                    print(f"[AI-PODEM] AI Solver failed to find assignment on attempt {attempt+1}.")

        if not result and verbose:
            print("[AI-PODEM] All AI activation attempts failed.")

    else:
        # No AI activation pre-fill, just run PODEM (maybe with AI propagation)
        backtracer = None
        if enable_ai_propagation and solver:
            backtracer = AIBacktracer(
                solver,
                verbose=verbose,
                no_fallback=no_fallback,
            )
        result = mogu_podem_wrapper(
            circuit,
            fault,
            total_gates,
            backtrace_func=backtracer,
            max_backtracks=max_backtracks,
        )
        if _podem_succeeded(result):
            if verbose:
                print("[AI-PODEM] Success (No Activation pre-fill)!")
                print("Test Pattern:", print_pi(circuit, total_gates))
            return True

    # --- Step 3: Global Fallback ---
    # If we used AI Activation and failed, retry Clean (unless no_fallback is set)
    if enable_ai_activation and not _podem_succeeded(result):
        if no_fallback:
            return False
        if verbose:
            print("[AI-PODEM] AI-assisted attempts failed. Retrying CLEAN (Standard PODEM)...")
        reset_gates(circuit, total_gates)

        result_retry = mogu_podem_wrapper(
            circuit,
            fault,
            total_gates,
            backtrace_func=None,
            max_backtracks=max_backtracks,
        )
        if _podem_succeeded(result_retry):
            if verbose:
                print("[AI-PODEM] Clean retry Success!")
                print("Test Pattern:", print_pi(circuit, total_gates))
            return True

    if verbose:
        print("[AI-PODEM] Failure.")
    return False


def mogu_podem_wrapper(circuit, fault, total_gates, backtrace_func=None, max_backtracks=2000):
    # Wrapper to call the global `podem` function from src.atpg.podem
    return podem(
        circuit,
        fault,
        total_gates,
        backtrace_func=backtrace_func,
        max_backtracks=max_backtracks,
    )


if __name__ == "__main__":
    # Test runner
    import sys

    bench = sys.argv[1] if len(sys.argv) > 1 else "data/bench/c17.bench"
    print(f"Testing AI-PODEM on {bench}")

    # Parse
    from src.util.io import parse_bench_file

    circuit, total_gates = parse_bench_file(bench)
    faults = get_all_faults(circuit, total_gates)

    # Pick a fault
    fault = faults[0]
    result = ai_podem(circuit, fault, total_gates, circuit_path=bench)
