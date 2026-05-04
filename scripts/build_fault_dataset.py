"""Build a fault-driven reconvergent-path dataset for supervised training.

Each sample in the output pickle represents one reconvergent path pair
encountered during the hierarchical solver's execution order for a specific
fault.  Constraints are accumulated exactly as the runtime solver does:

  - Pair 1 for a fault  →  constraint = {terminus_node: sat_value} only.
  - Pair k (k > 1)      →  all path-nodes from pairs 1..k-1 whose values are
                            in the accumulated assignment become constraints.

This ensures the training distribution matches inference exactly:
  * Constraints are always circuit-consistent (from real simulation).
  * Constraint density grows naturally from nearly-zero to heavily-constrained
    as the solver progresses through deeper/later pairs.
  * No random or conflicting constraint values.

Usage
-----
    conda activate deepgate
    python scripts/build_fault_dataset.py \\
        --bench_dirs data/bench/ISCAS85 data/bench/iscas89 \\
        --output /home/local1/cache-cw/fault_dataset.pkl \\
        --max_faults 0 \\
        --sim_attempts 10 \\
        --workers 1

Output pickle format (list of dicts, one per sample)
------------------------------------------------------
    {
        "file":         str,                # path to .bench file
        "circuit_emb":  str,                # path to per-circuit embedding cache (.pt)
        "info": {
            "start":  int,
            "reconv": int,
            "paths":  [[int, ...], [int, ...]],
        },
        "constraints":  {node_id: 0|1},     # pre-solved nodes visible to this pair
        "labels":       {node_id: 0|1},     # ground-truth for every path node
        "solvability":  0,                  # 0=SAT, 1=UNSAT
        "solvability_convention": "0_sat_1_unsat",
        "fault_id":     int,
        "fault_val":    int,
        "pattern_id":   int,
        "sample_id":    int,
    }

The per-circuit embedding file (circuit_emb field) is a .pt file containing:
    {
        "struct_emb":   Tensor [N, 128],
        "gate_mapping": {gate_id: row_idx},
    }
This avoids embedding duplication across the thousands of samples from the
same circuit.

The dataset loader (ReconvergentPathsDataset) reads this split format when
the sample dict does NOT contain the "struct_emb" key directly (new format
detection), loading embeddings on demand from "circuit_emb".
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import random
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.atpg.logic_sim_three import logic_sim, reset_gates
from src.atpg.podem import get_all_faults
from src.atpg.recursive_reconv_solver import HierarchicalReconvSolver
from src.ml.data.embedding import EmbeddingExtractor
from src.util.io import parse_bench_file
from src.util.struct import GateType, LogicValue

# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------


def _build_topo_order(circuit, total_gates: int) -> List[int]:
    """Return gates in topological order (PIs first, POs last) via Kahn's algo."""
    in_degree = [0] * (total_gates + 1)
    for i in range(1, total_gates + 1):
        g = circuit[i]
        if g is None:
            continue
        for fin in g.fin:
            in_degree[i] += 1

    queue = [i for i in range(1, total_gates + 1) if in_degree[i] == 0 and circuit[i]]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        g = circuit[node]
        if g is None:
            continue
        for fout in g.fot:
            if 1 <= fout <= total_gates:
                in_degree[fout] -= 1
                if in_degree[fout] == 0:
                    queue.append(fout)
    return order


def _populate_fanouts(circuit, total_gates: int) -> None:
    """Ensure every gate has a .fot list populated from fin relationships."""
    for i in range(total_gates + 1):
        g = circuit[i]
        if g is not None and not hasattr(g, "fot"):
            g.fot = []

    for gate_id in range(1, total_gates + 1):
        g = circuit[gate_id]
        if g is None:
            continue
        for fin_id in g.fin:
            if 0 < fin_id <= total_gates and circuit[fin_id] is not None:
                if gate_id not in circuit[fin_id].fot:
                    circuit[fin_id].fot.append(gate_id)


# ---------------------------------------------------------------------------
# Random simulation to get a consistent full-cone assignment
# ---------------------------------------------------------------------------


def _random_sim(circuit, total_gates: int, topo_order: List[int]) -> Dict[int, int]:
    """Assign random 0/1 to PIs, forward-simulate, return {gate_id: 0_or_1}."""
    reset_gates(circuit, total_gates)
    for i in range(1, total_gates + 1):
        g = circuit[i]
        if g and g.type == GateType.INPT:
            g.val = random.choice([LogicValue.ZERO, LogicValue.ONE])
    logic_sim(circuit, total_gates, fault=None, topo_order=topo_order)
    return {
        i: int(circuit[i].val)
        for i in range(1, total_gates + 1)
        if circuit[i] is not None and circuit[i].val in (LogicValue.ZERO, LogicValue.ONE)
    }


def _sim_until_target(
    circuit,
    total_gates: int,
    topo_order: List[int],
    target_gate: int,
    target_val: int,
    max_attempts: int,
) -> Optional[Dict[int, int]]:
    """Repeat random simulation until target_gate evaluates to target_val.

    Returns the assignment dict, or None if all attempts fail.
    """
    for _ in range(max_attempts):
        assignment = _random_sim(circuit, total_gates, topo_order)
        if assignment.get(target_gate) == target_val:
            return assignment
    return None


def _sim_patterns_until_target(
    circuit,
    total_gates: int,
    topo_order: List[int],
    target_gate: int,
    target_val: int,
    max_attempts: int,
    patterns_per_fault: int,
) -> List[Dict[int, int]]:
    """Collect up to ``patterns_per_fault`` concrete PI patterns for one fault."""
    patterns: List[Dict[int, int]] = []
    seen_pi_signatures: Set[Tuple[Tuple[int, int], ...]] = set()
    for _ in range(max_attempts):
        assignment = _random_sim(circuit, total_gates, topo_order)
        if assignment.get(target_gate) != target_val:
            continue
        pi_sig = tuple(
            sorted(
                (i, assignment[i])
                for i in range(1, total_gates + 1)
                if circuit[i] is not None and circuit[i].type == GateType.INPT and i in assignment
            )
        )
        if pi_sig in seen_pi_signatures:
            continue
        seen_pi_signatures.add(pi_sig)
        patterns.append(assignment)
        if len(patterns) >= patterns_per_fault:
            break
    return patterns


# ---------------------------------------------------------------------------
# Pair ordering (mirrors HierarchicalReconvSolver._collect_and_sort_pairs)
# ---------------------------------------------------------------------------


def _collect_sorted_pairs(
    solver: HierarchicalReconvSolver, target_node: int
) -> List[Dict[str, Any]]:
    """Return reconvergent pairs in the same order the solver would process them."""
    pairs_by_reconv = solver._collect_and_sort_pairs(target_node)
    flat: List[Dict[str, Any]] = []
    for pairs in pairs_by_reconv.values():
        flat.extend(pairs)

    # Re-sort with same key as solver to guarantee identical ordering
    cone_nodes = solver._get_transitive_fanin(target_node)
    from collections import deque

    distances: Dict[int, int] = {target_node: 0}
    q: deque = deque([target_node])
    while q:
        curr = q.popleft()
        curr_dist = distances[curr]
        g = solver.circuit[curr]
        if g is None:
            continue
        for fin in g.fin:
            if fin in cone_nodes and fin not in distances:
                distances[fin] = curr_dist + 1
                q.append(fin)

    def pair_cost(p: Dict[str, Any]) -> Tuple[int, int]:
        reconv_node = p["reconv"]
        dist_to_target = distances.get(reconv_node, 9999)
        total_path_len = len(p["paths"][0]) + len(p["paths"][1])
        return (total_path_len + dist_to_target, total_path_len)

    flat.sort(key=pair_cost)
    return flat


# ---------------------------------------------------------------------------
# Per-circuit dataset builder
# ---------------------------------------------------------------------------


def _path_nodes(pair: Dict[str, Any]) -> Set[int]:
    """Return the set of all gate IDs appearing on any path in the pair."""
    nodes: Set[int] = set()
    for path in pair["paths"]:
        nodes.update(path)
    return nodes


def build_samples_for_circuit(
    bench_path: str,
    max_faults: int,
    sim_attempts: int,
    emb_cache_path: str,
    patterns_per_fault: int = 1,
    unsat_ratio: float = 0.0,
    max_samples_per_circuit: int = 0,
) -> List[Dict[str, Any]]:
    """Generate all training samples for one circuit file.

    Returns a list of sample dicts (see module docstring for format).
    The embeddings are NOT embedded inline; the sample stores the path to
    the per-circuit embedding cache file instead.
    """
    circuit, total_gates = parse_bench_file(bench_path)
    _populate_fanouts(circuit, total_gates)
    topo_order = _build_topo_order(circuit, total_gates)

    faults = get_all_faults(circuit, total_gates)
    if max_faults > 0:
        random.shuffle(faults)
        faults = faults[:max_faults]

    # Build a solver for pair discovery (no predictor needed — we only use
    # the topology helpers, never the predict() path).
    solver = HierarchicalReconvSolver(circuit, predictor=None, circuit_path=bench_path)

    samples: List[Dict[str, Any]] = []
    skipped_no_sim = 0
    skipped_no_pairs = 0
    sample_id = 0

    def make_unsat_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        labels = sample.get("labels") or {}
        if not labels:
            return None
        constraints = dict(sample.get("constraints") or {})
        # Prefer corrupting an already-visible constrained node; otherwise add a
        # contradictory visible constraint on one labeled path node.
        candidates = [n for n in constraints if n in labels] or list(labels.keys())
        if not candidates:
            return None
        corrupt_node = random.choice(candidates)
        constraints[corrupt_node] = 1 - int(labels[corrupt_node])
        unsat = dict(sample)
        unsat["constraints"] = constraints
        unsat["labels"] = {}
        unsat["solvability"] = 1
        unsat["is_unsat"] = True
        unsat["sample_id"] = int(sample["sample_id"]) + 1
        unsat["unsat_reason"] = "flipped_visible_constraint"
        return unsat

    for fault in faults:
        # Fault value: LogicValue.D means stuck-at-1, DB means stuck-at-0.
        # The "good" value we want to activate is the opposite of the stuck-at.
        target_val: int = (
            LogicValue.ONE if fault.value == LogicValue.D else LogicValue.ZERO
        )

        assignments = _sim_patterns_until_target(
            circuit,
            total_gates,
            topo_order,
            fault.gate_id,
            target_val,
            sim_attempts,
            max(1, patterns_per_fault),
        )
        if not assignments:
            skipped_no_sim += 1
            continue

        # Collect reconvergent pairs in solver order for this fault's gate
        pairs = _collect_sorted_pairs(solver, fault.gate_id)
        if not pairs:
            skipped_no_pairs += 1
            continue

        for pattern_id, assignment in enumerate(assignments):
            # Walk pairs, accumulating constraints exactly like inference.
            constraint_accumulator: Dict[int, int] = {fault.gate_id: target_val}

            for pair in pairs:
                path_nodes = _path_nodes(pair)

                constraints = {
                    n: constraint_accumulator[n] for n in path_nodes if n in constraint_accumulator
                }
                labels = {n: assignment[n] for n in path_nodes if n in assignment}

                if not labels:
                    continue

                sat_sample = {
                    "file": bench_path,
                    "circuit_emb": emb_cache_path,
                    "info": {
                        "start": pair["start"],
                        "reconv": pair["reconv"],
                        "paths": [list(p) for p in pair["paths"]],
                    },
                    "constraints": constraints,
                    "labels": labels,
                    "solvability": 0,
                    "solvability_convention": "0_sat_1_unsat",
                    "fault_id": int(fault.gate_id),
                    "fault_val": int(fault.value),
                    "pattern_id": int(pattern_id),
                    "sample_id": int(sample_id),
                }
                samples.append(sat_sample)
                sample_id += 1

                if unsat_ratio > 0.0 and random.random() < unsat_ratio:
                    unsat_sample = make_unsat_sample(sat_sample)
                    if unsat_sample is not None:
                        unsat_sample["sample_id"] = int(sample_id)
                        samples.append(unsat_sample)
                        sample_id += 1

                constraint_accumulator.update(labels)

                if max_samples_per_circuit > 0 and len(samples) >= max_samples_per_circuit:
                    break
            if max_samples_per_circuit > 0 and len(samples) >= max_samples_per_circuit:
                break
        if max_samples_per_circuit > 0 and len(samples) >= max_samples_per_circuit:
            break

    tqdm.write(
        f"  {os.path.basename(bench_path)}: {len(faults)} faults → "
        f"{len(samples)} samples "
        f"(skipped: {skipped_no_sim} no-sim, {skipped_no_pairs} no-pairs)"
    )
    return samples


# ---------------------------------------------------------------------------
# Embedding extraction (per-circuit, cached)
# ---------------------------------------------------------------------------


def _ensure_embedding_cache(bench_path: str, extractor: EmbeddingExtractor) -> str:
    """Extract (or load from cache) DeepGate embeddings for one circuit.

    Returns the path to the .pt cache file containing struct_emb + gate_mapping.
    """
    import torch

    # Determine cache path (mirrors EmbeddingExtractor internal logic)
    sibling_cache_dir = os.path.join(os.path.dirname(bench_path), ".deepgate_cache")
    try:
        os.makedirs(sibling_cache_dir, exist_ok=True)
        cache_path = os.path.join(
            sibling_cache_dir, os.path.basename(bench_path) + ".full.pt"
        )
    except PermissionError:
        rel = os.path.relpath(os.path.dirname(bench_path))
        fallback_dir = os.path.join("data", ".deepgate_cache", rel)
        os.makedirs(fallback_dir, exist_ok=True)
        cache_path = os.path.join(fallback_dir, os.path.basename(bench_path) + ".full.pt")

    if os.path.exists(cache_path):
        return cache_path

    # Cache miss — run DeepGate
    tqdm.write(f"  Extracting embeddings for {os.path.basename(bench_path)} ...")
    struct_emb, _func_emb, gate_mapping, _circuit = extractor.extract_embeddings(bench_path)
    torch.save({"struct_emb": struct_emb.cpu(), "gate_mapping": gate_mapping}, cache_path)
    return cache_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def collect_bench_files(bench_dirs: List[str]) -> List[str]:
    files: List[str] = []
    for d in bench_dirs:
        if os.path.isfile(d) and d.endswith(".bench"):
            files.append(d)
        elif os.path.isdir(d):
            for fname in sorted(os.listdir(d)):
                if fname.endswith(".bench"):
                    files.append(os.path.join(d, fname))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fault-driven reconvergent-path dataset."
    )
    parser.add_argument(
        "--bench_dirs",
        nargs="+",
        required=True,
        help="Directories (or individual .bench files) to scan for circuits.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output path.  If it ends with .pkl a single file is written "
            "(may OOM for large benchmark sets).  Otherwise treated as a "
            "directory and one chunk_XXXXX.pkl is written per circuit — "
            "recommended for large runs."
        ),
    )
    parser.add_argument(
        "--max_faults",
        type=int,
        default=0,
        help="Max faults per circuit (0 = all faults).",
    )
    parser.add_argument(
        "--sim_attempts",
        type=int,
        default=20,
        help=(
            "Number of random simulation attempts per fault to find an "
            "assignment where the fault gate takes the desired value."
        ),
    )
    parser.add_argument(
        "--patterns-per-fault",
        type=int,
        default=1,
        help=(
            "Maximum distinct activating PI patterns to record per fault. "
            "Keep this small because samples scale as faults × patterns × path-pairs."
        ),
    )
    parser.add_argument(
        "--unsat-ratio",
        type=float,
        default=0.0,
        help=(
            "Probability of adding one corrupted-constraint UNSAT sample next to each SAT "
            "path-pair sample. Recommended starting range: 0.05-0.20."
        ),
    )
    parser.add_argument(
        "--max-samples-per-circuit",
        type=int,
        default=0,
        help="Hard cap on generated samples per circuit (0 = no cap).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    bench_files = collect_bench_files(args.bench_dirs)
    if not bench_files:
        print("No .bench files found. Check --bench_dirs.")
        sys.exit(1)

    # Decide output mode: single .pkl file vs streaming chunk directory
    single_file_mode = args.output.endswith(".pkl")
    if single_file_mode:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        print(f"Found {len(bench_files)} circuit(s). Output (single file): {args.output}")
    else:
        os.makedirs(args.output, exist_ok=True)
        print(f"Found {len(bench_files)} circuit(s). Output (chunk dir): {args.output}")

    extractor = EmbeddingExtractor()
    all_samples: List[Dict[str, Any]] = [] if single_file_mode else None  # type: ignore[assignment]
    total_samples = 0
    chunk_idx = 0

    circuit_pbar = tqdm(bench_files, desc="Circuits", unit="circuit")
    for bench_path in circuit_pbar:
        circuit_pbar.set_postfix_str(os.path.basename(bench_path))
        tqdm.write(f"\nProcessing: {bench_path}")
        try:
            emb_cache_path = _ensure_embedding_cache(bench_path, extractor)
        except Exception as e:
            tqdm.write(f"  [WARN] Embedding extraction failed for {bench_path}: {e}. Skipping.")
            continue

        try:
            samples = build_samples_for_circuit(
                bench_path=bench_path,
                max_faults=args.max_faults,
                sim_attempts=args.sim_attempts,
                emb_cache_path=emb_cache_path,
                patterns_per_fault=args.patterns_per_fault,
                unsat_ratio=args.unsat_ratio,
                max_samples_per_circuit=args.max_samples_per_circuit,
            )
        except Exception as e:
            tqdm.write(f"  [WARN] Sample generation failed for {bench_path}: {e}. Skipping.")
            continue

        total_samples += len(samples)
        circuit_pbar.set_postfix(samples=total_samples, last=len(samples))

        if single_file_mode:
            all_samples.extend(samples)
        else:
            # Cap each chunk at MAX_CHUNK_SAMPLES to keep pickle files loadable in RAM.
            MAX_CHUNK_SAMPLES = 2_000_000
            for sub_start in range(0, max(1, len(samples)), MAX_CHUNK_SAMPLES):
                sub = samples[sub_start : sub_start + MAX_CHUNK_SAMPLES]
                chunk_path = os.path.join(args.output, f"chunk_{chunk_idx:05d}.pkl")
                with open(chunk_path, "wb") as f:
                    pickle.dump(sub, f, protocol=pickle.HIGHEST_PROTOCOL)
                tqdm.write(
                    f"  Wrote {len(sub)} samples → {os.path.basename(chunk_path)}"
                )
                chunk_idx += 1

        del samples
        gc.collect()

    circuit_pbar.close()
    print(f"\nTotal samples: {total_samples}")

    if single_file_mode:
        print(f"Saving to {args.output} ...")
        with open(args.output, "wb") as f:
            pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"All chunks written to {args.output}/")

    print("Done.")

    try:
        extractor.cleanup()
    except Exception:
        pass


if __name__ == "__main__":
    main()
