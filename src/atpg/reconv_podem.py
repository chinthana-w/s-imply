"""Reconvergent PODEM utilities.

This module currently focuses on finding reconvergent fanout structures using a
beam search over the circuit graph. A reconvergent structure is characterized by
two (or more) distinct branches leaving a start node S that eventually feed into
the same downstream node R. Identifying such structures is useful for ATPG
heuristics (e.g., selecting candidate objectives that increase observability or
expose difficult to control lines.)

Circuit representation:
    Each element in `circuit` is a `Gate` (see src.util.struct). Relevant fields:
        - name: string identifier (often its numeric id as string)
        - fin: list[int]  fanin node indices
        - fot: list[int]  fanout node indices
        - nfo: int        number of fanouts (len(fot))

Beam search strategy:
    For a start node S with >=2 fanouts, we spawn an initial path per direct
    fanout (S, fo). We then iteratively expand a frontier of paths while keeping
    only the top K (beam_width) according to a heuristic score designed to bias
    towards nodes that themselves branch and/or overlap with original fanouts.
    We record for each reached node the set of distinct first-branch fanouts of
    S that led to it. When a node R is reached via >=2 distinct first branches
    we report a reconvergent structure.

Heuristic score (higher is better):
    score = (branching_factor * 2) + overlap
    where branching_factor = nfo of current path end
          overlap = |fot(end) ∩ initial_fanouts(S)|

Returned structure:
    {
        'start': S,
        'reconv': R,
        'branches': [b1, b2, ...],  # first-level fanouts of S that reconverged
        'paths': {b1: [...], b2: [...], ...} # complete node id paths from S to R
    }
"""

from __future__ import annotations

import os
import pickle
import random
from typing import Dict, List, Optional, Any, Tuple, Set

from src.util.io import parse_bench_file
from src.util.struct import LogicValue, GateType
from src.ml.embedding_extractor import EmbeddingExtractor

MAX_RECONV_PATH_COUNT = 1000
# NOTE: Retained for compatibility with other modules (e.g., RL trainers).
# Dataset generation below no longer uses logic values, but other codepaths may
# still rely on this table.
FANIN_LUT = {
    GateType.AND: {
        LogicValue.ZERO: [LogicValue.ZERO, LogicValue.ONE],
        LogicValue.ONE: [LogicValue.ONE],
    },
    GateType.NAND: {
        LogicValue.ZERO: [LogicValue.ONE],
        LogicValue.ONE: [LogicValue.ZERO, LogicValue.ONE],
    },
    GateType.OR: {
        LogicValue.ZERO: [LogicValue.ZERO],
        LogicValue.ONE: [LogicValue.ZERO, LogicValue.ONE],
    },
    GateType.NOR: {
        LogicValue.ZERO: [LogicValue.ONE],
        LogicValue.ONE: [LogicValue.ZERO, LogicValue.ONE],
    },
    GateType.XOR: {
        LogicValue.ZERO: [LogicValue.ZERO, LogicValue.ONE],
        LogicValue.ONE: [LogicValue.ZERO, LogicValue.ONE],
    },
    GateType.XNOR: {
        LogicValue.ZERO: [LogicValue.ZERO, LogicValue.ONE],
        LogicValue.ONE: [LogicValue.ZERO, LogicValue.ONE],
    },
    GateType.BUFF: {
        LogicValue.ZERO: [LogicValue.ZERO],
        LogicValue.ONE: [LogicValue.ONE],
    },
    GateType.NOT: {
        LogicValue.ZERO: [LogicValue.ONE],
        LogicValue.ONE: [LogicValue.ZERO],
    },
}

"""Paths-only dataset generation below; logic justification utilities removed."""


def reconv_podem(circuit_path: str, output_idx: int, desired_output: int):
    """Entry point (stub) invoking reconvergent fanout finder.

    Parameters
    ----------
    circuit_path : str
        Path to .bench file.
    output_idx : int
        Target output gate index (unused placeholder for future PODEM logic).
    desired_output : int
        Desired logic value at the output (unused placeholder).
    """
    circuit, _ = parse_bench_file(circuit_path)
    info = pick_reconv_pair(circuit, beam_width=16, max_depth=25)
    if not info:
        print("[ERROR] No reconvergent paths found in the circuit.")
        return None
    return info


def pick_reconv_pair(
    circuit: List[Any],
    beam_width: int = 8,
    max_depth: int = 20,
) -> Optional[Dict[str, Any]]:
    """Beam-search for a reconvergent fanout structure.

    A reconvergent structure exists if there is a start node S with at least
    two fanouts such that two (or more) distinct first fanout branches from S
    eventually reach a common node R (R != S).

    Parameters
    ----------
    circuit : list[Gate]
        Gate list (indexable by integer id) produced by parse_bench_file.
    seed : int
        RNG seed for shuffling candidate start nodes (adds stochastic variety).
    beam_width : int
        Max number of frontier paths kept per expansion step.
    max_depth : int
        Maximum path length expansions (number of edge traversals) before
        abandoning a start node.

    Returns
    -------
    dict | None
        Reconvergent structure info or None if none found.
    """
    node_ids = list(range(1, len(circuit)))  # skip index 0 (often dummy)
    random.shuffle(node_ids)

    for s in node_ids:
        start_gate = circuit[s]
        fanouts: List[int] = getattr(start_gate, "fot", []) or []
        if len(fanouts) < 2:
            continue

        # Initialize one path per direct fanout.
        frontier: List[List[int]] = [[s, fo] for fo in fanouts]
        initial_fanouts = set(fanouts)

        # reached[node][first_branch] = path
        reached: Dict[int, Dict[int, List[int]]] = {}
        depth = 0

        while frontier and depth < max_depth:
            # Score current frontier and prune to beam width
            scored = []
            for path in frontier:
                last = path[-1]
                gate = circuit[last]
                branching = getattr(gate, "nfo", 0)
                last_fot = getattr(gate, "fot", []) or []
                overlap = len(set(last_fot) & initial_fanouts)
                score = branching * 2 + overlap
                scored.append((score, path))
            scored.sort(key=lambda x: x[0], reverse=True)
            frontier = [p for _, p in scored[:beam_width]]

            next_frontier: List[List[int]] = []
            for path in frontier:
                last = path[-1]
                if len(path) < 2:
                    continue  # should not happen
                first_branch = path[1]

                # Record arrival at 'last' via this first branch
                reached.setdefault(last, {})
                if first_branch not in reached[last]:
                    reached[last][first_branch] = path.copy()

                # Check reconvergence condition
                if len(reached[last]) >= 2 and last != s:
                    branches = list(reached[last].keys())
                    paths = [reached[last][b] for b in branches]
                    return {
                        "start": s,
                        "reconv": last,
                        "branches": branches,
                        "paths": paths,
                    }

                # Expand path
                for fo in getattr(circuit[last], "fot", []) or []:
                    if fo in path:  # prevent simple cycles
                        continue
                    next_frontier.append(path + [fo])

            frontier = next_frontier
            depth += 1

    return None


def find_all_reconv_pairs(
    circuit: List[Any],
    beam_width: int = 16,
    max_depth: int = 25,
    max_pairs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Enumerate reconvergent fanout structures in the circuit.

    This function explores all start nodes with at least two fanouts and uses a
    beam search to find all nodes that can be reached from two distinct first
    fanout branches. For each reconvergent node found, it emits every unique
    pair of first-branch paths that reach it.

    Notes
    -----
    - This is exhaustive with respect to the chosen beam search parameters. It
      doesn't guarantee global exhaustiveness if pruning eliminates some paths.

    Parameters
    ----------
    circuit : list[Gate]
        Circuit gate list.
    beam_width : int
        Maximum frontier width per expansion.
    max_depth : int
        Maximum path expansion depth from a start node.
    max_pairs : int, optional
        Optional cap on the number of reconvergent pairs to return.

    Returns
    -------
    list[dict]
        A list of reconvergent structures. Each entry has keys:
        - 'start': start node id
        - 'reconv': reconvergent node id
        - 'branches': [b1, b2] the two first fanouts from start
        - 'paths': [path1, path2] node id sequences from start to reconv
    """
    results: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, int, Tuple[int, int]]] = set()  # (start, reconv, (b1,b2))

    node_ids = list(range(1, len(circuit)))  # skip index 0

    for s in node_ids:
        start_gate = circuit[s]
        fanouts: List[int] = getattr(start_gate, "fot", []) or []
        if len(fanouts) < 2:
            continue

        # Initialize one path per direct fanout.
        frontier: List[List[int]] = [[s, fo] for fo in fanouts]
        initial_fanouts = set(fanouts)

        # reached[node][first_branch] = path
        reached: Dict[int, Dict[int, List[int]]] = {}
        depth = 0

        while frontier and depth < max_depth:
            # Score current frontier and prune to beam width
            scored = []
            for path in frontier:
                last = path[-1]
                gate = circuit[last]
                branching = getattr(gate, "nfo", 0)
                last_fot = getattr(gate, "fot", []) or []
                overlap = len(set(last_fot) & initial_fanouts)
                score = branching * 2 + overlap
                scored.append((score, path))
            scored.sort(key=lambda x: x[0], reverse=True)
            frontier = [p for _, p in scored[:beam_width]]

            next_frontier: List[List[int]] = []
            for path in frontier:
                last = path[-1]
                if len(path) < 2:
                    continue
                first_branch = path[1]

                # Record arrival at 'last' via this first branch
                reached.setdefault(last, {})
                if first_branch not in reached[last]:
                    reached[last][first_branch] = path.copy()

                # If reconvergence at 'last', emit all unique branch pairs
                if len(reached[last]) >= 2 and last != s:
                    branches = list(reached[last].keys())
                    # generate all unique pairs
                    for i in range(len(branches)):
                        for j in range(i + 1, len(branches)):
                            b1, b2 = branches[i], branches[j]
                            ordered_pair: Tuple[int, int] = (
                                (b1, b2) if b1 < b2 else (b2, b1)
                            )
                            key = (s, last, ordered_pair)
                            if key in seen:
                                continue
                            seen.add(key)
                            paths = [reached[last][b1], reached[last][b2]]
                            results.append(
                                {
                                    "start": s,
                                    "reconv": last,
                                    "branches": [b1, b2],
                                    "paths": paths,
                                }
                            )
                            if max_pairs is not None and len(results) >= max_pairs:
                                return results

                # Expand path
                for fo in getattr(circuit[last], "fot", []) or []:
                    if fo in path:  # prevent simple cycles
                        continue
                    next_frontier.append(path + [fo])

            frontier = next_frontier
            depth += 1

    return results


## Legacy logic-justification utilities removed (paths-only datasets).


def build_dataset(
    base_path: str,
) -> List[Dict[str, Any]]:
    """Build a dataset of reconvergent path pairs with structural embeddings.

    The resulting entries include the circuit file path, the reconvergent 
    structure info (start, reconv, branches, paths), and structural embeddings
    extracted from the AIG representation of the circuit.

    Parameters
    ----------
    base_path : str
        Directory containing .bench files.

    Returns
    -------
    list[dict]
        Dataset entries with keys:
        - 'file': str - path to circuit file
        - 'info': dict - reconvergent structure (start, reconv, branches, paths)
        - 'struct_emb': torch.Tensor - structural embeddings
        - 'gate_mapping': dict - mapping from original to AIG gate IDs
    """
    bench_files = [
        os.path.join(base_path, f)
        for f in os.listdir(base_path)
        if f.endswith(".bench")
    ]

    dataset: List[Dict[str, Any]] = []
    extractor = EmbeddingExtractor()
    
    for bench_file in bench_files:
        print(f"Processing {bench_file}...")
        circuit, _ = parse_bench_file(bench_file)

        # Extract embeddings for this circuit
        try:
            struct_emb, func_emb, gate_mapping, original_circuit = extractor.extract_embeddings(bench_file)
            print(f"  Extracted embeddings: struct_emb shape {struct_emb.shape}")
        except Exception as e:
            print(f"  [WARNING] Failed to extract embeddings: {e}")
            print(f"  Skipping {bench_file}")
            continue

        # Enumerate all reconvergent pairs (subject to beam/depth constraints)
        infos: List[Dict[str, Any]] = find_all_reconv_pairs(circuit, beam_width=16, max_depth=25)
        print(f"  Found {len(infos)} reconvergent path pairs")

        # Add each reconvergent pair as a separate dataset entry
        for info in infos:
            entry = {
                "file": bench_file,
                "info": info,
                "struct_emb": struct_emb,
                "gate_mapping": gate_mapping,
            }
            dataset.append(entry)

        print(
            f"  Extracted {len([d for d in dataset if d['file'] == bench_file])} samples"
        )

    return dataset


def save_dataset(dataset, output_path):
    """Save dataset to pickle file.

    Parameters
    ----------
    dataset : list[dict]
        Dataset to save.
    output_path : str
        Path to save the pickle file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {output_path} ({len(dataset)} entries)")


def load_dataset(dataset_path):
    """Load dataset from pickle file.

    Parameters
    ----------
    dataset_path : str
        Path to the pickle file.

    Returns
    -------
    list[dict]
        Loaded dataset.
    """
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    print(f"Dataset loaded from {dataset_path} ({len(dataset)} entries)")
    return dataset


if __name__ == "__main__":
    input_dir = "data/bench/ISCAS85/"
    output_path = "data/datasets/reconv_dataset.pkl"

    dataset = build_dataset(
        base_path=input_dir,
    )
    save_dataset(dataset, output_path)
