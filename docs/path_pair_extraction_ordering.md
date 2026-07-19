# Reconvergent Path Pair Extraction and Ordering

This document summarizes the algorithm used by
`HierarchicalReconvSolver._collect_and_sort_pairs()` to find reconvergent path
pairs for a target gate and order them for hierarchical justification.

## Goal

Given a target node `root_node`, the solver identifies reconvergent structures in
the logic cone that feeds that target. Each pair has the shape:

```text
start stem S
  branch 1: S -> ... -> R
  branch 2: S -> ... -> R
reconvergent node R
```

The result is a sorted list of dictionaries:

```python
{
    "start": S,
    "reconv": R,
    "branches": [first_fanout_1, first_fanout_2],
    "paths": [[S, ..., R], [S, ..., R]],
}
```

## Extraction Algorithm

The active solver implementation is in
`src/atpg/recursive_reconv_solver.py`.

1. **Build the target fan-in cone**

   `_get_transitive_fanin(root_node)` runs a backward BFS from the target through
   gate fan-ins. The current default depth limit is 20. This produces the
   allowed node set for the rest of the search, so pair discovery is scoped to
   structures that can affect the current objective.

2. **Find candidate stems**

   `_find_pairs_in_set(allowed_nodes)` scans nodes in the fan-in cone. A node is
   a candidate stem when it has at least two fanouts that also remain inside the
   allowed node set.

3. **Propagate branch reachability forward**

   For each stem `S`, the solver starts one forward search branch from each
   valid direct fanout of `S`.

   It maintains:

   ```python
   reached[node][branch_index] = path_from_S_to_node
   ```

   Here `branch_index` identifies which direct fanout of `S` the path used as
   its first step.

4. **Record reconvergences**

   During forward BFS, when a node `R` has been reached from two or more distinct
   first branches, the solver records one reconvergent pair:

   ```python
   {
       "start": S,
       "reconv": R,
       "branches": [fanout_for_branch_a, fanout_for_branch_b],
       "paths": [path_a, path_b],
   }
   ```

   The solver records only the first pair reported for each `(S, R)` during this
   stem search. The selected paths are the first two branch paths discovered by
   BFS propagation.

5. **Continue branch propagation**

   The BFS continues through fanouts inside the allowed cone. If a downstream
   node receives a branch index it has not seen before, the node is requeued so
   that the new reachability information can propagate farther.

## Ordering Algorithm

After extraction, `_collect_and_sort_pairs()` computes a distance map from the
target back through its fan-in cone:

```python
distances[root_node] = 0
distance[fanin] = distance[current] + 1
```

Each pair is assigned the sort key:

```python
total_path_len = len(pair["paths"][0]) + len(pair["paths"][1])
dist_to_target = distances.get(pair["reconv"], 9999)
cost = (total_path_len, dist_to_target)
```

Pairs are sorted in ascending lexicographic order. This means:

1. **Shorter total path pairs are solved first.**
2. **Among equal-length pairs, reconvergences closer to the target are solved
   first.**

This ordering gives the recursive solver smaller local regions before larger
ones, while preferring equal-sized regions that are immediately relevant to the
current target.

## Runtime Use

The sorted list is cached per target node in `pair_cache`. If a circuit path is
available, the topology cache can also be persisted on disk through
`src/atpg/reconv_cache.py`; reconvergent topology depends on the netlist, not on
the current fault or model checkpoint.

During `_backward_justify()`, the solver does not blindly consume every pair.
For each recursive step, it scans the sorted list and chooses the first unsolved
pair whose `reconv` node or `start` node is currently active in the justification
queue. That makes the static order a priority list, while the dynamic queue
keeps prediction just-in-time for the current backtrace.

## Dataset Consistency

`scripts/build_fault_dataset.py` mirrors this ordering in
`_collect_sorted_pairs()`. It flattens the solver output if necessary and
re-sorts pairs with the same `(total_path_len, dist_to_target)` key. This keeps
training samples aligned with the inference order used by
`HierarchicalReconvSolver`.

## Related Utility

`src/atpg/reconv_podem.py` also contains `find_all_reconv_pairs()`, a broader
beam-search enumerator. It scans all possible stems in a circuit, expands paths
forward with `beam_width` and `max_depth` limits, and emits unique pairs when a
node is reached from at least two first branches. That utility is useful for
standalone enumeration and older dataset/reporting flows, but the active
hierarchical solver path described above uses the target-cone extraction and
ordering implemented in `recursive_reconv_solver.py`.
