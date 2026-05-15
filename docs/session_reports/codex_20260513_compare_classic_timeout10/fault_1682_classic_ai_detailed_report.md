# Fault 1682 Detailed Classic vs AI Pipeline Report

## Fault and Run Configuration

- Fault index: `1682`
- Gate: `8211`
- Fault value: `4` (`DB`)
- Gate type: `NOT` from the parsed circuit
- Circuit: `data/bench/ITC99/b17.bench`
- Total gates: `32229`
- Model: `checkpoints/reconv_solver_fix_20260511/best_model.pth`
- Classic timeout: `300s`
- AI timeout: `300s`
- Max backtracks: `5000`

## Executive Summary

- Classic result: `SUCCESS` in `14.1978s`.
- Classic completed `177` counted backtrack events and `363` recursive calls before stopping.
- Classic spent `13.5482s` in `539` full-circuit `logic_sim` calls.
- AI-guided result: `SUCCESS` in `0.2460s` after model solving took `0.2964s`.
- AI-guided PODEM completed `0` counted backtrack events and `9` recursive calls.
- AI activation precheck detected the fault: `False`; the hierarchical solver returned `24` hinted assignments, `7` of them on primary inputs. These PI values are not necessarily direct model outputs; they can be derived by recursive gate-level justification after the model constrains reconvergent/path nodes.

## Classic Pipeline

Classic used `simple_backtrace` with no model hints. Each recursive step selects an objective, backtraces it to a PI, assigns the desired PI value, runs full-circuit logic simulation, recurses, then tries the opposite value if the first branch is untestable. The counted `backtrack_count` only increments after both PI branches fail and that PI is reset to X.

- Result code: `1` (`SUCCESS`)
- Elapsed: `14.1978s`
- Backtracks counted: `177`
- Recursive calls: `363`
- Backtrace hops: `409`
- Logic simulation calls: `539`
- Logic simulation time: `13.5482s`

### Classic First Decisions

| # | Rec calls before | Backtracks before | Objective | Value | PI chosen | PI value |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 1 | 0 | 8211 (NOT) | 0 | 908 | 1 |
| 2 | 2 | 0 | 8211 (NOT) | 0 | 1048 | 1 |
| 3 | 3 | 0 | 7098 (NOT) | 1 | 1050 | 0 |
| 4 | 4 | 0 | 30598 (AND) | 1 | 907 | 1 |
| 5 | 5 | 0 | 7097 (NOR) | 1 | 1051 | 0 |
| 6 | 6 | 0 | 10006 (NOT) | 1 | 1052 | 0 |
| 7 | 7 | 0 | 10841 (NAND) | 1 | 1053 | 1 |
| 8 | 9 | 0 | 11783 (NAND) | 1 | 1054 | 1 |
| 9 | 11 | 0 | 12987 (NAND) | 1 | 1055 | 1 |
| 10 | 13 | 0 | 13879 (NAND) | 1 | 1056 | 1 |
| 11 | 15 | 0 | 14559 (NAND) | 1 | 1057 | 1 |
| 12 | 17 | 0 | 15574 (NAND) | 1 | 1058 | 1 |
| 13 | 19 | 0 | 16857 (NAND) | 1 | 1059 | 1 |
| 14 | 21 | 0 | 17686 (NAND) | 1 | 1060 | 1 |
| 15 | 23 | 0 | 18450 (NAND) | 1 | 1061 | 1 |
| 16 | 25 | 0 | 19183 (NAND) | 1 | 1062 | 1 |
| 17 | 27 | 0 | 19844 (NAND) | 1 | 1063 | 1 |
| 18 | 29 | 0 | 20661 (NAND) | 1 | 1064 | 1 |
| 19 | 31 | 0 | 21545 (NAND) | 1 | 1065 | 1 |
| 20 | 33 | 0 | 22386 (NAND) | 1 | 1066 | 1 |

### Classic Repeated Objectives

| Objective gate | Value | Count |
|---:|---:|---:|
| 8211 | 0 | 4 |
| 7097 | 1 | 4 |
| 10006 | 1 | 4 |
| 10841 | 1 | 4 |
| 11783 | 1 | 4 |
| 12987 | 1 | 4 |
| 13879 | 1 | 4 |
| 14559 | 1 | 4 |
| 15574 | 1 | 4 |
| 16857 | 1 | 4 |

### Classic Repeated PI Assignments

| PI gate | Value | Count |
|---:|---:|---:|
| 1051 | 0 | 4 |
| 1052 | 0 | 4 |
| 1053 | 1 | 4 |
| 1054 | 1 | 4 |
| 1055 | 1 | 4 |
| 1056 | 1 | 4 |
| 1057 | 1 | 4 |
| 1058 | 1 | 4 |
| 1059 | 1 | 4 |
| 1060 | 1 | 4 |

### Classic Profile Hotspots

```text
16894507 function calls (16894145 primitive calls) in 14.198 seconds
1    0.000    0.000   14.198   14.198 podem.py:136(podem)
363/1    0.003    0.000   14.149   14.149 podem.py:223(podem_recursion)
539    0.002    0.000   13.549    0.025 <stdin>:110(counted_logic_sim)
539    7.809    0.014   13.546    0.025 logic_sim_three.py:109(logic_sim)
16588803    5.735    0.000    5.735    0.000 logic_sim_three.py:25(compute_gate_value)
363    0.590    0.002    0.590    0.002 logic_sim_three.py:185(fault_is_at_po)
1    0.017    0.017    0.019    0.019 util.py:64(calculate_distance_to_primary_outputs)
1    0.013    0.013    0.017    0.017 util.py:39(get_topological_order)
1    0.011    0.011    0.012    0.012 util.py:13(calculate_distance_to_primary_inputs)
362    0.002    0.000    0.004    0.000 podem.py:348(get_objective)
185    0.001    0.000    0.003    0.000 <stdin>:118(traced_backtrace)
185    0.001    0.000    0.002    0.000 podem.py:302(simple_backtrace)
```

## AI-Guided Pipeline

AI first ran the hierarchical reconvergent solver/model for the target gate and activation value. The model predicts values on reconvergent path nodes; then `HierarchicalReconvSolver` recursively justifies those internal requirements through ordinary gate rules until it may reach primary inputs. The precheck applied only primary-input assignments from this final solver assignment and simulated once; for this fault, precheck did not detect. The benchmark then used the full solver assignment as backtrace hints inside PODEM, without doing a clean classic retry.

- Result code: `1` (`SUCCESS`)
- Model solve seed: `20262186`
- Model solve time: `0.2964s`
- PODEM elapsed after hints: `0.2460s`
- Assignment size: `24` gates
- Primary-input assignments in precheck: `7`
- Precheck detected fault: `False`
- Backtracks counted during AI-guided PODEM: `0` internal PODEM diagnostic, not an AI/model backtrack metric
- Recursive calls: `9`
- Backtrace hops: `31`
- Logic simulation calls: `8`
- Logic simulation time: `0.2020s`

### AI Assignment Sample

| Gate | Type | Value |
|---:|---|---:|
| 908 | INPT | 0 |
| 1047 | INPT | 0 |
| 1048 | INPT | 0 |
| 1049 | INPT | 0 |
| 1079 | INPT | 1 |
| 1080 | INPT | 1 |
| 1081 | INPT | 1 |
| 3207 | NOT | 1 |
| 3230 | NAND | 1 |
| 3241 | NAND | 1 |
| 3242 | NAND | 1 |
| 3723 | NAND | 0 |
| 3724 | NOT | 0 |
| 3727 | NOT | 0 |
| 3728 | NAND | 1 |
| 4372 | NAND | 1 |
| 4374 | NAND | 0 |
| 4375 | NAND | 1 |
| 4471 | NAND | 1 |
| 4472 | NAND | 0 |
| 5211 | NAND | 1 |
| 6133 | NAND | 0 |
| 7095 | OR | 1 |
| 8211 | NOT | 0 |

### AI-Guided First Decisions

| # | Rec calls before | Backtracks before | Objective | Value | PI chosen | PI value |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 1 | 0 | 8211 (NOT) | 0 | 908 | 0 |
| 2 | 2 | 0 | 8211 (NOT) | 0 | 1079 | 1 |
| 3 | 3 | 0 | 7098 (NOT) | 1 | 1082 | 0 |
| 4 | 4 | 0 | 7098 (NOT) | 1 | 1080 | 0 |
| 5 | 5 | 0 | 30598 (AND) | 1 | 907 | 1 |
| 6 | 6 | 0 | 30598 (AND) | 1 | 1277 | 0 |
| 7 | 7 | 0 | 30598 (AND) | 1 | 906 | 0 |
| 8 | 8 | 0 | 30598 (AND) | 1 | 1110 | 1 |

### AI-Guided Repeated Objectives

| Objective gate | Value | Count |
|---:|---:|---:|
| 30598 | 1 | 4 |
| 8211 | 0 | 2 |
| 7098 | 1 | 2 |

### AI-Guided Profile Hotspots

```text
406712 function calls (406704 primitive calls) in 0.246 seconds
1    0.000    0.000    0.246    0.246 podem.py:136(podem)
9/1    0.000    0.000    0.217    0.217 podem.py:223(podem_recursion)
8    0.000    0.000    0.202    0.025 <stdin>:204(counted_logic_sim)
8    0.114    0.014    0.202    0.025 logic_sim_three.py:109(logic_sim)
246216    0.088    0.000    0.088    0.000 logic_sim_three.py:25(compute_gate_value)
1    0.014    0.014    0.017    0.017 util.py:64(calculate_distance_to_primary_outputs)
9    0.014    0.002    0.014    0.002 logic_sim_three.py:185(fault_is_at_po)
1    0.010    0.010    0.012    0.012 util.py:13(calculate_distance_to_primary_inputs)
8    0.000    0.000    0.000    0.000 <stdin>:214(traced_ai_backtrace)
8    0.000    0.000    0.000    0.000 benchmark_itc99_gate.py:78(__call__)
```

## Interpretation

With the timeout raised to 300s, classic no longer fails: it succeeds in `14.1978s`. The earlier 10s run timed out because it was close to the eventual solution but had not completed enough recursive work before the wall-clock cutoff. The profile shows that runtime is dominated by repeated full-circuit `logic_sim` calls on b17, not by a large number of completed backtrack events. Therefore the original 10s report showing timeout with fewer than 200 counted backtracks was not contradictory; it reflected a narrow completed-backtrack counter plus expensive simulation per recursive step.

For this fault, AI still wins decisively by giving PODEM a useful activation/justification hint path. The precheck alone is insufficient, but the hinted PODEM search reaches success quickly with zero completed backtrack events.

## Artifacts

- Raw profile JSON: `docs/session_reports/codex_20260513_compare_classic_timeout10/fault_1682_classic_ai_profile.json`
- Report: `docs/session_reports/codex_20260513_compare_classic_timeout10/fault_1682_classic_ai_detailed_report.md`
