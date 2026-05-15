# ITC99 Gate Compare-Classic Summary

## Run

- Run ID: `codex_20260513_compare_classic_timeout10`
- Model: `checkpoints/reconv_solver_fix_20260511/best_model.pth`
- Fault list: `data/bench/ITC99/b17_gate_10pct_faults.json`
- Device: `cuda`
- AI timeout: `10.0` seconds per fault
- Classic timeout: `10.0` seconds per fault
- Max backtracks: `5000`
- Compare flags: `--compare-classic --backtrack-target`

## Result

- AI no-fallback coverage: `5171/6445` = `80.2327%`
- Coverage target: `80.00%`; pass=`True`
- Classic solved: `5170/6445`
- AI solved faults included in ranking: `5171`
- Classic backtracks on AI-solved faults: total `17687`, median `0.0`, max `170`
- AI time on AI-solved faults: total `5606.87s`, median `0.8589s`
- Classic time on AI-solved faults: total `3480.06s`, median `0.4264s`
- Full AI timed section: `19065.55s`; full classic timed section: `16092.97s`

## Backtrack Semantics

AI does not have a comparable backtrack count. The ranking below uses only classic PODEM backtracks for faults that AI solved. Any `ai_backtracks` field in legacy JSON/CSV output is an internal PODEM search diagnostic, not an AI/model backtrack metric.

Classic `classic_backtracks` is also not a wall-clock effort counter. In `src/atpg/podem.py`, it increments only when PODEM has tried both values for a PI decision, resets that PI to X, and returns `UNTESTABLE` from that subtree. The timeout is checked at recursion entry, while each recursive step can spend substantial time in full-circuit simulation and D-frontier rebuilds. On b17, a single suspicious classic timeout profile for fault index `1682` spent about `9.50s` of `10.01s` inside `logic_sim`, with only a few hundred recursive calls. Therefore a classic timeout with fewer than 200 counted backtracks is plausible: the run timed out on expensive simulation work, not on a high completed-backtrack count.

The exact classic backtrack count for timeout rows is wall-clock dependent and should be treated as "completed backtrack events before timeout", not as a deterministic exhaustive-search size.

## Top AI-Solved Faults by Classic Backtracks

| Rank | Fault index | Gate | Fault val | Classic backtracks | Classic time (s) | AI time (s) | Classic solved |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 1682 | 8211 | 4 | 170 | 10.4357 | 0.2595 | False |
| 2 | 1366 | 6641 | 3 | 159 | 10.0638 | 0.5400 | False |
| 3 | 1065 | 5196 | 3 | 154 | 10.0108 | 0.6429 | False |
| 4 | 1574 | 7622 | 3 | 154 | 10.0062 | 2.3394 | False |
| 5 | 1618 | 7864 | 4 | 154 | 9.6274 | 9.7433 | True |
| 6 | 972 | 4676 | 3 | 152 | 10.0151 | 4.5234 | False |
| 7 | 6080 | 30348 | 3 | 152 | 10.0151 | 0.8547 | False |
| 8 | 3091 | 15119 | 4 | 152 | 9.8852 | 9.9302 | True |
| 9 | 4391 | 21848 | 4 | 151 | 9.7712 | 0.5546 | True |
| 10 | 3701 | 18252 | 4 | 151 | 9.5528 | 9.7239 | True |
| 11 | 1490 | 7282 | 3 | 149 | 10.0048 | 0.7390 | False |
| 12 | 1300 | 6356 | 3 | 149 | 10.0022 | 0.7881 | False |
| 13 | 1498 | 7304 | 3 | 148 | 10.1258 | 0.7788 | False |
| 14 | 1489 | 7272 | 3 | 148 | 10.1236 | 0.9049 | False |
| 15 | 1520 | 7385 | 3 | 148 | 10.1190 | 0.7462 | False |
| 16 | 1478 | 7248 | 3 | 148 | 10.0109 | 0.9742 | False |
| 17 | 4805 | 23866 | 3 | 148 | 9.4152 | 10.2183 | True |
| 18 | 1481 | 7254 | 3 | 146 | 10.0066 | 0.7510 | False |
| 19 | 1317 | 6434 | 3 | 146 | 10.0050 | 0.6898 | False |
| 20 | 926 | 4431 | 4 | 143 | 9.1282 | 9.1225 | True |
| 21 | 1335 | 6497 | 3 | 142 | 10.0158 | 0.6826 | False |
| 22 | 2344 | 11640 | 3 | 142 | 8.9510 | 8.9955 | True |
| 23 | 6305 | 31458 | 4 | 142 | 8.9499 | 10.1715 | True |
| 24 | 1531 | 7427 | 3 | 141 | 10.0171 | 0.6506 | False |
| 25 | 1340 | 6517 | 3 | 139 | 10.0154 | 5.8939 | False |
| 26 | 1310 | 6401 | 3 | 138 | 10.0504 | 0.8705 | False |
| 27 | 927 | 4432 | 3 | 137 | 8.6843 | 8.7679 | True |
| 28 | 5217 | 25982 | 4 | 136 | 8.9729 | 9.3035 | True |
| 29 | 2590 | 12796 | 3 | 136 | 8.7628 | 8.6373 | True |
| 30 | 1479 | 7250 | 3 | 135 | 10.0404 | 0.7967 | False |

## Artifacts

- Full ranked CSV: `docs/session_reports/codex_20260513_compare_classic_timeout10/ai_solved_ranked_by_classic_backtracks.csv`
- Raw per-fault CSV: `docs/session_reports/codex_20260513_compare_classic_timeout10/itc99_gate_per_fault.csv`
- JSON report: `docs/session_reports/codex_20260513_compare_classic_timeout10/itc99_gate_report.json`
- Manifest: `docs/session_reports/codex_20260513_compare_classic_timeout10/itc99_gate_run_manifest.json`
- Notion-style summary: `docs/session_reports/codex_20260513_compare_classic_timeout10/notion_result_summary.md`
