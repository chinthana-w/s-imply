# AI on Classic 10s Timeout Faults Report

Created: 2026-07-06T18:27:39.278911+00:00

## Benchmark Structure
- Source pool: `data/bench/ITC99/reconv_pool_10000_seed20260629.json`
- Bench: `data/bench/ITC99/b17.bench`
- Classic first pass: simple-backtrace PODEM, `timeout=10s`, `max_backtracks=5000`.
- Timeout subset: `2544` faults where classic returned result code `2` (`TIMEOUT`).
- AI second pass: `checkpoints/reconv_solver_fix_20260511/best_model.pth`, `ai_timeout=10s`, `ai_solve_timeout=10s`, `candidate_count=8`, `ai_attempts=1`, `max_confidence_retries=3`.

## Coverage
| Metric | Count | Percentage |
| --- | ---: | ---: |
| Classic detected on full pool | `7448/10000` | `74.4800%` |
| Classic timed out | `2544/10000` | `25.4400%` |
| AI detected among classic timeouts | `649/2544` | `25.5110%` |
| Newly added AI coverage over full pool | `+649/10000` | `+6.4900%` |
| Combined classic + AI detected | `8097/10000` | `80.9700%` |


## Timing
| Metric | Value |
| --- | ---: |
| Classic timeout-subset summed time | `7.08h` (`25505.698s`) |
| AI summed time on timeout subset | `5.89h` (`21204.344s`) |
| AI mean / median / max per fault | `8.33s` / `8.92s` / `10.191s` |
| AI success mean / median time | `2.658s` / `1.418s` |
| AI precheck solve total | `2905.891s` |
| AI precheck sim total | `37.287s` |
| AI hint solve total | `2438.471s` |
| AI PODEM search total | `15822.695s` |

## Backtracks and Result Codes
- Classic timeout-subset backtracks: `566615` total.
- AI diagnostic backtracks: `387591` total, `7787` on AI-detected faults.
- AI diagnostic backtracks mean / median / max: `152.35` / `127.00` / `246`.
- AI precheck successes: `115`.
- AI result code counts: `{'0': 67, '1': 649, '2': 1810, 'None': 18}`.

## Interpretation
The AI method added `649` detections that classic PODEM did not produce under the 10 second limit. That is `25.5110%` coverage on the classic-timeout subset and `+6.4900%` absolute coverage on the original 10,000-fault pool. The combined flow raises coverage from `74.4800%` to `80.9700%`.

The timeout-only subset is intentionally hard: AI also failed most of it, and the summed AI runtime is larger than the classic timeout-subset wall time. The value shown here is therefore incremental coverage, not a speed win.

## Artifacts
- classic_summary: `docs/session_reports/20260706_classic10_timeout_ai_on_timeouts/classic_merged_summary.json`
- classic_per_fault: `docs/session_reports/20260706_classic10_timeout_ai_on_timeouts/classic_merged_per_fault.csv`
- timeout_fault_list: `docs/session_reports/20260706_classic10_timeout_ai_on_timeouts/classic_timeout_faults.json`
- ai_per_fault: `docs/session_reports/20260706_classic10_timeout_ai_on_timeouts/ai_on_classic_timeouts_per_fault.csv`
- merged_summary: `docs/session_reports/20260706_classic10_timeout_ai_on_timeouts/merged_summary.json`
- report: `docs/session_reports/20260706_classic10_timeout_ai_on_timeouts/detailed_ai_on_classic_timeouts_report.md`

## Run Notes
- AI shard logs reported `Failed to load reconv pair cache ... Ran out of input`; the benchmark recomputed reconvergence data during the AI pass.
