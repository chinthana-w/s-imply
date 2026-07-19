# ITC99 10,000-Fault Reconvergent Pool AI vs Classic PODEM Report

Created: 2026-07-01T16:01:41.202798+00:00

## Input Pool
- Pool: `data/bench/ITC99/reconv_pool_10000_seed20260629.json`
- Bench: `data/bench/ITC99/b17.bench`
- Selected faults: `10000` random reconvergent faults
- Candidate reconvergent faults: `45796`
- Seed: `20260629`
- Selection SHA-256: `b632777c8d71ecd9f222be7378e706a91952f8d94c25466e6f124ffa87c73e32`
- Source `data/bench/ITC99/b17.bench`: `45796`/`64458` faults reconvergent

## Benchmark Configuration
- AI method: `checkpoints/reconv_solver_fix_20260511/best_model.pth`, `candidate_count=8`, `ai_attempts=1`, `candidate_seed_base=20260629`, `max_confidence_retries=3`.
- Classic method: `simple_backtrace` PODEM with no learned assignment model.
- Shared limits for comparison: `max_backtracks=5000`, `timeout=5s` for the PODEM search loop.
- AI resumed shards also used `ai_solve_timeout=10s` to bound learned assignment generation; classic has no equivalent model-solve stage.
- Both methods ran on the exact same `10,000` original fault indices from the same pool JSON.

## Headline Comparison
| Metric | AI-guided PODEM | Classic PODEM | Winner / Readout |
| --- | ---: | ---: | --- |
| Coverage | `5251/10000` = `52.51%` | `7293/10000` = `72.93%` | Classic +`2042` faults |
| Failures | `4749` | `2707` | Classic has `2042` fewer failures |
| Timeout result codes | `1724` | `2702` | Classic has `978` more TIMEOUT rows, but many more successes overall |
| Summed per-fault runtime | `29.07h` | `5.01h` | Classic `5.80x` lower compute time |
| Mean per-fault runtime | `10.464s` | `1.805s` | Classic faster on average |
| Median per-fault runtime | `7.966s` | `0.656s` | Classic faster at median |
| Max per-fault runtime | `50.558s` | `6.231s` | Classic has tighter tail |
| Backtracks total | `115633` diagnostic | `139870` classic | AI records fewer diagnostic backtracks, but metrics are not semantically identical |
| Mean backtracks | `11.56` diagnostic | `13.99` classic | AI diagnostic mean is lower; compare cautiously |
| Max backtracks | `91` | `77` | AI lower tail count, but with more timeouts/failures |

## Paired Fault Outcomes
- Both methods detected: `5138`
- AI-only detections: `113`
- Classic-only detections: `2155`
- Neither detected: `2594`
- Classic was faster on `9566` faults; AI was faster on `434` faults; ties `0`.
- Classic had fewer recorded backtracks on `1591` faults; AI had fewer on `1323` faults; ties `7086`.
- On faults both detected, mean AI time was `4.633s`; mean classic time was `0.624s`.
- On AI-only detections, mean AI time was `5.434s`.
- On classic-only detections, mean classic time was `0.570s`.

## Result Codes
### AI-guided PODEM
- `0` UNTESTABLE: `2940`
- `1` SUCCESS: `5251`
- `2` TIMEOUT: `1724`
- `None` NO_RESULT_RECORDED: `85`

### Classic PODEM
- `0` UNTESTABLE: `5`
- `1` SUCCESS: `7293`
- `2` TIMEOUT: `2702`

## Interpretation
Classic PODEM is the stronger method on this pool under the measured settings. It detects more faults and uses substantially less total runtime. Its raw TIMEOUT count is higher than the AI run's, but classic still converts far more faults to detections and has far fewer total failures. The AI-guided method still has a useful success mode: it finds `113` faults that classic misses, and many AI successes are low-backtrack direct-assignment cases. That makes the current model more useful as a candidate generator or fallback adjunct than as a replacement for classic PODEM.

The main AI weakness is reliability. The AI run has `1,724` TIMEOUT rows, `85` rows without explicit result codes, and one external watchdog timeout from an unbounded learned-solve call. Classic has no learned assignment stage and therefore avoids that operational risk. The AI method also spent `29.07h` summed per-fault time versus classic `5.01h`, so the learned stage is not yet paying for itself at this scale.

Backtracks need careful reading. Classic backtracks are ordinary PODEM backtracks. AI backtracks are diagnostic backtracks inside the AI-guided PODEM path, excluding model candidate generation. In this run, AI records lower total and mean diagnostic backtracks than classic records ordinary PODEM backtracks, but this is not a clean win: many hard AI cases fail by timeout or solve timeout instead of exhausting a deeper search. The max backtrack counts are also close enough that coverage and runtime are the more important conclusions.

## Practical Recommendation
- Keep classic PODEM as the primary reliable engine for this benchmark class.
- Use AI predictions selectively where confidence is high or where activation precheck succeeds quickly.
- Add an explicit classic fallback for AI timeout/no-result cases.
- Treat AI-only detections as valuable training/debugging targets: they identify cases where learned reconvergent reasoning may add coverage beyond classic heuristics.
- Before claiming backtrack reduction, compare only faults detected by both methods and report both classic backtracks and AI diagnostic backtracks side by side.

## Artifacts
- AI merged per-fault CSV: `docs/session_reports/20260629_itc99_reconv_pool_10000/merged_per_fault.csv`
- AI summary JSON: `docs/session_reports/20260629_itc99_reconv_pool_10000/merged_summary.json`
- Classic merged per-fault CSV: `docs/session_reports/20260629_itc99_reconv_pool_10000/classic/classic_merged_per_fault.csv`
- Classic summary JSON: `docs/session_reports/20260629_itc99_reconv_pool_10000/classic/classic_summary.json`
- Pool JSON: `data/bench/ITC99/reconv_pool_10000_seed20260629.json`
