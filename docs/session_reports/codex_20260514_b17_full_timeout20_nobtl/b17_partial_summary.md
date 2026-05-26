# b17 Full-Fault Coverage Timing Summary

- CSV: `docs/session_reports/codex_20260514_b17_full_timeout20_nobtl/b17_full_per_fault.csv`
- Processed faults: `64458/64458` (100.00%)
- AI/system-mode coverage over processed faults: `13705/64458` = `21.2619%`
- AI/system-mode coverage lower bound over all expected faults: `13705/64458` = `21.2619%`
- Classic coverage over processed faults: `35583/64458` = `55.2034%`
- Target metric: AI/system mode must cover `80.0000%` of faults covered by classic PODEM; observed `13705/35583` = `38.5156%`, required `28467`, pass=`False`

## Classic Result Codes

| Result | Count |
|---|---:|
| BACKTRACK_LIMIT | 23772 |
| SUCCESS | 35583 |
| TIMEOUT | 4988 |
| UNTESTABLE | 115 |

## Timing Breakdown

| Segment | Total s | Mean s | Median s | P90 s | P99 s | Max s |
|---|---:|---:|---:|---:|---:|---:|
| AI total per fault | 152853.79 | 2.3714 | 0.7465 | 2.9500 | 20.4771 | 24.1121 |
| AI precheck solver | 21003.33 | 0.3258 | 0.0972 | 0.8817 | 1.8929 | 7.5881 |
| AI precheck simulation | 231.64 | 0.0036 | 0.0000 | 0.0317 | 0.0340 | 0.0519 |
| AI hint solver | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| AI-guided PODEM search | 24930.33 | 0.3868 | 0.0000 | 0.4377 | 20.0017 | 20.2331 |
| Classic total per fault | 128885.44 | 1.9995 | 0.2881 | 1.8975 | 20.0532 | 20.5913 |

## Slowest AI Faults

| Fault index | Gate | Fault val | AI ok | AI time s | Classic ok | Classic time s |
|---:|---:|---:|:---|---:|:---|---:|
| 25682 | 12842 | 3 | False | 24.1121 | False | 20.0172 |
| 25704 | 12853 | 3 | False | 23.9527 | False | 20.0146 |
| 22489 | 11245 | 4 | False | 23.5700 | False | 20.0038 |
| 21390 | 10696 | 3 | False | 23.4969 | False | 20.0180 |
| 21480 | 10741 | 3 | False | 22.7636 | False | 20.0124 |
| 26152 | 13077 | 3 | False | 22.6305 | False | 20.0184 |
| 26154 | 13078 | 3 | False | 22.5479 | False | 20.0175 |
| 22497 | 11249 | 4 | False | 22.5312 | False | 20.0050 |
| 23396 | 11699 | 3 | False | 22.1932 | False | 20.0167 |
| 25640 | 12821 | 3 | False | 22.1880 | False | 20.0280 |
| 22459 | 11230 | 4 | False | 22.1778 | False | 20.0607 |
| 22484 | 11243 | 3 | False | 22.1760 | False | 20.0006 |
| 25618 | 12810 | 3 | False | 22.1428 | False | 20.0029 |
| 22486 | 11244 | 3 | False | 22.0982 | False | 20.0118 |
| 26792 | 13397 | 3 | False | 22.0961 | False | 20.0010 |
| 26821 | 13411 | 4 | False | 22.0946 | False | 20.0282 |
| 26799 | 13400 | 4 | False | 22.0895 | False | 20.0170 |
| 22378 | 11190 | 3 | False | 21.9208 | False | 20.0396 |
| 19654 | 9828 | 3 | False | 21.9160 | False | 20.0090 |
| 23292 | 11647 | 3 | False | 21.8785 | False | 20.0069 |

## Slowest Classic Faults

| Fault index | Gate | Fault val | Classic result | Classic backtracks | Classic recursive calls | Classic time s | AI ok | AI time s |
|---:|---:|---:|---|---:|---:|---:|:---|---:|
| 25815 | 12908 | 4 | TIMEOUT | 331 | 697 | 20.5913 | False | 20.9118 |
| 25840 | 12921 | 3 | TIMEOUT | 343 | 705 | 20.5551 | False | 21.2507 |
| 23599 | 11800 | 4 | TIMEOUT | 335 | 711 | 20.5118 | False | 20.7894 |
| 19607 | 9804 | 4 | TIMEOUT | 337 | 702 | 20.4998 | False | 21.2385 |
| 12411 | 6206 | 4 | TIMEOUT | 357 | 745 | 20.4978 | False | 20.5659 |
| 22837 | 11419 | 4 | TIMEOUT | 340 | 707 | 20.4772 | False | 20.8392 |
| 25845 | 12923 | 4 | TIMEOUT | 338 | 705 | 20.4660 | False | 21.1383 |
| 23566 | 11784 | 3 | TIMEOUT | 335 | 682 | 20.4550 | False | 20.7412 |
| 20316 | 10159 | 3 | TIMEOUT | 341 | 707 | 20.4090 | False | 20.7259 |
| 20332 | 10167 | 3 | TIMEOUT | 340 | 707 | 20.3899 | False | 20.7490 |
| 17885 | 8943 | 4 | TIMEOUT | 334 | 709 | 20.3885 | False | 20.7857 |
| 22658 | 11330 | 3 | TIMEOUT | 337 | 720 | 20.3873 | False | 20.6689 |
| 20446 | 10224 | 3 | TIMEOUT | 324 | 718 | 20.3747 | False | 20.4827 |
| 23818 | 11910 | 3 | TIMEOUT | 335 | 706 | 20.3682 | False | 20.7093 |
| 20450 | 10226 | 3 | TIMEOUT | 324 | 719 | 20.3679 | False | 20.4612 |
| 22877 | 11439 | 4 | TIMEOUT | 324 | 719 | 20.3655 | False | 20.5413 |
| 20494 | 10248 | 3 | TIMEOUT | 324 | 718 | 20.3626 | False | 20.4605 |
| 20498 | 10250 | 3 | TIMEOUT | 324 | 719 | 20.3463 | False | 20.4366 |
| 18028 | 9015 | 3 | TIMEOUT | 319 | 729 | 20.3454 | False | 20.8588 |
| 21799 | 10900 | 4 | TIMEOUT | 323 | 714 | 20.3451 | False | 20.4691 |

## Highest Classic Backtrack Counts

| Fault index | Gate | Fault val | Classic result | Classic backtracks | Classic time s | AI ok | AI time s |
|---:|---:|---:|---|---:|---:|:---|---:|
| 12411 | 6206 | 4 | TIMEOUT | 357 | 20.4978 | False | 20.5659 |
| 3345 | 1673 | 4 | TIMEOUT | 351 | 20.0146 | False | 20.0146 |
| 3344 | 1673 | 3 | TIMEOUT | 351 | 20.0022 | False | 20.0022 |
| 15430 | 7716 | 3 | TIMEOUT | 350 | 20.0336 | False | 20.0735 |
| 26213 | 13107 | 4 | TIMEOUT | 349 | 20.1658 | False | 20.5653 |
| 14370 | 7186 | 3 | TIMEOUT | 349 | 20.0133 | False | 20.0962 |
| 7973 | 3987 | 4 | TIMEOUT | 348 | 20.0834 | False | 20.1383 |
| 14597 | 7299 | 4 | TIMEOUT | 347 | 20.2053 | False | 20.2812 |
| 3027 | 1514 | 4 | TIMEOUT | 347 | 20.0311 | False | 20.0311 |
| 3026 | 1514 | 3 | TIMEOUT | 347 | 20.0294 | False | 20.0294 |
| 12735 | 6368 | 4 | TIMEOUT | 347 | 20.0130 | False | 20.0648 |
| 15432 | 7717 | 3 | TIMEOUT | 347 | 20.0012 | False | 20.0739 |
| 14087 | 7044 | 4 | TIMEOUT | 346 | 20.0378 | False | 20.1220 |
| 14509 | 7255 | 4 | TIMEOUT | 346 | 20.0336 | False | 20.1135 |
| 4630 | 2316 | 3 | TIMEOUT | 346 | 20.0037 | False | 20.0037 |
| 13839 | 6920 | 4 | TIMEOUT | 345 | 20.0878 | False | 20.1470 |
| 13755 | 6878 | 4 | TIMEOUT | 345 | 20.0744 | False | 20.0744 |
| 13831 | 6916 | 4 | TIMEOUT | 345 | 20.0698 | False | 20.1199 |
| 5272 | 2637 | 3 | TIMEOUT | 345 | 20.0500 | False | 20.0500 |
| 13751 | 6876 | 4 | TIMEOUT | 345 | 20.0497 | False | 20.0497 |
