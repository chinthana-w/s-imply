# Review Notes

Run ID: 20260508T200729Z-make-iterative-improvements-to-the-train-test-pi
Owner: atpg_solver

Status: complete

Initial notes:
- Worktree contains existing changes outside ATPG scope; they are preserved.
- Artifacts are stored in this run-specific directory to avoid recreating deleted root files.

Changes:
- `ai_podem` now passes the caller-provided `max_backtracks` into propagation-only mode.
- `ai_podem` now passes the caller-provided `max_backtracks` into the clean fallback retry.
- `AIBacktracer` construction now receives `no_fallback` when used from `ai_podem`.
- Tests now cover those contracts and update the stale activation retry assertion to the current
  five-seed retry behavior.

Review notes:
- This is a pipeline contract fix, not a claim of broad benchmark improvement.
- The train/test wrapper's `MAX_BACKTRACKS` can now affect all `ai_podem` execution paths covered
  here instead of being silently forced to 2000 in two paths.
- No theory, architecture, or paper-draft updates were needed.
