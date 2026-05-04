# Multi-Agent Orchestration Workflow

This repo uses a coordinator-first multi-agent workflow.  LangGraph is the
coordination layer: it routes work, records state, calls narrow repo tools, and
evaluates artifacts.  It is not given broad write or shell authority in v1.

## Roles

- **Planner/Router Agent**: decomposes goals, assigns the specialist owner, and
  defines validation gates.
- **ATPG Solver Agent**: owns PODEM, reconvergence, fallback behavior, and solver
  correctness tests.
- **ML Training Agent**: owns training, AMP/GradScaler behavior, dataset/model/loss
  integration, and checkpoint compatibility.
- **Benchmark/Experiment Agent**: owns run manifests, held-out gates, benchmark
  artifacts, and result claims.
- **Quality Reviewer Agent**: reviews diffs, tests, regressions, repo-rule
  compliance, and unsupported claims.
- **Docs/Results Agent**: updates `docs/project_summary.md` for material
  architecture/workflow changes or validated result improvements, updates
  `docs/paper_draft.tex` whenever the theoretical framework changes, and updates
  the canonical Notion project page for method/results documentation.

## Local Commands

Create a one-off task packet:

```bash
python -m src.orchestration.cli route "Fix GradScaler checkpoint compatibility"
```

Create a persistent runner ledger and agent handoff prompt:

```bash
python -m src.orchestration.cli dispatch "Fix GradScaler checkpoint compatibility"
```

This creates `runs/orchestration/<run_id>/` with:

- `state.json`: current status, packet, and event history.
- `task_packet.json`: the specialist routing packet.
- `agent_prompt.md`: the prompt to paste into the assigned specialist agent.
- `events.jsonl`: append-only progress events.
- `run_manifest.json`: created when the packet requires a manifest.

Show the latest run, or a specific run:

```bash
python -m src.orchestration.cli status
python -m src.orchestration.cli status <run_id>
```

List all runs:

```bash
python -m src.orchestration.cli list
```

Print the handoff prompt for the latest run, or a specific run:

```bash
python -m src.orchestration.cli prompt
python -m src.orchestration.cli prompt <run_id>
```

Record progress from an agent or human operator:

```bash
python -m src.orchestration.cli checkpoint <run_id> \
    "Implemented checkpoint loading compatibility" \
    --artifact src/ml/train.py \
    --artifact tests/test_orchestration.py
```

Mark a run blocked:

```bash
python -m src.orchestration.cli checkpoint <run_id> \
    "Blocked: pytest is missing from the deepgate environment" \
    --status blocked
```

Close a run:

```bash
python -m src.orchestration.cli complete <run_id> \
    "Focused checks passed and docs were updated"
```

Mark a failed run:

```bash
python -m src.orchestration.cli fail <run_id> \
    "Could not reproduce the benchmark artifact"
```

Launch one queued run through the default agent command:

```bash
python -m src.orchestration.cli launch <run_id>
```

By default this uses:

```bash
codex exec --cd <repo> --sandbox workspace-write --ask-for-approval never -
```

Override the command for one launch:

```bash
python -m src.orchestration.cli launch <run_id> \
    --agent-cmd "codex exec --cd /home/local1/chinthana/s-imply --sandbox workspace-write --ask-for-approval never -"
```

Or configure it once in the shell:

```bash
export S_IMPLY_AGENT_CMD="codex exec --cd /home/local1/chinthana/s-imply --sandbox workspace-write --ask-for-approval never -"
python -m src.orchestration.cli launch <run_id>
```

Launch queued runs oldest first:

```bash
python -m src.orchestration.cli worker --max-runs 3
```

Each launch writes:

- `agent_stdout.log`
- `agent_stderr.log`

The runner marks the run `running` before launch. If the agent command exits
with code `0`, the run is marked `completed` unless the agent already marked it
`blocked`, `failed`, or `completed`. Nonzero exits mark the run `failed`.

Summarize a benchmark artifact:

```bash
python -m src.orchestration.cli summarize-benchmark ai_success_subset_c1908.csv
```

Check whether theory-related changes include the paper draft:

```bash
python -m src.orchestration.cli check-theory-doc-sync \
    src/atpg/reconv_podem.py docs/paper_draft.tex
```

Check whether the Notion documentation target is configured:

```bash
python -m src.orchestration.cli check-notion-target \
    --target "Back Implication Prediction using Attention" \
    --format "canonical wiki page with method and results" \
    --audience "research engineers" \
    --sync-style notion_canonical
```

## Validation Policy

- Use the `deepgate` conda environment.
- Prefer `python -m` entrypoints for repo modules.
- Keep edits compatible with Ruff line length 100 and rules `E`, `F`, and `I`.
- Require run manifests for ML and benchmark jobs that use GPU/eval resources.
- Require artifact provenance before claiming result improvements.
- Keep `docs/paper_draft.tex` synchronized with theoretical-framework changes.
- Treat the Notion page `Back Implication Prediction using Attention` as the
  canonical method/results page.
- Update experiment steps in Notion as dated log entries with commands,
  artifacts, results, and next steps.

## Operating Loop

1. Dispatch the task and read the owner, owned files, validation plan, and
   expected artifacts.
2. Give `agent_prompt.md` to the assigned specialist agent.
3. Require the specialist to checkpoint after analysis, after implementation,
   after validation, and whenever blocked.
4. Keep benchmark, ML, and GPU/eval work tied to `run_manifest.json` before
   execution.
5. Close the run only after the validation plan is either completed or the
   exceptions are recorded in the event log.

## Automation Loop

For one task:

```bash
python -m src.orchestration.cli dispatch "Verify new training process and present results"
python -m src.orchestration.cli launch <run_id>
python -m src.orchestration.cli status <run_id>
```

For a queue:

```bash
python -m src.orchestration.cli dispatch "Fix GradScaler checkpoint compatibility"
python -m src.orchestration.cli dispatch "Summarize benchmark artifact ai_success_subset_c1908.csv"
python -m src.orchestration.cli worker --max-runs 2
```
