# Experiment Logs

This directory is intentionally committed. It contains the evidence used to review candidate changes:

- `metrics.json` — `evidence-response-v3` measurements; metrics written by the updated evaluator also include report-only mean episode return and terminal-action rates
- `trace.json` — current scenario-by-scenario belief and action record
- `model.pt` — saved model checkpoint
- `learning_curve.csv` — training return and belief-match history
- `selection/selection.json` — baseline-versus-candidate keep/discard report
- `trace.decisions.jsonl` — human review notes, where present

Files ending in `.pre-ambiguous-abstention-v2.json` preserve results from before ambiguous outcomes were treated consistently. Files ending in `.pre-evidence-response-v3.json` preserve results from before ignored evidence was separated from belief that moved but remained uncertain.

Do not compare scores across evaluation versions as if they used the same rules. Current comparisons must use `evaluation_contract_version: evidence-response-v3` on both baseline and candidate.

The separate `high-risk-belief-update-v1` experiment lives under `logs/high-risk-belief-update-v1/`. It has no `BeliefUpdateScore` selector. Each `seedN/` directory contains `training_schedule.json` plus `baseline/`, `arm3/`, and `arm4/`; v1 outputs carry `contract_id`, scenario-suite ID, and schedule provenance. Do not move those metrics or traces into the legacy comparison layout.

Comparison directories use this layout:

```text
seed7-arm4/
  baseline/
  candidate/
  selection/
```

The current Arm 4 summary and next workflow are in `../PROJECT_STATUS.md`.
