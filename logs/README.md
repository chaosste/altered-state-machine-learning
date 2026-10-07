# Experiment Logs

This directory is intentionally committed. It contains the evidence used to review candidate changes:

- `metrics.json` — current `evidence-response-v3` measurements
- `trace.json` — current scenario-by-scenario belief and action record
- `model.pt` — saved model checkpoint
- `learning_curve.csv` — training return and belief-match history
- `selection/selection.json` — baseline-versus-candidate keep/discard report
- `trace.decisions.jsonl` — human review notes, where present

Files ending in `.pre-ambiguous-abstention-v2.json` preserve results from before ambiguous outcomes were treated consistently. Files ending in `.pre-evidence-response-v3.json` preserve results from before ignored evidence was separated from belief that moved but remained uncertain.

Do not compare scores across evaluation versions as if they used the same rules. Current comparisons must use `evaluation_contract_version: evidence-response-v3` on both baseline and candidate.

Comparison directories use this layout:

```text
seed7-arm4/
  baseline/
  candidate/
  selection/
```

The current Arm 4 summary and next workflow are in `../PROJECT_STATUS.md`.
