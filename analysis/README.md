# High-Risk Belief Update Suite v1 notebook

Upload `high_risk_belief_update_v1_modal_embedded.ipynb` alone to Modal Notebooks and run the cells from top to bottom. It embeds the v1 comparison summary, all 11 training schedules, and each arm's metrics, controlled-replay traces, and closed-loop traces. The notebook validates contract IDs and schedule hashes, shows replay and task results separately, and writes downloadable CSV summaries. It does not need the local repository or a separate archive.

The source comparison report is `arm3-arm4-high-risk-v1-report.md`; the original Conditions 3/4/6 notebooks below remain under `evidence-response-v3`.

# Conditions 3, 4, and 6 notebook

The easiest option is to upload **`conditions_3_4_6_modal_embedded.ipynb` alone** to [Modal Notebooks](https://modal.com/notebooks), then run the cells from top to bottom. The results are embedded in the notebook, so no separate data upload or Volume is required.

Alternatively, upload `conditions_3_4_6_modal.ipynb` together with `conditions_3_4_6_logs.zip`. The zip contains only the current `metrics.json`, `trace.json`, `learning_curve.csv`, and `selection/selection.json` files for the 11 matched seeds in Conditions 3, 4, and 6. Model checkpoints and archived scoring snapshots are excluded.

Both notebooks also run locally if this repository's `logs/` directory is available. The setup cell accepts an explicit `LOGS_DIR` or `ARCHIVE_PATH` when the files are in another location. Modal's notebook filesystem is ephemeral outside an attached Volume; download the output CSVs or save them to a Volume if they need to persist.

The separate v1 Arm 3 / Arm 4 comparison report is `arm3-arm4-high-risk-v1-report.md`. Its metrics and schedule provenance live under `logs/high-risk-belief-update-v1/`; they use `high-risk-belief-update-v1` and must not be combined with the notebook's `evidence-response-v3` scores.

The follow-up cost sensitivity report is `high-risk-belief-update-v1-cost-sensitivity.md`. The post-training held-out template expansion uses scenario suite `high-risk-belief-update-scenarios-v2`, reuses the original v1 checkpoints, and is reported in `high-risk-belief-update-v1-heldout-v2-report.md`.
