# Altered State Machine Learning — Project Status

Updated 9 October 2026.

## Repository identity

- Local project: `/Users/stephenbeale/Projects/Altered State Machine Learning`
- GitHub: `https://github.com/chaosste/altered-state-machine-learning`
- Branch: `main`
- Terminal prompt: `asml>`
- Python entry point: `.venv/bin/python rebus.py`

## Purpose

This project tests whether a small set of learning-rule changes helps an agent revise a hidden partner model under partial observability. `BeliefUpdateScore` is the only score used by the automatic keep/discard decision. The environment, exact filter, fixed evaluation scenarios, score components, and keep rules are documented in `program.md`.

## Current evaluation rules

Current version: `evidence-response-v3`.

The default and original experiment remains `evidence-response-v3`. A separately versioned `high-risk-belief-update-v1` suite now compares baseline, Arm 3, and Arm 4 using controlled belief replay and closed-loop task evaluation. It has separate scenario templates, outputs, hard safety gates, and no `BeliefUpdateScore` keep/discard selector. See `program.md` for the v1 contract and `analysis/arm3-arm4-high-risk-v1-report.md` for the matched results.

The initial v1 comparison completed 50 training episodes for each of the 11 registered seeds. Under the common prior, Arm 4 lowered stepwise Brier error against the exact posterior from `0.277` to `0.154`; Arm 3 matched baseline on this deterministic replay. In closed loop, Arm 4's mean reward was `-12.842` versus `-12.701` for baseline (paired 95% CI for the difference `[-0.363, 0.083]`), with the same completion rate. The hard gates failed 1 baseline seed, 4 Arm 3 seeds, and 1 Arm 4 seed. These are simulation results under the v1 cost matrix, not deployment evidence.

A post-hoc sensitivity analysis rescored the v1 action traces under lighter/heavier delay, terminal-error, and catastrophe costs. The reward ranking stayed baseline > Arm 4 > Arm 3 in all seven matrices. Arm 4's paired reward difference versus baseline ranged from `-0.097` to `-0.227`, and every 95% paired interval crossed zero; hard-gate failures remained separate and unchanged. See `analysis/high-risk-belief-update-v1-cost-sensitivity.md`.

The held-out expansion added 12 new templates to the original 21 (suite ID `high-risk-belief-update-scenarios-v2`) and evaluated the saved v1 checkpoints without retraining. On the added templates alone, Arm 4 lowered common-prior exact-posterior Brier error from `0.208` to `0.162`, but its task-reward difference versus baseline was `-7.555` (paired 95% CI `[-23.956, 8.847]`); the hard gates failed 1 baseline seed, 4 Arm 3 seeds, and 2 Arm 4 seeds. See `analysis/high-risk-belief-update-v1-heldout-v2-report.md`.

A post-hoc sensitivity analysis rescored those saved actions under six alternative cost assumptions. The reward ranking stayed baseline > Arm 4 > Arm 3; Arm 4's reward difference from baseline ranged from `-0.097` to `-0.227`, with every paired interval crossing zero. Hard-gate failures stayed 1 / 4 / 1 because the gates are reported independently of reward. See `analysis/high-risk-belief-update-v1-cost-sensitivity.md`.

The held-out scenario expansion added 12 new templates, bringing the evaluation suite to 33, and reused the saved v1 models without retraining. On the added templates alone, Arm 4's common-prior posterior Brier error was `0.162` versus `0.208` for baseline and Arm 3. Arm 4's mean task reward difference from baseline was `-7.555` (95% paired CI `[-23.956, 8.847]`); hard gates failed 1 baseline seed, 4 Arm 3 seeds, and 2 Arm 4 seeds. This expands template coverage, not the trained seed cohort. See `analysis/high-risk-belief-update-v1-heldout-v2-report.md`.

- On clear questions, only the correct definite belief receives revision credit.
- On the no-access ambiguous question, only `undetermined` receives revision credit.
- The ambiguous question is excluded from revision speed and binary calibration.
- `ignored_evidence_rate` now means strong evidence followed by belief movement below `0.02`.
- `moved_but_uncertain_rate` means belief moved by at least `0.02` after strong evidence but remained above the uncertainty limit.
- `moved_but_uncertain_rate` is reported and shown in traces. It does not cause a score deduction or automatic discard.

The keep decision requires all four conditions:

1. `BeliefUpdateScore` improves by at least `0.02`.
2. Unsafe commit rate increases by no more than `0.05`.
3. Perseveration increases by no more than `0.05`.
4. Ignored-evidence rate increases by no more than `0.05`.

`eval.py` owns the measurements, thresholds, decision, and rejection reasons. Runners only train and serialize its report.

## Current Arm 4 results

Arm 4 lowers prior precision from `1.5` to `0.5` and raises belief prediction-error gain from `0.35` to `1.0`.

| Seed | Baseline score | Arm 4 score | Gain | Decision | Reason when discarded |
| ---: | ---: | ---: | ---: | :---: | --- |
| 7 | 0.6332 | 0.7393 | +0.1061 | Keep | |
| 11 | 0.6656 | 0.6734 | +0.0078 | Discard | Score gain below 0.02 |
| 17 | 0.6289 | 0.6179 | -0.0110 | Discard | Score decreased |
| 23 | 0.6564 | 0.7393 | +0.0829 | Discard | Perseveration increased by 0.2083 |
| 29 | 0.6332 | 0.7393 | +0.1061 | Keep | |
| 31 | 0.6606 | 0.7647 | +0.1041 | Discard | Perseveration increased by 0.0833 |
| 37 | 0.6332 | 0.7393 | +0.1061 | Keep | |
| 41 | 0.6332 | 0.7393 | +0.1061 | Keep | |
| 43 | 0.6332 | 0.7393 | +0.1061 | Keep | |
| 47 | 0.6342 | 0.6991 | +0.0649 | Keep | |
| 53 | 0.6332 | 0.7393 | +0.1061 | Keep | |

Summary across 11 seeds:

- Kept: 7 seeds (`7, 29, 37, 41, 43, 47, 53`)
- Discarded: 4 seeds (`11, 17, 23, 31`)
- Mean baseline score: `0.6405`
- Mean Arm 4 score: `0.7209`
- Mean score gain: `0.0805`
- Mean ignored-evidence rate: `0.2996 → 0.2703`
- Mean moved-but-uncertain rate: `0.0510 → 0.2157`

The project does not yet define one decision that combines several training seeds. Each seed currently receives its own keep/discard decision. Choosing and documenting the multi-seed promotion rule is the next important decision.

All saved policies currently have `ambiguous_abstention_accuracy: 0`: none returned `undetermined` on the no-access scenario.

## Working commands

The next registered candidates are Arm 7, which adds the existing commit guard to Arm 6, and Arm 8, which lowers Arm 4's belief prediction-error gain to 0.8 while keeping other settings at baseline. Run each as a paired 11-seed comparison:

```text
/compare episodes=50 seeds=7,11,17,23,29,31,37,41,43,47,53 candidate=arm7 output=logs
/compare episodes=50 seeds=7,11,17,23,29,31,37,41,43,47,53 candidate=arm8 output=logs
```

The exact settings are recorded in `program.md` and implemented by the corresponding variants in `train.py`.

Start the terminal interface:

```bash
cd "/Users/stephenbeale/Projects/Altered State Machine Learning"
.venv/bin/python rebus.py
```

Compare Arm 7 across the registered 11-seed cohort:

```text
/compare episodes=50 seeds=7,11,17,23,29,31,37,41,43,47,53 candidate=arm7 output=logs
```

Open a candidate trace:

```text
/trace output=logs/seed7-arm4/candidate
```

Show a saved score:

```text
/score metrics=logs/seed7-arm4/candidate/metrics.json
```

## Repository rules for continuing the work

- Read `program.md` before changing evaluation meaning.
- Record a score or scenario change in `program.md` before editing `eval.py` or the fixed scenarios.
- Compare baseline and candidate using the same seed, episode count, fixed scenarios, and evaluation version.
- Do not reconstruct keep/discard logic outside `eval.py`.
- Preserve old evaluation JSON before rewriting current results.
- Run logs, checkpoints, traces, selection reports, and decision notes are intentionally committed.

## Suggested next workflow

1. Run Arms 7 and 8 across the 11 registered seeds using the commands above.
2. Compare both with baseline and Arms 2–6 on score, perseveration, evidence uptake, belief flexibility, and unsafe commits.
3. Use per-seed and scenario-level outcomes to identify where the commit guard or lower prediction-error gain changes the Arm 6 trade-offs.
