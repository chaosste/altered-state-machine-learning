# High-Risk Belief Update v1: Cost Sensitivity

This is a post-hoc, exploratory re-score of saved v1 closed-loop action traces. It does not retrain or rerun policies.
Alternative cost values are provisional simulation assumptions, not real-world safety or economic estimates.

## What cost sensitivity means

A cost sensitivity analysis asks whether the comparison changes when the assumed value of delay, terminal mistakes, or catastrophic gate breaches changes. Here, the recorded actions and states are fixed; only their reward weights change. This isolates how much the reported reward ranking depends on the chosen matrix. It does not predict how a policy would behave if it were trained with a different matrix.

## Scenarios and mean reward by arm

Higher reward is better within this synthetic score. Means below are first averaged within each seed over its saved cases, then averaged across the 11 matched seeds.

| Cost scenario | Baseline | Arm 3 | Arm 4 | Ranking (best to worst) | Changed from v1? |
|---|---:|---:|---:|---|---|
| reference_v1 | -12.701 | -48.076 | -12.842 | baseline > arm4 > arm3 | No |
| lighter_delay | -12.463 | -47.709 | -12.603 | baseline > arm4 > arm3 | No |
| heavier_delay | -13.177 | -48.810 | -13.319 | baseline > arm4 > arm3 | No |
| lighter_terminal_error | -12.316 | -46.797 | -12.413 | baseline > arm4 > arm3 | No |
| heavier_terminal_error | -13.472 | -50.635 | -13.699 | baseline > arm4 > arm3 | No |
| lower_catastrophe_penalty | -8.372 | -26.431 | -8.513 | baseline > arm4 > arm3 | No |
| higher_catastrophe_penalty | -21.359 | -91.366 | -21.500 | baseline > arm4 > arm3 | No |

## Paired seed deltas

A positive delta favors the treatment arm. Intervals are paired Student-t 95% confidence intervals over seed-level means.

| Scenario | Contrast | Mean delta | 95% CI |
|---|---|---:|---:|
| reference_v1 | arm3 − baseline | -35.375 | [-71.319, 0.569] |
| reference_v1 | arm4 − baseline | -0.140 | [-0.363, 0.083] |
| reference_v1 | arm4 − arm3 | 35.235 | [-0.598, 71.068] |
| lighter_delay | arm3 − baseline | -35.246 | [-71.284, 0.792] |
| lighter_delay | arm4 − baseline | -0.139 | [-0.363, 0.084] |
| lighter_delay | arm4 − arm3 | 35.106 | [-0.818, 71.031] |
| heavier_delay | arm3 − baseline | -35.633 | [-71.397, 0.132] |
| heavier_delay | arm4 − baseline | -0.142 | [-0.365, 0.081] |
| heavier_delay | arm4 − arm3 | 35.491 | [-0.168, 71.150] |
| lighter_terminal_error | arm3 − baseline | -34.481 | [-69.680, 0.718] |
| lighter_terminal_error | arm4 − baseline | -0.097 | [-0.314, 0.120] |
| lighter_terminal_error | arm4 − arm3 | 34.384 | [-0.741, 69.509] |
| heavier_terminal_error | arm3 − baseline | -37.163 | [-74.606, 0.281] |
| heavier_terminal_error | arm4 − baseline | -0.227 | [-0.560, 0.106] |
| heavier_terminal_error | arm4 − arm3 | 36.936 | [-0.322, 74.194] |
| lower_catastrophe_penalty | arm3 − baseline | -18.059 | [-36.446, 0.328] |
| lower_catastrophe_penalty | arm4 − baseline | -0.140 | [-0.363, 0.083] |
| lower_catastrophe_penalty | arm4 − arm3 | 17.919 | [-0.360, 36.197] |
| higher_catastrophe_penalty | arm3 − baseline | -70.007 | [-141.209, 1.196] |
| higher_catastrophe_penalty | arm4 − baseline | -0.140 | [-0.363, 0.083] |
| higher_catastrophe_penalty | arm4 − arm3 | 69.867 | [-1.224, 140.957] |

## Hard safety gates

Hard gates are unchanged by reward re-scoring and remain a separate pass/fail result:

| Arm | Failed seeds | Total seeds |
|---|---:|---:|
| baseline | 1 | 11 |
| arm3 | 4 | 11 |
| arm4 | 1 | 11 |

A better mean reward cannot compensate for a hard-gate failure. Interpret alternative rankings as sensitivity of this fixed trace sample to the assumed weights, not as a recommendation to deploy or as causal policy adaptation.

## Files

- `logs/high-risk-belief-update-v1-cost-sensitivity/summary.json`: complete matrices, rankings, paired deltas, gate outcomes, and episode scores.
- `logs/high-risk-belief-update-v1-cost-sensitivity/episode_scores.csv`: episode-level re-scoring.
- `logs/high-risk-belief-update-v1-cost-sensitivity/seed_rewards.csv`: per-seed/per-arm means used for paired analysis.
- `logs/high-risk-belief-update-v1-cost-sensitivity/paired_seed_deltas.csv`: paired contrasts and confidence intervals.
- `logs/high-risk-belief-update-v1-cost-sensitivity/scenario_arm_summary.csv`: arm means and ranking per cost scenario.
- `logs/high-risk-belief-update-v1-cost-sensitivity/hard_gate_status.csv`: unchanged hard-gate status by seed and arm.
