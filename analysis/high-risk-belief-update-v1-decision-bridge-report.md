# High-Risk Belief Update v1: Decision Bridge Evaluation

Scenario suite: `high-risk-belief-update-scenarios-v2` (33 templates).
Checkpoints: existing matched baseline and Arm 4 v1 checkpoints across 11 training seeds. No retraining was performed.

This evaluation keeps each checkpoint's learned weights fixed and crosses them with the existing baseline and Arm 4 belief-filter settings. It does not vary a separate belief-update-rate parameter.

Across both checkpoint types, the Arm 4 filter lowered mean step Brier error by about 0.55, but local action flips were rare. With Arm 4 checkpoint weights, 7 of 1035 recorded decisions changed (0.7%), all on a single training seed. The paired full-trajectory reward difference was 0.004 [-0.005, 0.012] and its interval includes zero; terminal and safety rates did not change. The filter changes beliefs much more often than it changes choices in these scenarios.

## Closed-loop outcomes and local decision influence

The action-flip rate asks: at a recorded decision, does the same checkpoint choose a different action when only its current belief is replaced by the other filter's belief? For that local comparison, the observation and recurrent state are held fixed. The reward delta is the immediate simulator score for the reference action minus the actual action, including any current hard-gate penalty; it is not a full-episode counterfactual.

| Checkpoint weights | Belief filter | Mean reward | Completion | Appropriate terminal | Unsafe terminal | Gate-failing seeds | Step Brier vs truth | Changed decisions (seeds) | Local action flip | Helpful flips / flips | Harmful flips / flips | Reference gate-risk rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | baseline_filter | -15.067 | 0.300 | 0.171 | 0.066 | 1/11 | 1.111 | 1 (1) | 0.001 | 1.000 / 1 | 0.000 / 1 | 0.004 |
| baseline | arm4_filter | -15.128 | 0.295 | 0.165 | 0.066 | 1/11 | 0.565 | 0 (0) | 0.000 | 0.000 / 0 | 0.000 / 0 | 0.004 |
| arm4 | baseline_filter | -17.907 | 0.306 | 0.168 | 0.074 | 2/11 | 1.104 | 7 (1) | 0.007 | 1.000 / 7 | 0.000 / 7 | 0.005 |
| arm4 | arm4_filter | -17.903 | 0.306 | 0.168 | 0.074 | 2/11 | 0.559 | 7 (1) | 0.007 | 0.000 / 7 | 1.000 / 7 | 0.005 |

## Paired effect of the belief-filter setting

Arm 4 filter minus baseline filter, paired within training seed while holding checkpoint weights fixed. Intervals are Student-t 95% CIs over the 11 seed-level means.

| Fixed checkpoint weights | Metric | Mean difference | Paired 95% CI |
|---|---|---:|---:|
| baseline | reward | -0.062 | [-0.199, 0.076] |
| baseline | completed | -0.006 | [-0.018, 0.007] |
| baseline | appropriate_terminal | -0.006 | [-0.018, 0.007] |
| baseline | unsafe_terminal | 0.000 | [0.000, 0.000] |
| baseline | mean_step_brier_to_truth | -0.546 | [-0.551, -0.541] |
| arm4 | reward | 0.004 | [-0.005, 0.012] |
| arm4 | completed | 0.000 | [0.000, 0.000] |
| arm4 | appropriate_terminal | 0.000 | [0.000, 0.000] |
| arm4 | unsafe_terminal | 0.000 | [0.000, 0.000] |
| arm4 | mean_step_brier_to_truth | -0.545 | [-0.551, -0.538] |

## Where the filter changed an Arm 4 decision

On training seed(s) `17`, substituting the Arm 4 belief for the baseline belief changed 7 local decisions across these templates: `informative_omission`, `irreducible_sensor_noise`, `operator_belief_query`, `query_target_changes`, `reordered_evidence_stop_go`, `v2_handoff_omission_vs_sensor_loss`, `v2_robot_informative_then_masked_omission`. The local action direction was wait → yield. In the full paired episodes, using the Arm 4 filter changed mean reward by +0.200 per affected template; 7/7 ended appropriately, 0 were unsafe, and 0 breached a hard gate. These are useful case-specific examples, concentrated in 1 of 11 trained seeds, rather than a general performance effect.

## Interpretation limits

- The two filters change prior precision and cue gain together, so this comparison estimates the combined Arm 4 belief-filter effect, not separate causal effects of each parameter.
- The local action flip holds one checkpoint's current recurrent state fixed. It measures whether the current belief input can change the selected action at that moment; it does not estimate a complete alternative trajectory.
- The full closed-loop filter comparison branches from the start of each scenario, so its rewards and safety outcomes show the end-to-end effect of changing the filter for these frozen policy weights. Crossed filter settings were not used to train those checkpoints and are diagnostic counterfactuals.
- Scenario rewards and hard gates remain synthetic suite assumptions, not estimates of real-world harm or deployment safety.

## Artifacts

- `summary.json`: cell summaries, paired intervals, bridge rates, and provenance.
- `episode_outcomes.csv`: one row per seed, checkpoint, filter, and scenario.
- `decision_steps.csv`: actual and belief-substitution actions at each decision.
- `traces.json`: full closed-loop traces with actual and shadow beliefs.
