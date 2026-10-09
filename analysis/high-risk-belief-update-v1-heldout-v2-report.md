# Held-out scenario expansion for High-Risk Belief Update v1

Contract: `high-risk-belief-update-v1`  
Expanded scenario suite: `high-risk-belief-update-scenarios-v2`  
Original suite: `high-risk-belief-update-scenarios-v1` with 21 cases; added 12 new templates; combined suite has 33 cases.
Training: 50 episodes per arm on the original v1 schedule; device cpu. Existing saved baseline, Arm 3, and Arm 4 checkpoints were reused; no retraining was performed.

This is a post-training held-out-template extension. The new cases alter evidence order, access recovery, operator/world attribution, stale reports, label mappings, and reversal patterns. The domain consequences and cost matrix remain v1 simulation assumptions.

## Combined original plus added templates

| Arm | Common replay Brier vs exact | Common replay Brier vs truth | Mean task reward | Completion | Appropriate terminal | Inappropriate terminal | Unsafe terminal | Gate-failing seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.249 | 1.111 | -15.067 | 0.300 | 0.171 | 0.129 | 0.066 | 1/11 |
| arm3 | 0.249 | 1.111 | -54.310 | 0.468 | 0.262 | 0.207 | 0.168 | 4/11 |
| arm4 | 0.157 | 0.928 | -17.903 | 0.306 | 0.168 | 0.138 | 0.074 | 2/11 |

## Added templates only

| Arm | Common replay Brier vs exact | Common replay Brier vs truth | Mean task reward | Completion | Appropriate terminal | Inappropriate terminal | Unsafe terminal | Gate-failing seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.208 | 1.180 | -19.206 | 0.311 | 0.174 | 0.136 | 0.068 | 1/11 |
| arm3 | 0.208 | 1.180 | -65.220 | 0.477 | 0.280 | 0.197 | 0.152 | 4/11 |
| arm4 | 0.162 | 1.051 | -26.761 | 0.326 | 0.182 | 0.144 | 0.076 | 2/11 |

## Added-template family outcomes

| Arm | Family | Mean reward | Completion | Appropriate terminal | Inappropriate terminal | Unsafe terminal | Critical misses | Safe handoff | No terminal action |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | critical_reversal | -1.800 | 0.455 | 0.364 | 0.091 | 0.000 | 0.636 | 0.091 | 0.545 |
| baseline | false_belief_attribution | -4.000 | 0.273 | 0.091 | 0.182 | 0.000 | 0.000 | 0.000 | 0.727 |
| baseline | heldout_cue_mapping | -4.000 | 0.273 | 0.091 | 0.182 | 0.000 | 0.000 | 0.000 | 0.727 |
| baseline | omission | -4.273 | 0.273 | 0.182 | 0.091 | 0.091 | 0.818 | 0.182 | 0.727 |
| baseline | query_target_change | -4.418 | 0.273 | 0.182 | 0.091 | 0.091 | 0.818 | 0.182 | 0.727 |
| baseline | reordered_evidence | -4.418 | 0.273 | 0.182 | 0.091 | 0.091 | 0.818 | 0.182 | 0.727 |
| baseline | stale_handoff | -4.764 | 0.364 | 0.091 | 0.273 | 0.091 | 0.909 | 0.000 | 0.636 |
| baseline | unavailable_access | -94.745 | 0.318 | 0.227 | 0.091 | 0.091 | 0.773 | 0.182 | 0.682 |
| baseline | unfamiliar_source | -4.273 | 0.273 | 0.182 | 0.091 | 0.091 | 0.818 | 0.182 | 0.727 |
| arm3 | critical_reversal | -3.400 | 0.455 | 0.364 | 0.091 | 0.091 | 0.636 | 0.091 | 0.545 |
| arm3 | false_belief_attribution | -1.491 | 0.545 | 0.364 | 0.182 | 0.000 | 0.000 | 0.000 | 0.455 |
| arm3 | heldout_cue_mapping | -0.291 | 0.636 | 0.455 | 0.182 | 0.000 | 0.000 | 0.000 | 0.364 |
| arm3 | omission | -51.873 | 0.364 | 0.182 | 0.182 | 0.182 | 0.818 | 0.182 | 0.636 |
| arm3 | query_target_change | -98.636 | 0.636 | 0.182 | 0.455 | 0.455 | 0.818 | 0.182 | 0.364 |
| arm3 | reordered_evidence | -190.473 | 0.455 | 0.091 | 0.364 | 0.364 | 0.909 | 0.091 | 0.545 |
| arm3 | stale_handoff | -1.282 | 0.545 | 0.455 | 0.091 | 0.000 | 0.545 | 0.000 | 0.455 |
| arm3 | unavailable_access | -187.382 | 0.455 | 0.273 | 0.182 | 0.182 | 0.727 | 0.182 | 0.545 |
| arm3 | unfamiliar_source | -7.273 | 0.273 | 0.091 | 0.182 | 0.182 | 0.909 | 0.091 | 0.727 |
| arm4 | critical_reversal | -2.800 | 0.364 | 0.273 | 0.091 | 0.000 | 0.727 | 0.091 | 0.636 |
| arm4 | false_belief_attribution | -3.091 | 0.364 | 0.182 | 0.182 | 0.000 | 0.000 | 0.000 | 0.636 |
| arm4 | heldout_cue_mapping | -3.091 | 0.364 | 0.182 | 0.182 | 0.000 | 0.000 | 0.000 | 0.636 |
| arm4 | omission | -4.273 | 0.273 | 0.182 | 0.091 | 0.091 | 0.818 | 0.182 | 0.727 |
| arm4 | query_target_change | -4.418 | 0.273 | 0.182 | 0.091 | 0.091 | 0.818 | 0.182 | 0.727 |
| arm4 | reordered_evidence | -95.764 | 0.364 | 0.182 | 0.182 | 0.182 | 0.818 | 0.182 | 0.636 |
| arm4 | stale_handoff | -4.755 | 0.364 | 0.091 | 0.273 | 0.091 | 0.909 | 0.000 | 0.636 |
| arm4 | unavailable_access | -94.809 | 0.318 | 0.227 | 0.091 | 0.091 | 0.773 | 0.182 | 0.682 |
| arm4 | unfamiliar_source | -4.291 | 0.273 | 0.182 | 0.091 | 0.091 | 0.818 | 0.182 | 0.727 |

## Paired differences on added templates

Intervals use the 11 paired saved training seeds. Replay contrasts are deterministic on the fixed scripted histories and are reported as exact differences without a seed confidence interval.

| Contrast | Outcome | Mean difference | Median difference | Paired 95% CI |
|---|---|---:|---:|---:|
| arm3 − baseline | common_prior_stepwise_brier_to_exact | 0.000 | 0.000 | fixed replay |
| arm3 − baseline | closed_loop_mean_reward | -46.014 | -0.833 | [-91.580, -0.447] |
| arm3 − baseline | closed_loop_completion_rate | 0.167 | 0.000 | [-0.170, 0.503] |
| arm3 − baseline | closed_loop_appropriate_terminal_rate | 0.106 | 0.000 | [-0.099, 0.311] |
| arm3 − baseline | closed_loop_inappropriate_terminal_rate | 0.061 | 0.000 | [-0.079, 0.200] |
| arm3 − baseline | closed_loop_unsafe_terminal_rate | 0.083 | 0.000 | [-0.010, 0.177] |
| arm3 − baseline | closed_loop_critical_event_miss_rate | -0.053 | 0.000 | [-0.232, 0.126] |
| arm4 − baseline | common_prior_stepwise_brier_to_exact | -0.046 | -0.046 | fixed replay |
| arm4 − baseline | closed_loop_mean_reward | -7.555 | 0.000 | [-23.956, 8.847] |
| arm4 − baseline | closed_loop_completion_rate | 0.015 | 0.000 | [-0.063, 0.094] |
| arm4 − baseline | closed_loop_appropriate_terminal_rate | 0.008 | 0.000 | [-0.056, 0.071] |
| arm4 − baseline | closed_loop_inappropriate_terminal_rate | 0.008 | 0.000 | [-0.009, 0.024] |
| arm4 − baseline | closed_loop_unsafe_terminal_rate | 0.008 | 0.000 | [-0.009, 0.024] |
| arm4 − baseline | closed_loop_critical_event_miss_rate | 0.008 | 0.000 | [-0.032, 0.047] |
| arm4 − arm3 | common_prior_stepwise_brier_to_exact | -0.046 | -0.046 | fixed replay |
| arm4 − arm3 | closed_loop_mean_reward | 38.459 | 0.833 | [0.309, 76.609] |
| arm4 − arm3 | closed_loop_completion_rate | -0.152 | 0.000 | [-0.479, 0.176] |
| arm4 − arm3 | closed_loop_appropriate_terminal_rate | -0.098 | 0.000 | [-0.309, 0.112] |
| arm4 − arm3 | closed_loop_inappropriate_terminal_rate | -0.053 | 0.000 | [-0.184, 0.078] |
| arm4 − arm3 | closed_loop_unsafe_terminal_rate | -0.076 | 0.000 | [-0.157, 0.005] |
| arm4 − arm3 | closed_loop_critical_event_miss_rate | 0.061 | 0.000 | [-0.125, 0.246] |

Paired task-outcome intervals use the 11 existing training seeds and describe model-to-model variability over this fixed added-template set. Controlled replay is deterministic and has no seed-sampling confidence interval. Any hard-gate breach remains a failure independent of average reward.

## Worst added-template family

| Arm | Lowest-reward family | Mean reward | Unsafe terminal rate | Critical miss rate |
|---|---|---:|---:|---:|
| baseline | unavailable_access | -94.745 | 0.091 | 0.773 |
| arm3 | reordered_evidence | -190.473 | 0.364 | 0.909 |
| arm4 | reordered_evidence | -95.764 | 0.182 | 0.818 |

## Provenance and limits

- Original v1 comparison root: `logs/high-risk-belief-update-v1/`; schedule hashes and training metrics remain unchanged.
- New outputs: `logs/high-risk-belief-update-v1-heldout-v2/`; canonical contract ID `high-risk-belief-update-v1` and scenario suite ID `high-risk-belief-update-scenarios-v2`.
- Source/package provenance is recorded in the new comparison summary. The v1 trained models were evaluated as saved.
- Robot, route, communication, source-reliability, safety, and reward mappings are synthetic assumptions. Results do not validate real systems.
- This expands scenario-template coverage but retains the original 11 trained seed models; it is not an independent retraining cohort.
