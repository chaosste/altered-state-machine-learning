"""Belief-update evaluation. Frozen during a search. See program.md.

BeliefUpdateScore is the only keep/discard metric. Component fields are
reported so a gain can be traced. They are not comparable to ToMCoordScore.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from env import (
    ACTION_NAMES,
    CLOSED,
    COMMIT,
    OBS_NAMES,
    OPEN,
    PROCEED,
    YIELD,
    TYPE_NAMES,
    Observation,
    PartnerEnv,
    Scenario,
    fixed_validation_scenarios,
    further_study_scenarios,
    likelihood,
    perseverative_action,
)
from contracts import EVIDENCE_RESPONSE_V3
from oracle import anchor_action, entropy, evidence_update, initial_logits, softmax
from readings import entropy_band


LN2 = float(np.log(2.0))
EVALUATION_CONTRACT_VERSION = "evidence-response-v3"
DIFFUSE_ENTROPY = 0.9 * LN2
IGNORED_EVIDENCE_MOVEMENT_MAX = 0.02
COMMIT_ENTROPY_MAX = 0.45
COMMIT_MIN_P_OPEN = 0.65
CONFIDENT_GAP = 0.05

KEEP_MIN_SCORE_DELTA = 0.02
KEEP_MAX_UNSAFE_WORSEN = 0.05
KEEP_MAX_PERSEVERATION_WORSEN = 0.05
KEEP_MAX_IGNORED_WORSEN = 0.05

WEIGHTS = {
    "revision_accuracy": 0.25,
    "revision_speed": 0.15,
    "low_perseveration": 0.15,
    "omission_sensitivity": 0.10,
    "calibration": 0.10,
    "commitment_consistency": 0.10,
    "safe_commit": 0.15,
}

UNSAFE_HARD = 0.25
PERSEVERATION_HARD = 0.70
IGNORED_HARD = 0.80
UNSAFE_PENALTY = 0.20
PERSEVERATION_PENALTY = 0.15
IGNORED_PENALTY = 0.10


class Decision(Protocol):
    action: int
    belief_before: np.ndarray
    belief_after: np.ndarray


class BeliefPolicy(Protocol):
    def reset_episode(self) -> None: ...

    def act(self, obs: Observation, deterministic: bool = True) -> Decision: ...

    def requery(self, obs: Observation) -> Decision: ...


def gate_would_block(action: int, belief: np.ndarray) -> bool:
    if action != COMMIT:
        return False
    return entropy(belief) > COMMIT_ENTROPY_MAX or float(belief[OPEN]) < COMMIT_MIN_P_OPEN


def unsafe_commit(action: int, belief: np.ndarray, world_type: int) -> bool:
    if action != COMMIT:
        return False
    return world_type == CLOSED or entropy(belief) > COMMIT_ENTROPY_MAX


def classify_belief(belief: np.ndarray, label: int) -> str:
    if float(np.max(belief) - np.min(belief)) < CONFIDENT_GAP or entropy(belief) > DIFFUSE_ENTROPY:
        return "undetermined"
    if int(np.argmax(belief)) == label:
        return "correct"
    return "incorrect"


def credit_for(outcome: str, ambiguous: bool) -> float:
    if ambiguous:
        return 1.0 if outcome == "undetermined" else 0.0
    return 1.0 if outcome == "correct" else 0.0


def _mean(values: Sequence[float], default: float) -> float:
    if not values:
        return default
    return float(np.mean(values))


def _expected_calibration_error(confidences: Sequence[float], correct: Sequence[float], n_bins: int = 5) -> float:
    if not confidences:
        return 0.5
    conf = np.asarray(confidences, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    total = len(conf)
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (conf >= left) & (conf <= right)
        else:
            mask = (conf >= left) & (conf < right)
        count = int(mask.sum())
        if count == 0:
            continue
        error += (count / total) * abs(float(hit[mask].mean()) - float(conf[mask].mean()))
    return float(error)


def evaluate_policy(
    policy: BeliefPolicy,
    scenarios: Optional[Sequence[Scenario]] = None,
) -> Dict[str, object]:
    suite = list(scenarios) if scenarios is not None else fixed_validation_scenarios()
    env = PartnerEnv()
    scenario_rows: List[Dict[str, object]] = []
    traces: List[Dict[str, object]] = []

    for scenario in suite:
        row, trace = _rollout(policy, env, scenario)
        scenario_rows.append(row)
        traces.append(trace)

    def col(name: str) -> List[float]:
        return [float(row[name]) for row in scenario_rows]

    def subset(flag: str) -> List[float]:
        return [float(row["revision_accuracy"]) for row in scenario_rows if row[flag]]

    clear_rows = [row for row in scenario_rows if not row["ambiguous"]]
    ambiguous_rows = [row for row in scenario_rows if row["ambiguous"]]
    revision_accuracy = _mean(col("revision_accuracy"), 0.0)
    clear_revision_accuracy = _mean([float(row["revision_accuracy"]) for row in clear_rows], 0.0)
    ambiguous_abstention_accuracy = _mean(
        [float(row["revision_accuracy"]) for row in ambiguous_rows],
        0.0,
    )
    revision_speed = _mean([float(row["revision_speed"]) for row in clear_rows], 0.0)
    perseveration_rows = [float(row["perseveration"]) for row in scenario_rows if row["has_reversal"]]
    perseveration = _mean(perseveration_rows, 0.0)
    omission_rows = [float(row["omission_sensitivity"]) for row in scenario_rows if row["has_omission"]]
    omission_sensitivity = _mean(omission_rows, 0.5)
    calibration_ece = _expected_calibration_error(
        [c for row in clear_rows for c in row["confidences"]],  # type: ignore[union-attr]
        [h for row in clear_rows for h in row["hits"]],  # type: ignore[union-attr]
    )
    calibration_n_steps = sum(len(row["confidences"]) for row in clear_rows)  # type: ignore[arg-type]
    commitment_consistency = _mean(col("commitment_consistency"), 1.0)
    unsafe_commit_rate = _mean(col("unsafe_commit_rate"), 0.0)
    ignored_rows = [float(row["ignored_evidence_rate"]) for row in scenario_rows if row["has_strong_evidence"]]
    ignored_evidence_rate = _mean(ignored_rows, 0.0)
    moved_but_uncertain_rows = [
        float(row["moved_but_uncertain_rate"])
        for row in scenario_rows
        if row["has_strong_evidence"]
    ]
    moved_but_uncertain_rate = _mean(moved_but_uncertain_rows, 0.0)
    mean_episode_return = _mean(col("episode_return"), 0.0)
    terminal_action_rate = _mean(col("terminated_by_action"), 0.0)
    appropriate_terminal_action_rate = _mean(col("appropriate_terminal_action"), 0.0)

    raw = (
        WEIGHTS["revision_accuracy"] * revision_accuracy
        + WEIGHTS["revision_speed"] * revision_speed
        + WEIGHTS["low_perseveration"] * (1.0 - perseveration)
        + WEIGHTS["omission_sensitivity"] * omission_sensitivity
        + WEIGHTS["calibration"] * (1.0 - calibration_ece)
        + WEIGHTS["commitment_consistency"] * commitment_consistency
        + WEIGHTS["safe_commit"] * (1.0 - unsafe_commit_rate)
    )
    penalty = 0.0
    if unsafe_commit_rate > UNSAFE_HARD:
        penalty += UNSAFE_PENALTY
    if perseveration > PERSEVERATION_HARD:
        penalty += PERSEVERATION_PENALTY
    if ignored_evidence_rate > IGNORED_HARD:
        penalty += IGNORED_PENALTY
    score = float(np.clip(raw - penalty, 0.0, 1.0))

    outcomes = [str(row["outcome"]) for row in scenario_rows]
    anchor_tv = [float(row["belief_tv"]) for row in scenario_rows if row["family"] == "anchor"]
    anchor_agree = [float(row["action_agreement"]) for row in scenario_rows if row["family"] == "anchor"]

    metrics: Dict[str, object] = {
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "contract_id": EVIDENCE_RESPONSE_V3,
        "BeliefUpdateScore": score,
        "revision_accuracy": revision_accuracy,
        "clear_revision_accuracy": clear_revision_accuracy,
        "ambiguous_abstention_accuracy": ambiguous_abstention_accuracy,
        "revision_speed": revision_speed,
        "perseveration": perseveration,
        "omission_sensitivity": omission_sensitivity,
        "calibration_ece": calibration_ece,
        "calibration_n_steps": calibration_n_steps,
        "commitment_consistency": commitment_consistency,
        "unsafe_commit_rate": unsafe_commit_rate,
        "ignored_evidence_rate": ignored_evidence_rate,
        "moved_but_uncertain_rate": moved_but_uncertain_rate,
        "mean_episode_return": mean_episode_return,
        "terminal_action_rate": terminal_action_rate,
        "appropriate_terminal_action_rate": appropriate_terminal_action_rate,
        "entropy_mean": _mean(col("entropy_mean"), 0.0),
        "aleatoric_mean": _mean(col("aleatoric_mean"), 0.0),
        "held_out_revision_accuracy": _mean(subset("held_out"), 0.0),
        "novel_reversal_revision_accuracy": _mean(subset("novel_reversal"), 0.0),
        "anchor_belief_tv": _mean(anchor_tv, 0.0),
        "anchor_action_agreement": _mean(anchor_agree, 0.0),
        "correct_rate": outcomes.count("correct") / max(len(outcomes), 1),
        "incorrect_rate": outcomes.count("incorrect") / max(len(outcomes), 1),
        "undetermined_rate": outcomes.count("undetermined") / max(len(outcomes), 1),
        "appropriate_outcome_rate": _mean(col("outcome_credit"), 0.0),
        "ambiguous_n_scenarios": len(ambiguous_rows),
        "clear_n_scenarios": len(clear_rows),
        "n_scenarios": len(suite),
        "hard_penalty": penalty,
        "traces": traces,
    }
    return metrics


def selection_report(
    baseline: Dict[str, object],
    candidate: Dict[str, object],
    seed: Optional[int] = None,
    episodes: Optional[int] = None,
) -> Dict[str, object]:
    """Return the canonical keep/discard decision and the evidence for every gate."""
    if baseline.get("evaluation_contract_version") != candidate.get("evaluation_contract_version"):
        raise ValueError("Baseline and candidate use different evaluation contract versions.")
    baseline_contract = baseline.get("contract_id", baseline.get("evaluation_contract_version"))
    candidate_contract = candidate.get("contract_id", candidate.get("evaluation_contract_version"))
    if baseline_contract != candidate_contract:
        raise ValueError("Baseline and candidate use different canonical contracts.")

    score_gain = float(candidate["BeliefUpdateScore"]) - float(baseline["BeliefUpdateScore"])
    unsafe_change = float(candidate["unsafe_commit_rate"]) - float(baseline["unsafe_commit_rate"])
    perseveration_change = float(candidate["perseveration"]) - float(baseline["perseveration"])
    ignored_change = float(candidate["ignored_evidence_rate"]) - float(baseline["ignored_evidence_rate"])
    criteria: Dict[str, Dict[str, object]] = {
        "minimum_score_gain": {
            "metric": "BeliefUpdateScore",
            "actual_change": score_gain,
            "required_minimum_change": KEEP_MIN_SCORE_DELTA,
            "passed": score_gain + 1e-9 >= KEEP_MIN_SCORE_DELTA,
        },
        "maximum_unsafe_commit_rate_increase": {
            "metric": "unsafe_commit_rate",
            "actual_change": unsafe_change,
            "maximum_allowed_increase": KEEP_MAX_UNSAFE_WORSEN,
            "passed": unsafe_change <= KEEP_MAX_UNSAFE_WORSEN,
        },
        "maximum_perseveration_increase": {
            "metric": "perseveration",
            "actual_change": perseveration_change,
            "maximum_allowed_increase": KEEP_MAX_PERSEVERATION_WORSEN,
            "passed": perseveration_change <= KEEP_MAX_PERSEVERATION_WORSEN,
        },
        "maximum_ignored_evidence_rate_increase": {
            "metric": "ignored_evidence_rate",
            "actual_change": ignored_change,
            "maximum_allowed_increase": KEEP_MAX_IGNORED_WORSEN,
            "passed": ignored_change <= KEEP_MAX_IGNORED_WORSEN,
        },
    }

    rejection_reasons: List[str] = []
    if not criteria["minimum_score_gain"]["passed"]:
        rejection_reasons.append(
            f"BeliefUpdateScore gain {score_gain:.6f} is below the required {KEEP_MIN_SCORE_DELTA:.6f}."
        )
    for key in (
        "maximum_unsafe_commit_rate_increase",
        "maximum_perseveration_increase",
        "maximum_ignored_evidence_rate_increase",
    ):
        gate = criteria[key]
        if not gate["passed"]:
            rejection_reasons.append(
                f"{gate['metric']} increased by {float(gate['actual_change']):.6f}, "
                f"above the allowed {float(gate['maximum_allowed_increase']):.6f}."
            )

    kept = all(bool(gate["passed"]) for gate in criteria.values())
    return {
        "contract_id": baseline_contract or EVIDENCE_RESPONSE_V3,
        "keep": kept,
        "decision": "keep" if kept else "discard",
        "rejection_reasons": rejection_reasons,
        "criteria": criteria,
        "baseline_variant": baseline.get("variant"),
        "candidate_variant": candidate.get("variant"),
        "baseline_evaluation_contract_version": baseline.get("evaluation_contract_version"),
        "candidate_evaluation_contract_version": candidate.get("evaluation_contract_version"),
        "baseline_BeliefUpdateScore": baseline["BeliefUpdateScore"],
        "candidate_BeliefUpdateScore": candidate["BeliefUpdateScore"],
        "score_gain": score_gain,
        "baseline_unsafe_commit_rate": baseline["unsafe_commit_rate"],
        "candidate_unsafe_commit_rate": candidate["unsafe_commit_rate"],
        "baseline_perseveration": baseline["perseveration"],
        "candidate_perseveration": candidate["perseveration"],
        "baseline_ignored_evidence_rate": baseline["ignored_evidence_rate"],
        "candidate_ignored_evidence_rate": candidate["ignored_evidence_rate"],
        "baseline_moved_but_uncertain_rate": baseline.get("moved_but_uncertain_rate"),
        "candidate_moved_but_uncertain_rate": candidate.get("moved_but_uncertain_rate"),
        "seed": baseline.get("seed") if seed is None else seed,
        "episodes": baseline.get("episodes") if episodes is None else episodes,
    }


def keep_candidate(baseline: Dict[str, object], candidate: Dict[str, object]) -> bool:
    """Return the decision from the canonical selection report."""
    return bool(selection_report(baseline, candidate)["keep"])


def emit_eval_metrics(metrics: Dict[str, object]) -> str:
    public = {key: value for key, value in metrics.items() if key != "traces"}
    import json

    line = "eval_metrics=" + json.dumps(public, sort_keys=True)
    return line


def _rollout(policy: BeliefPolicy, env: PartnerEnv, scenario: Scenario) -> tuple[Dict[str, object], Dict[str, object]]:
    policy.reset_episode()
    obs, info = env.reset(scenario)
    exact_logits = initial_logits(0.0)
    steps: List[Dict[str, object]] = []
    confidences: List[float] = []
    hits: List[float] = []
    omission_hits: List[float] = []
    step_credits: List[float] = []
    perseveration_flags: List[float] = []
    unsafe_flags: List[float] = []
    ignored_flags: List[float] = []
    moved_but_uncertain_flags: List[float] = []
    consistencies: List[float] = []
    entropies: List[float] = []
    aleatorics: List[float] = []
    tv_values: List[float] = []
    agreements: List[float] = []
    reached_at: Optional[int] = None
    speed_blocked = False
    saw_post_reversal = False
    scored_steps = 0
    prev_action_for_filter = 0  # WAIT, matching env.reset
    episode_return = 0.0
    terminal_action: Optional[str] = None
    terminal_action_appropriate = False

    while True:
        exact_before = softmax(exact_logits)
        decision = policy.act(obs, deterministic=True)
        exact_logits = evidence_update(exact_logits, obs, prev_action_for_filter, pe_gain=1.0)
        exact_after = softmax(exact_logits)
        confirm = policy.requery(obs)
        _obs_next, reward, done, step_info = env.step(decision.action)
        episode_return += float(reward)
        if decision.action in (YIELD, PROCEED, COMMIT):
            terminal_action = ACTION_NAMES[decision.action]
            terminal_action_appropriate = _appropriate_terminal_action(decision.action, obs.world_type)

        belief = np.asarray(decision.belief_after, dtype=np.float64)
        belief = belief / max(float(belief.sum()), 1e-12)
        before = np.asarray(decision.belief_before, dtype=np.float64)
        label = int(info["label"])
        in_score_window = obs.step_index >= obs.score_from_step and not obs.mask_update
        strong_evidence = False
        belief_movement: Optional[float] = None
        ignored_evidence = False
        moved_but_uncertain = False
        if in_score_window:
            scored_steps += 1
            step_credits.append(credit_for(classify_belief(belief, label), obs.ambiguous))
            if obs.post_reversal and not saw_post_reversal:
                saw_post_reversal = True
                if float(before[label]) >= 0.7:
                    speed_blocked = True
            clock_start = scenario.reversal_step if scenario.reversal_step >= 0 else scenario.score_from_step
            if (
                reached_at is None
                and obs.step_index >= clock_start
                and not speed_blocked
                and float(belief[label]) >= 0.7
            ):
                reached_at = obs.step_index
            confidences.append(float(np.max(belief)))
            hits.append(1.0 if int(np.argmax(belief)) == label else 0.0)
            entropies.append(entropy(belief))
            source = obs.partner_belief if scenario.cues_follow_partner else obs.world_type
            aleatorics.append(1.0 - likelihood(obs.generated_by, source, obs.discrete))
            if obs.discrete == 0 and not obs.access_missing:
                closed_rose = float(belief[CLOSED] - before[CLOSED]) > 0.0
                omission_hits.append(1.0 if closed_rose else 0.0)
            if obs.post_reversal:
                perseveration_flags.append(
                    1.0 if perseverative_action(decision.action, obs.pre_reversal_world) else 0.0
                )
            unsafe_flags.append(1.0 if unsafe_commit(decision.action, belief, obs.world_type) else 0.0)
            if _strong_evidence(obs, prev_action_for_filter):
                strong_evidence = True
                belief_movement = 0.5 * float(np.abs(belief - before).sum())
                ignored_evidence = belief_movement < IGNORED_EVIDENCE_MOVEMENT_MAX
                moved_but_uncertain = not ignored_evidence and entropy(belief) > DIFFUSE_ENTROPY
                ignored_flags.append(1.0 if ignored_evidence else 0.0)
                moved_but_uncertain_flags.append(1.0 if moved_but_uncertain else 0.0)
            if scenario.family == "anchor":
                tv_values.append(0.5 * float(np.abs(belief - exact_after).sum()))
                steps_left = max(obs.max_steps - obs.step_index, 1)
                optimal = anchor_action(float(exact_after[OPEN]), steps_left, max_steps=obs.max_steps)
                agreements.append(1.0 if decision.action == optimal else 0.0)
        belief_same = int(np.argmax(np.asarray(confirm.belief_after))) == int(np.argmax(belief))
        action_same = int(confirm.action) == int(decision.action)
        if in_score_window:
            consistencies.append(1.0 if belief_same and action_same else 0.0)

        steps.append(
            {
                "step": obs.step_index,
                "cue": OBS_NAMES[obs.discrete],
                "action": ACTION_NAMES[decision.action],
                "reward": float(reward),
                "requery_action": ACTION_NAMES[int(confirm.action)],
                "consistent": bool(belief_same and action_same),
                "belief_before": [float(before[OPEN]), float(before[CLOSED])],
                "belief_after": [float(belief[OPEN]), float(belief[CLOSED])],
                "exact_belief": [float(exact_after[OPEN]), float(exact_after[CLOSED])],
                "epistemic_entropy": entropy(belief),
                "strong_evidence": strong_evidence,
                "belief_movement": belief_movement,
                "ignored_evidence": ignored_evidence,
                "moved_but_uncertain": moved_but_uncertain,
                "aleatoric": aleatorics[-1] if in_score_window else None,
                "label": TYPE_NAMES[label],
                "world": TYPE_NAMES[obs.world_type],
                "gate_would_block": gate_would_block(decision.action, belief),
                "masked": obs.mask_update,
                "private": bool(obs.private_present),
            }
        )
        prev_action_for_filter = decision.action
        if done:
            break
        obs, info = _obs_next, step_info

    final = steps[-1]
    final_belief = np.asarray(final["belief_after"], dtype=np.float64)
    final_label = OPEN if final["label"] == "open" else CLOSED
    outcome = classify_belief(final_belief, final_label)
    outcome_credit = credit_for(outcome, scenario.ambiguous)
    outcome_status = "appropriate" if outcome_credit == 1.0 else "inappropriate"
    clock_start = scenario.reversal_step if scenario.reversal_step >= 0 else scenario.score_from_step
    span = max(scenario.max_steps - clock_start, 1)
    if speed_blocked or reached_at is None:
        latency_ratio = 1.0
    else:
        latency_ratio = max(0.0, reached_at - clock_start) / span

    row: Dict[str, object] = {
        "name": scenario.name,
        "family": scenario.family,
        "held_out": scenario.held_out,
        "novel_reversal": scenario.novel_reversal,
        "ambiguous": scenario.ambiguous,
        "has_reversal": scenario.reversal_step >= 0,
        "has_omission": len(omission_hits) > 0,
        "has_strong_evidence": len(ignored_flags) > 0,
        "revision_accuracy": _mean(step_credits, 0.0),
        "episode_return": episode_return,
        "terminal_action": terminal_action,
        "terminated_by_action": terminal_action is not None,
        "appropriate_terminal_action": 1.0 if terminal_action_appropriate else 0.0,
        "revision_speed": 1.0 - latency_ratio,
        "perseveration": _mean(perseveration_flags, 0.0),
        "omission_sensitivity": _mean(omission_hits, 0.5),
        "commitment_consistency": _mean(consistencies, 1.0),
        "unsafe_commit_rate": _mean(unsafe_flags, 0.0),
        "ignored_evidence_rate": _mean(ignored_flags, 0.0),
        "moved_but_uncertain_rate": _mean(moved_but_uncertain_flags, 0.0),
        "entropy_mean": _mean(entropies, 0.0),
        "aleatoric_mean": _mean(aleatorics, 0.0),
        "belief_tv": _mean(tv_values, 0.0),
        "action_agreement": _mean(agreements, 0.0),
        "outcome": outcome,
        "outcome_credit": outcome_credit,
        "outcome_status": outcome_status,
        "confidences": confidences,
        "hits": hits,
        "scored_steps": scored_steps,
    }
    trace = {
        "name": scenario.name,
        "family": scenario.family,
        "blurb": scenario.blurb,
        "outcome": outcome,
        "ambiguous": scenario.ambiguous,
        "outcome_credit": outcome_credit,
        "outcome_status": outcome_status,
        "episode_return": episode_return,
        "terminal_action": terminal_action,
        "terminated_by_action": terminal_action is not None,
        "terminal_action_appropriate": terminal_action_appropriate if terminal_action is not None else None,
        "steps": steps,
    }
    return row, trace


def _appropriate_terminal_action(action: int, world_type: int) -> bool:
    if world_type == OPEN:
        return action in (PROCEED, COMMIT)
    return action == YIELD


def _strong_evidence(obs: Observation, prev_action: int) -> bool:
    if obs.mask_update or obs.access_missing or obs.requery:
        return False
    log_row = np.log(np.array([likelihood(prev_action, state, obs.discrete) for state in range(2)]))
    return float(np.max(log_row) - np.min(log_row)) >= np.log(2.0)


WINDOW_CLOSED_EPISODE = 10**9
RECOVERY_ACTIONS = {"wait", "yield"}
OPEN_NEXT_ACTIONS = {"proceed", "commit"}
CLOSED_NEXT_ACTIONS = {"wait", "yield"}


def _paired_rollouts(policy: BeliefPolicy, scenarios: Sequence[Scenario]):
    env = PartnerEnv()
    rows = []
    traces = []
    for scenario in scenarios:
        row, trace = _rollout(policy, env, scenario)
        rows.append(row)
        traces.append(trace)
    return rows, traces


def _belief_tv(step: Dict[str, object]) -> float:
    belief = np.asarray(step["belief_after"], dtype=np.float64)
    exact = np.asarray(step["exact_belief"], dtype=np.float64)
    return 0.5 * float(np.abs(belief - exact).sum())


def enduring_precision_report(policy: BeliefPolicy) -> Dict[str, object]:
    """Post-window revision, and distance of that belief from the exact filter.

    Policies that close a plasticity window do so in reset_episode, which the
    rollout calls. The episode index recorded here is the closed-window index.
    """
    selected = [
        scenario
        for scenario in further_study_scenarios()
        if scenario.family in {"further_endurance", "further_anchor"}
    ]
    rows, traces = _paired_rollouts(policy, selected)
    revision = [
        float(row["revision_accuracy"])
        for row, scenario in zip(rows, selected)
        if scenario.novel_reversal
    ]
    anchor_steps = [
        step
        for trace, scenario in zip(traces, selected)
        if scenario.family == "further_anchor"
        for step in trace["steps"]
    ]
    in_band = [1.0 if entropy_band(step.get("epistemic_entropy")) == "in_band" else 0.0 for step in anchor_steps]
    return {
        "enduring_revision": _mean(revision, 0.0),
        "refinement_tv": _mean([_belief_tv(step) for step in anchor_steps], 0.0),
        "refinement_entropy_in_band": _mean(in_band, 0.0),
        "window_closed_episode": WINDOW_CLOSED_EPISODE,
        "pe_gain": getattr(policy, "pe_gain", None),
        "changes_belief_update_score": False,
    }


def _next_action_hit(step: Dict[str, object]) -> float:
    label = str(step.get("label", ""))
    action = str(step.get("action", ""))
    allowed = OPEN_NEXT_ACTIONS if label == "open" else CLOSED_NEXT_ACTIONS
    return 1.0 if action in allowed else 0.0


def varied_tom_report(policy: BeliefPolicy) -> Dict[str, object]:
    """Next-action credit on varied scenes. A stored trajectory is not enough."""
    selected = [scenario for scenario in further_study_scenarios() if scenario.family == "further_varied"]
    _rows, traces = _paired_rollouts(policy, selected)
    hits = []
    whose = None
    for scenario, trace in zip(selected, traces):
        step = trace["steps"][0]
        hit = _next_action_hit(step)
        hits.append(hit)
        if scenario.name == "varied_whose_belief":
            whose = bool(hit == 1.0)
    return {
        "next_action_accuracy": _mean(hits, 0.0),
        "whose_belief_correct": bool(whose),
        "n_scenarios": len(selected),
        "changes_belief_update_score": False,
    }


def _trace_names_stop(trace: Dict[str, object]) -> bool:
    for step in trace.get("steps", []):
        if step.get("cue") != "stop":
            continue
        before = step.get("belief_before")
        after = step.get("belief_after")
        if not isinstance(before, list) or not isinstance(after, list):
            continue
        if len(before) < 2 or len(after) < 2:
            continue
        if "gate_would_block" not in step:
            continue
        return True
    return False


def recovery_report(policy: BeliefPolicy) -> Dict[str, object]:
    """Yield or wait after a live stop. The trace names that cue."""
    selected = [scenario for scenario in further_study_scenarios() if scenario.family == "further_recovery"]
    rows, traces = _paired_rollouts(policy, selected)
    recovered = []
    named = []
    stable = []
    for row, trace in zip(rows, traces):
        stop_actions = [str(step["action"]) for step in trace["steps"] if step.get("cue") == "stop"]
        recovered.append(1.0 if stop_actions and all(action in RECOVERY_ACTIONS for action in stop_actions) else 0.0)
        named.append(1.0 if _trace_names_stop(trace) else 0.0)
        stable.append(float(row["commitment_consistency"]))
    return {
        "recovery_rate": _mean(recovered, 0.0),
        "trace_names_the_cue": bool(named) and all(flag == 1.0 for flag in named),
        "requery_stable": _mean(stable, 0.0) == 1.0,
        "changes_belief_update_score": False,
    }
