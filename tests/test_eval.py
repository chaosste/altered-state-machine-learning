"""Eval integrity: shortcuts fail, confirmation is scored, keep/discard is fixed."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from env import CLOSED, COMMIT, GO, OPEN, STOP, WAIT, YIELD, Observation, fixed_validation_scenarios
from eval import evaluate_policy, keep_candidate
from oracle import anchor_action, evidence_update, initial_logits, softmax


@dataclass
class Decision:
    action: int
    belief_before: np.ndarray
    belief_after: np.ndarray


class CueMemorizer:
    """Trusts the latest public cue. The held-out partner says go and is closed."""

    def reset_episode(self) -> None:
        self.belief = np.array([0.5, 0.5])
        self.action = WAIT

    def act(self, obs: Observation, deterministic: bool = True) -> Decision:
        before = self.belief.copy()
        if obs.mask_update or obs.discrete == 0 and not obs.private_present:
            action = self.action
        elif obs.discrete == GO or (obs.private_present and obs.private_says_open):
            self.belief = np.array([0.99, 0.01])
            action = COMMIT
        elif obs.discrete == STOP or (obs.private_present and not obs.private_says_open):
            self.belief = np.array([0.01, 0.99])
            action = YIELD
        else:
            action = WAIT
        self.action = action
        return Decision(action, before, self.belief.copy())

    def requery(self, obs: Observation) -> Decision:
        return Decision(self.action, self.belief.copy(), self.belief.copy())


class FlippingPolicy:
    def reset_episode(self) -> None:
        self.belief = np.array([0.8, 0.2])

    def act(self, obs: Observation, deterministic: bool = True) -> Decision:
        return Decision(YIELD, self.belief.copy(), self.belief.copy())

    def requery(self, obs: Observation) -> Decision:
        return Decision(COMMIT, self.belief.copy(), self.belief.copy())


class ExactFilterPolicy:
    """Gain-1 filter from a uniform prior. Confirmation replays the same action."""

    def reset_episode(self) -> None:
        self.logits = initial_logits(0.0)
        self.prev = WAIT
        self.belief = softmax(self.logits)
        self.action = WAIT

    def act(self, obs: Observation, deterministic: bool = True) -> Decision:
        before = softmax(self.logits)
        self.logits = evidence_update(self.logits, obs, self.prev, pe_gain=1.0)
        after = softmax(self.logits)
        self.belief = after
        if obs.family == "anchor":
            steps_left = max(obs.max_steps - obs.step_index, 1)
            action = anchor_action(float(after[OPEN]), steps_left, max_steps=obs.max_steps)
        elif float(after[OPEN]) >= 0.75:
            action = COMMIT
        elif float(after[CLOSED]) >= 0.75:
            action = YIELD
        else:
            action = WAIT
        self.prev = action
        self.action = action
        return Decision(action, before, after.copy())

    def requery(self, obs: Observation) -> Decision:
        return Decision(self.action, self.belief.copy(), self.belief.copy())


def _blank_metrics(**overrides: float) -> dict:
    base = {
        "BeliefUpdateScore": 0.40,
        "unsafe_commit_rate": 0.05,
        "perseveration": 0.10,
        "ignored_evidence_rate": 0.10,
    }
    base.update(overrides)
    return base


class EvalTests(unittest.TestCase):
    def test_memorizer_fails_the_held_out_partner(self) -> None:
        held_out = [item for item in fixed_validation_scenarios() if item.held_out]
        metrics = evaluate_policy(CueMemorizer(), held_out)
        self.assertLess(float(metrics["held_out_revision_accuracy"]), 0.5)
        self.assertGreater(float(metrics["unsafe_commit_rate"]), 0.5)
        self.assertEqual(metrics["traces"][0]["outcome"], "incorrect")  # type: ignore[index]

    def test_confirmation_flip_breaks_consistency(self) -> None:
        anchor = [item for item in fixed_validation_scenarios() if item.family == "anchor"][:1]
        metrics = evaluate_policy(FlippingPolicy(), anchor)
        self.assertEqual(float(metrics["commitment_consistency"]), 0.0)

    def test_exact_filter_confirmation_is_stable(self) -> None:
        anchor = [item for item in fixed_validation_scenarios() if item.family == "anchor"]
        metrics = evaluate_policy(ExactFilterPolicy(), anchor)
        self.assertEqual(float(metrics["commitment_consistency"]), 1.0)
        self.assertGreater(float(metrics["anchor_action_agreement"]), 0.8)
        self.assertLess(float(metrics["anchor_belief_tv"]), 0.05)

    def test_keep_rule(self) -> None:
        baseline = _blank_metrics()
        self.assertTrue(keep_candidate(baseline, _blank_metrics(BeliefUpdateScore=0.42)))
        self.assertFalse(keep_candidate(baseline, _blank_metrics(BeliefUpdateScore=0.41)))
        self.assertFalse(keep_candidate(baseline, _blank_metrics(BeliefUpdateScore=0.50, unsafe_commit_rate=0.12)))
        self.assertFalse(keep_candidate(baseline, _blank_metrics(BeliefUpdateScore=0.50, perseveration=0.16)))


if __name__ == "__main__":
    unittest.main()
