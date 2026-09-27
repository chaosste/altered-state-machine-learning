"""Enduring revision and refinement stay off the keep rule."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from env import WAIT, fixed_validation_scenarios, further_study_scenarios
from eval import KEEP_MIN_SCORE_DELTA, WEIGHTS, WINDOW_CLOSED_EPISODE, enduring_precision_report
from oracle import evidence_update, initial_logits, softmax
from train import phenotype_constants


FIXED_SEEDS = (13, 29, 41, 53, 67, 79, 97, 113, 131, 149, 167, 181, 199, 211, 227)


@dataclass
class Decision:
    action: int
    belief_before: np.ndarray
    belief_after: np.ndarray


class FilterPolicy:
    def __init__(self, pe_gain: float = 1.0, precision: float = 0.0, read_closed_window: bool = False) -> None:
        self._pe_gain = pe_gain
        self._precision = precision
        self.read_closed_window = read_closed_window
        self.pe_gain = pe_gain
        self.logits = initial_logits(precision)
        self.prev = WAIT
        self.action = WAIT

    def reset_episode(self) -> None:
        if self.read_closed_window:
            constants = phenotype_constants(WINDOW_CLOSED_EPISODE)
            self.pe_gain = float(constants["pe_gain"])
            self._precision = float(constants["prior_precision"])
        self.logits = initial_logits(self._precision)
        self.prev = WAIT
        self.action = WAIT

    def act(self, obs, deterministic: bool = True) -> Decision:
        before = softmax(self.logits)
        self.logits = evidence_update(self.logits, obs, self.prev, self.pe_gain)
        after = softmax(self.logits)
        self.action = WAIT
        self.prev = self.action
        return Decision(self.action, before.copy(), after.copy())

    def requery(self, obs) -> Decision:
        after = softmax(self.logits)
        return Decision(self.action, after.copy(), after.copy())


class FlatPolicy:
    def reset_episode(self) -> None:
        self.belief = np.array([0.5, 0.5])
        self.pe_gain = 0.0
        self.action = WAIT

    def act(self, obs, deterministic: bool = True) -> Decision:
        return Decision(self.action, self.belief.copy(), self.belief.copy())

    def requery(self, obs) -> Decision:
        return Decision(self.action, self.belief.copy(), self.belief.copy())


class EnduranceTests(unittest.TestCase):
    def test_fixed_validation_seeds_stay_put(self) -> None:
        seeds = [scenario.seed for scenario in fixed_validation_scenarios()]
        self.assertEqual(seeds, list(FIXED_SEEDS))
        extra = {scenario.seed for scenario in further_study_scenarios()}
        self.assertTrue(extra.isdisjoint(FIXED_SEEDS))
        self.assertEqual(WEIGHTS["revision_accuracy"], 0.25)
        self.assertEqual(KEEP_MIN_SCORE_DELTA, 0.02)

    def test_flat_belief_fails_refinement_and_exact_filter_is_closer(self) -> None:
        flat = enduring_precision_report(FlatPolicy())
        exact = enduring_precision_report(FilterPolicy())
        self.assertEqual(flat["refinement_entropy_in_band"], 0.0)
        self.assertLess(exact["refinement_tv"], flat["refinement_tv"])
        self.assertGreater(exact["enduring_revision"], flat["enduring_revision"])
        self.assertFalse(exact["changes_belief_update_score"])
        self.assertEqual(exact["window_closed_episode"], WINDOW_CLOSED_EPISODE)

    def test_report_reads_the_closed_window(self) -> None:
        policy = FilterPolicy(read_closed_window=True)
        report = enduring_precision_report(policy)
        closed = phenotype_constants(WINDOW_CLOSED_EPISODE)
        self.assertEqual(report["pe_gain"], closed["pe_gain"])
        self.assertEqual(report["window_closed_episode"], 10**9)


if __name__ == "__main__":
    unittest.main()
