"""Next-action credit on varied scenes. A stored trajectory misses the switch."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from env import CLOSED, COMMIT, OPEN, YIELD, fixed_validation_scenarios, further_study_scenarios
from eval import KEEP_MIN_SCORE_DELTA, WEIGHTS, varied_tom_report


FIXED_SEEDS = (13, 29, 41, 53, 67, 79, 97, 113, 131, 149, 167, 181, 199, 211, 227)


@dataclass
class Decision:
    action: int
    belief_before: np.ndarray
    belief_after: np.ndarray


class LabelPolicy:
    """Uses the harness label. That is the appropriate next action for the question asked."""

    def reset_episode(self) -> None:
        self.belief = np.array([0.5, 0.5])
        self.action = YIELD

    def act(self, obs, deterministic: bool = True) -> Decision:
        if obs.label == OPEN:
            self.action = COMMIT
            self.belief = np.array([0.9, 0.1])
        else:
            self.action = YIELD
            self.belief = np.array([0.1, 0.9])
        return Decision(self.action, self.belief.copy(), self.belief.copy())

    def requery(self, obs) -> Decision:
        return Decision(self.action, self.belief.copy(), self.belief.copy())


class TrajectoryMemorizer:
    """Replays a stored commit sequence from a training family that treated go as commit."""

    def reset_episode(self) -> None:
        self.belief = np.array([0.95, 0.05])
        self.action = COMMIT

    def act(self, obs, deterministic: bool = True) -> Decision:
        self.action = COMMIT
        return Decision(self.action, self.belief.copy(), self.belief.copy())

    def requery(self, obs) -> Decision:
        return Decision(self.action, self.belief.copy(), self.belief.copy())


class VariedTomTests(unittest.TestCase):
    def test_fixed_validation_seeds_stay_put(self) -> None:
        seeds = [scenario.seed for scenario in fixed_validation_scenarios()]
        self.assertEqual(seeds, list(FIXED_SEEDS))
        varied = [scenario for scenario in further_study_scenarios() if scenario.family == "further_varied"]
        self.assertTrue({scenario.seed for scenario in varied}.isdisjoint(FIXED_SEEDS))
        self.assertEqual(WEIGHTS["safe_commit"], 0.15)
        self.assertEqual(KEEP_MIN_SCORE_DELTA, 0.02)

    def test_scenes_are_loosely_related_and_paired(self) -> None:
        varied = [scenario for scenario in further_study_scenarios() if scenario.family == "further_varied"]
        names = {scenario.name for scenario in varied}
        self.assertIn("varied_world_true", names)
        self.assertIn("varied_world_false", names)
        self.assertIn("varied_partner_true", names)
        self.assertIn("varied_partner_false", names)
        self.assertIn("varied_whose_belief", names)
        self.assertIn("varied_next_action", names)
        timings = {(scenario.score_from_step, scenario.private_from_step, scenario.max_steps) for scenario in varied}
        self.assertGreater(len(timings), 1)
        whose = next(scenario for scenario in varied if scenario.name == "varied_whose_belief")
        self.assertEqual(whose.query_target, "partner")
        self.assertEqual(whose.partner_belief, CLOSED)
        self.assertEqual(whose.world_type, OPEN)

    def test_label_policy_passes_and_memorizer_misses_the_switch(self) -> None:
        good = varied_tom_report(LabelPolicy())
        memorized = varied_tom_report(TrajectoryMemorizer())
        self.assertEqual(good["next_action_accuracy"], 1.0)
        self.assertTrue(good["whose_belief_correct"])
        self.assertLess(memorized["next_action_accuracy"], 1.0)
        self.assertFalse(memorized["whose_belief_correct"])
        self.assertFalse(memorized["changes_belief_update_score"])
        self.assertEqual(good["n_scenarios"], 6)


if __name__ == "__main__":
    unittest.main()
