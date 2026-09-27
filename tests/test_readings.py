"""Later readings stay off the reward and off BeliefUpdateScore."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from env import ACTIONS, CLOSED, COMMIT, OPEN, WAIT, YIELD, Observation, fixed_validation_scenarios, immediate_reward
from eval import CONFIDENT_GAP, LN2, evaluate_policy
from readings import DIFFUSE_ENTROPY, POLARITY, RICHNESS_LOW, cue_stance, read_trace, read_traces
from scripts.read_trace import write_readings


@dataclass
class Decision:
    action: int
    belief_before: np.ndarray
    belief_after: np.ndarray


class FlatPolicy:
    def reset_episode(self) -> None:
        self.belief = np.array([0.5, 0.5])

    def act(self, obs: Observation, deterministic: bool = True) -> Decision:
        before = self.belief.copy()
        return Decision(WAIT, before, self.belief.copy())

    def requery(self, obs: Observation) -> Decision:
        return Decision(WAIT, self.belief.copy(), self.belief.copy())


def _step(cue: str, before, after, entropy: float, masked: bool = False) -> dict:
    return {
        "cue": cue,
        "masked": masked,
        "belief_before": before,
        "belief_after": after,
        "epistemic_entropy": entropy,
        "action": "wait",
    }


class ReadingTests(unittest.TestCase):
    def test_cutoffs_match_the_belief_classifier(self) -> None:
        self.assertEqual(CONFIDENT_GAP, 0.05)
        self.assertAlmostEqual(DIFFUSE_ENTROPY, 0.9 * LN2)
        self.assertLess(RICHNESS_LOW, DIFFUSE_ENTROPY)

    def test_stance_and_polarity_name_the_cue(self) -> None:
        trace = {
            "name": "fixture",
            "steps": [
                _step("go", [0.5, 0.5], [0.8, 0.2], 0.50),
                _step("stop", [0.8, 0.2], [0.3, 0.7], 0.61),
                _step("none", [0.3, 0.7], [0.3, 0.7], 0.61),
                _step("go", [0.3, 0.7], [0.3, 0.7], 0.61, masked=True),
            ],
        }
        report = read_trace(trace)
        self.assertEqual(report["cue_stance"], ["approach", "avoid", "withhold", "unavailable"])
        self.assertEqual([POLARITY[name] for name in report["cue_stance"]], [1, -1, 0, 0])
        self.assertEqual(report["cue_polarity_sum"], 0)
        self.assertFalse(report["cue_polarity_used_as_reward"])
        self.assertIsNone(report["selection_metric"])
        self.assertFalse(report["changes_belief_update_score"])

    def test_richness_uses_the_entropy_band(self) -> None:
        low = read_trace({"name": "low", "steps": [_step("go", [0.99, 0.01], [0.99, 0.01], 0.05)]})
        mid = read_trace({"name": "mid", "steps": [_step("go", [0.8, 0.2], [0.7, 0.3], 0.50)]})
        high = read_trace(
            {"name": "high", "steps": [_step("none", [0.5, 0.5], [0.52, 0.48], 0.9 * math.log(2.0) + 0.01)]}
        )
        self.assertEqual(low["richness"][0]["band"], "over_precise")
        self.assertEqual(low["richness"][0]["call_after"], "open")
        self.assertFalse(low["richness"][0]["call_changed"])
        self.assertEqual(mid["richness"][0]["band"], "in_band")
        self.assertEqual(mid["richness"][0]["call_before"], "open")
        self.assertEqual(mid["richness"][0]["call_after"], "open")
        self.assertAlmostEqual(mid["richness"][0]["belief_movement"], 0.1)
        self.assertEqual(high["richness"][0]["band"], "diffuse")
        self.assertEqual(high["richness"][0]["call_after"], "undetermined")

    def test_a_call_change_is_visible_and_unscored(self) -> None:
        report = read_trace(
            {"name": "flip", "steps": [_step("stop", [0.85, 0.15], [0.15, 0.85], 0.42)]}
        )
        self.assertTrue(report["richness"][0]["call_changed"])
        self.assertEqual(report["partner_call_changes"], 1)
        self.assertNotIn("reward", report)

    def test_readings_leave_the_reward_and_the_score_alone(self) -> None:
        rewards = {(action, world): immediate_reward(action, world) for action in ACTIONS for world in (OPEN, CLOSED)}
        scenario = fixed_validation_scenarios()[0]
        metrics = evaluate_policy(FlatPolicy(), [scenario])
        score = metrics["BeliefUpdateScore"]
        keys = set(metrics)
        report = read_traces(metrics["traces"])  # type: ignore[arg-type]
        self.assertEqual(metrics["BeliefUpdateScore"], score)
        self.assertEqual(set(metrics), keys)
        self.assertFalse(report["changes_belief_update_score"])
        self.assertEqual(
            rewards,
            {(action, world): immediate_reward(action, world) for action in ACTIONS for world in (OPEN, CLOSED)},
        )
        self.assertEqual(rewards[(COMMIT, OPEN)][0], 1.0 - 0.01)
        self.assertTrue(rewards[(COMMIT, OPEN)][1])
        self.assertEqual(rewards[(YIELD, CLOSED)][0], 0.40 - 0.01)

    def test_writer_refuses_to_replace_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.json"
            metrics = root / "metrics.json"
            trace.write_text(json.dumps([{"name": "anchor_open", "steps": [_step("go", [0.6, 0.4], [0.7, 0.3], 0.4)]}]))
            metrics.write_text("{}\n")
            with self.assertRaises(ValueError):
                write_readings(trace, metrics)
            self.assertEqual(metrics.read_text(), "{}\n")
            written = write_readings(trace, root / "trace.readings.json")
            payload = json.loads(written.read_text())
            self.assertEqual(payload["instrument"], "later_readings")
            self.assertEqual(payload["scenarios"][0]["cue_stance"], ["approach"])
            self.assertIsNone(payload["selection_metric"])


class StanceHelperTests(unittest.TestCase):
    def test_unknown_cue_is_unavailable(self) -> None:
        self.assertEqual(cue_stance({"cue": "smile", "masked": False}), "unavailable")
        self.assertEqual(cue_stance({"cue": "go", "masked": True}), "unavailable")


if __name__ == "__main__":
    unittest.main()
