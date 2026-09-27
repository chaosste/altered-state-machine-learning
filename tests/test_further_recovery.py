"""A live stop is recoverable, and the trace names that cue."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from env import CLOSED, COMMIT, GO, STOP, WAIT, YIELD, PartnerEnv, fixed_validation_scenarios, further_study_scenarios
from eval import KEEP_MIN_SCORE_DELTA, WEIGHTS, recovery_report


FIXED_SEEDS = (13, 29, 41, 53, 67, 79, 97, 113, 131, 149, 167, 181, 199, 211, 227)


@dataclass
class Decision:
    action: int
    belief_before: np.ndarray
    belief_after: np.ndarray


class _CuePolicy:
    def __init__(self, on_stop: int) -> None:
        self.on_stop = on_stop
        self.belief = np.array([0.7, 0.3])
        self.action = WAIT

    def reset_episode(self) -> None:
        self.belief = np.array([0.7, 0.3])
        self.action = WAIT

    def act(self, obs, deterministic: bool = True) -> Decision:
        self.action = self.on_stop if obs.discrete == STOP else WAIT
        if obs.discrete == STOP:
            self.belief = np.array([0.2, 0.8])
        return Decision(self.action, np.array([0.7, 0.3]), self.belief.copy())

    def requery(self, obs) -> Decision:
        return Decision(self.action, self.belief.copy(), self.belief.copy())


class RecoveryTests(unittest.TestCase):
    def test_fixed_validation_seeds_stay_put(self) -> None:
        seeds = [scenario.seed for scenario in fixed_validation_scenarios()]
        self.assertEqual(seeds, list(FIXED_SEEDS))
        recovery = [scenario for scenario in further_study_scenarios() if scenario.family == "further_recovery"]
        self.assertEqual(len(recovery), 1)
        self.assertTrue({scenario.seed for scenario in recovery}.isdisjoint(FIXED_SEEDS))
        self.assertIsNone(fixed_validation_scenarios()[0].scripted_cues)
        self.assertEqual(WEIGHTS["commitment_consistency"], 0.10)
        self.assertEqual(KEEP_MIN_SCORE_DELTA, 0.02)

    def test_scripted_stop_arrives_as_the_world_closes(self) -> None:
        scenario = next(item for item in further_study_scenarios() if item.name == "recovery_stop")
        env = PartnerEnv()
        obs, _info = env.reset(scenario)
        cues = [obs.discrete]
        worlds = [obs.world_type]
        done = False
        while not done:
            obs, _reward, done, _info = env.step(WAIT)
            cues.append(obs.discrete)
            worlds.append(obs.world_type)
        self.assertEqual(cues[0], GO)
        self.assertEqual(cues[1], GO)
        self.assertEqual(cues[2], STOP)
        self.assertEqual(worlds[0], 0)
        self.assertEqual(worlds[2], CLOSED)

    def test_yield_recovers_and_commit_does_not(self) -> None:
        recovered = recovery_report(_CuePolicy(YIELD))
        committed = recovery_report(_CuePolicy(COMMIT))
        self.assertEqual(recovered["recovery_rate"], 1.0)
        self.assertEqual(committed["recovery_rate"], 0.0)
        self.assertTrue(recovered["trace_names_the_cue"])
        self.assertTrue(committed["trace_names_the_cue"])
        self.assertTrue(recovered["requery_stable"])
        self.assertTrue(committed["requery_stable"])
        self.assertFalse(recovered["changes_belief_update_score"])


if __name__ == "__main__":
    unittest.main()
