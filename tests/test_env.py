"""Frozen environment invariants."""

from __future__ import annotations

import random
import unittest

import numpy as np

from env import (
    BASE_LIKELIHOOD,
    CLOSED,
    OPEN,
    PROBE,
    PROBE_LIKELIHOOD,
    WAIT,
    PartnerEnv,
    fixed_validation_scenarios,
    perseverative_action,
    sample_training_scenario,
)
from oracle import rows_are_distributions


class EnvTests(unittest.TestCase):
    def test_likelihood_rows_are_distributions(self) -> None:
        self.assertTrue(rows_are_distributions())
        self.assertTrue(np.all(BASE_LIKELIHOOD > 0))
        self.assertTrue(np.all(PROBE_LIKELIHOOD > 0))

    def test_validation_suite_is_stable(self) -> None:
        first = fixed_validation_scenarios()
        second = fixed_validation_scenarios()
        self.assertEqual([item.name for item in first], [item.name for item in second])
        self.assertEqual([item.seed for item in first], [item.seed for item in second])
        self.assertEqual(len({item.seed for item in first}), len(first))
        self.assertEqual(sum(item.held_out for item in first), 1)
        self.assertEqual(sum(item.novel_reversal for item in first), 2)
        self.assertEqual(sum(item.ambiguous for item in first), 1)
        self.assertEqual(sum(item.family == "anchor" for item in first), 2)

    def test_scripted_rollout_is_deterministic(self) -> None:
        scenario = fixed_validation_scenarios()[0]

        def roll() -> list[int]:
            env = PartnerEnv()
            obs, _info = env.reset(scenario)
            seen = [obs.discrete]
            done = False
            while not done:
                obs, _reward, done, _info = env.step(WAIT)
                seen.append(obs.discrete)
            return seen

        self.assertEqual(roll(), roll())

    def test_probe_marks_the_next_cue_as_clear(self) -> None:
        env = PartnerEnv()
        _obs, _info = env.reset(fixed_validation_scenarios()[0])
        obs, _reward, _done, _info = env.step(PROBE)
        self.assertEqual(obs.clarity, 1.0)
        self.assertEqual(obs.generated_by, PROBE)

    def test_novel_reversal_has_a_gap_then_a_flip(self) -> None:
        scenario = next(item for item in fixed_validation_scenarios() if item.name == "novel_reversal_after_gap")
        env = PartnerEnv()
        obs, _info = env.reset(scenario)
        seen = [obs]
        done = False
        while not done:
            obs, _reward, done, _info = env.step(WAIT)
            seen.append(obs)
        by_step = {item.step_index: item for item in seen}
        self.assertTrue(by_step[3].intervening and by_step[3].mask_update)
        self.assertTrue(by_step[5].intervening)
        self.assertFalse(by_step[6].intervening)
        self.assertTrue(by_step[6].post_reversal)
        self.assertEqual(by_step[6].world_type, OPEN)
        self.assertEqual(by_step[0].world_type, CLOSED)

    def test_private_evidence_contradicts_the_public_partner(self) -> None:
        scenario = next(item for item in fixed_validation_scenarios() if item.name == "false_belief_private")
        env = PartnerEnv()
        obs, _info = env.reset(scenario)
        self.assertFalse(obs.private_present)
        obs, _reward, _done, _info = env.step(WAIT)
        self.assertTrue(obs.private_present)
        self.assertFalse(obs.private_says_open)
        self.assertEqual(obs.world_type, CLOSED)
        self.assertEqual(obs.partner_belief, OPEN)

    def test_training_draws_avoid_the_held_out_partner_and_validation_seeds(self) -> None:
        rng = random.Random(7)
        validation_seeds = {item.seed for item in fixed_validation_scenarios()}
        for _ in range(40):
            scenario = sample_training_scenario(rng)
            self.assertFalse(scenario.held_out)
            self.assertFalse(scenario.novel_reversal)
            self.assertNotIn(scenario.seed, validation_seeds)

    def test_perseveration_follows_the_old_partner_model(self) -> None:
        from env import COMMIT, YIELD

        self.assertTrue(perseverative_action(COMMIT, OPEN))
        self.assertTrue(perseverative_action(YIELD, CLOSED))
        self.assertFalse(perseverative_action(YIELD, OPEN))


if __name__ == "__main__":
    unittest.main()
