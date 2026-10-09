"""Baseline trainer smoke and the neutral phenotype constants."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from train import (
        BELIEF_PE_GAIN,
        PHASE_DEPENDENT_SENSITIVITY,
        PLASTICITY_UNTIL_EPISODE,
        PRIOR_PRECISION,
        PUNISHMENT_LEARNING_RATE,
        REWARD_LEARNING_RATE,
        BaselinePolicy,
        phenotype_constants,
        train,
    )

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TrainTests(unittest.TestCase):
    def test_baseline_constants_are_the_placebo_point(self) -> None:
        self.assertEqual(REWARD_LEARNING_RATE, PUNISHMENT_LEARNING_RATE)
        self.assertFalse(PHASE_DEPENDENT_SENSITIVITY)
        self.assertIsNone(PLASTICITY_UNTIL_EPISODE)
        self.assertLess(BELIEF_PE_GAIN, 1.0)
        self.assertGreater(PRIOR_PRECISION, 1.0)
        self.assertEqual(phenotype_constants(0)["pe_gain"], BELIEF_PE_GAIN)
        self.assertEqual(phenotype_constants(10**9)["pe_gain"], BELIEF_PE_GAIN)

    def test_rebus_arm_is_a_named_switch(self) -> None:
        from train import apply_variant, active_variant

        previous = apply_variant("baseline")
        try:
            apply_variant("rebus")
            self.assertEqual(active_variant(), "arm4_rebus")
            open_window = phenotype_constants(0)
            self.assertEqual(open_window["prior_precision"], 0.5)
            self.assertEqual(open_window["pe_gain"], 1.0)
            self.assertEqual(open_window["reward_lr"], REWARD_LEARNING_RATE)
            apply_variant("arm5")
            self.assertEqual(phenotype_constants(0)["pe_gain"], 1.0)
            self.assertEqual(phenotype_constants(10**9)["pe_gain"], BELIEF_PE_GAIN)
            self.assertEqual(phenotype_constants(10**9)["prior_precision"], PRIOR_PRECISION)
        finally:
            apply_variant(previous)

    def test_requery_does_not_move_the_belief(self) -> None:
        from env import PartnerEnv, fixed_validation_scenarios

        policy = BaselinePolicy(hidden_dim=8, device="cpu")
        policy.reset_episode()
        env = PartnerEnv()
        obs, _info = env.reset(fixed_validation_scenarios()[0])
        policy.act(obs, deterministic=True)
        logits = policy.logits.copy()
        policy.requery(obs)
        self.assertTrue((policy.logits == logits).all())

    def test_one_episode_writes_the_eval_contract(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            payload = train(episodes=1, seed=7, output_dir=output, device="cpu", hidden_dim=16, lr=3e-3)
            self.assertIn("BeliefUpdateScore", payload)
            self.assertNotIn("traces", payload)
            self.assertEqual(payload["variant"], "baseline")
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["contract_id"], "evidence-response-v3")
            self.assertEqual(metrics["BeliefUpdateScore"], payload["BeliefUpdateScore"])
            traces = json.loads((output / "trace.json").read_text())
            self.assertEqual(traces["contract_id"], "evidence-response-v3")
            self.assertGreaterEqual(len(traces["traces"]), 1)
            self.assertIn("belief_after", traces["traces"][0]["steps"][0])
            self.assertTrue((output / "model.pt").exists())
            self.assertTrue((output / "learning_curve.csv").exists())


if __name__ == "__main__":
    unittest.main()
