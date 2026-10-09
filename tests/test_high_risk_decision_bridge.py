"""Tests for local belief-to-action counterfactual logging."""

from __future__ import annotations

import copy
import unittest

import torch

from high_risk_suite import evaluate_closed_loop, high_risk_cases
from train import BaselinePolicy, apply_variant


class DecisionBridgeTests(unittest.TestCase):
    def test_reference_belief_logs_local_choice_without_changing_actual_run(self) -> None:
        apply_variant("baseline")
        torch.manual_seed(13)
        initial_policy = BaselinePolicy(hidden_dim=8, device="cpu")
        state = copy.deepcopy(initial_policy.state_dict())
        without_reference = BaselinePolicy(hidden_dim=8, device="cpu")
        without_reference.load_state_dict(state)
        with_reference = BaselinePolicy(hidden_dim=8, device="cpu")
        with_reference.load_state_dict(state)
        case = high_risk_cases()[0]

        _metrics, ordinary_traces = evaluate_closed_loop(without_reference, cases=[case])
        _metrics, bridge_traces = evaluate_closed_loop(
            with_reference,
            cases=[case],
            belief_filter={"prior_precision": 1.5, "pe_gain": 0.35},
            decision_reference_filter={"prior_precision": 0.5, "pe_gain": 1.0},
        )

        ordinary_steps = ordinary_traces[0]["steps"]
        bridge_steps = bridge_traces[0]["steps"]
        self.assertEqual([step["action"] for step in ordinary_steps], [step["action"] for step in bridge_steps])
        self.assertTrue(bridge_steps)
        for step in bridge_steps:
            self.assertIn("decision_reference_action", step)
            self.assertIn("decision_changed_under_reference_belief", step)
            self.assertIn("decision_reference_immediate_reward_delta", step)
            self.assertIn("decision_reference_would_breach_safety_gate", step)


if __name__ == "__main__":
    unittest.main()
