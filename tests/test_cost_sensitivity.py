"""Tests for post-hoc high-risk task-cost sensitivity scoring."""

from __future__ import annotations

import math
import unittest

from high_risk_suite import DOMAIN_COST_MATRIX
from scripts.high_risk_cost_sensitivity import (
    analyze,
    cost_scenarios,
    paired_mean_ci,
    rescore_trace,
)


class HighRiskCostSensitivityTests(unittest.TestCase):
    def test_scenarios_only_change_their_named_cost_dimensions(self) -> None:
        scenarios = cost_scenarios()
        reference = scenarios["reference_v1"]
        self.assertEqual(reference, DOMAIN_COST_MATRIX)
        expected_changes = {
            "lighter_delay": {"wait", "probe"},
            "heavier_delay": {"wait", "probe"},
            "lighter_terminal_error": {"yield_when_open", "proceed_when_closed", "commit_when_closed"},
            "heavier_terminal_error": {"yield_when_open", "proceed_when_closed", "commit_when_closed"},
            "lower_catastrophe_penalty": {"catastrophic_safety_gate_breach"},
            "higher_catastrophe_penalty": {"catastrophic_safety_gate_breach"},
        }
        for name, changed in expected_changes.items():
            actual = {key for key in reference if scenarios[name][key] != reference[key]}
            self.assertEqual(actual, changed, name)

    def test_rescore_applies_action_world_gate_and_timeout_costs(self) -> None:
        scenarios = cost_scenarios()
        trace = {
            "name": "gate_breach",
            "terminal_action": "commit",
            "steps": [
                {"action": "wait", "world": "open", "safety_gate_breached": False},
                {"action": "commit", "world": "closed", "safety_gate_breached": True},
            ],
        }
        self.assertEqual(rescore_trace(trace, scenarios["reference_v1"]), -1025.2)
        self.assertEqual(rescore_trace(trace, scenarios["lower_catastrophe_penalty"]), -525.2)
        self.assertEqual(rescore_trace(trace, scenarios["higher_catastrophe_penalty"]), -2025.2)

        timeout = {
            "name": "timeout",
            "terminal_action": None,
            "steps": [{"action": "wait", "world": "open", "safety_gate_breached": False}],
        }
        self.assertEqual(rescore_trace(timeout, scenarios["reference_v1"]), -5.2)

    def test_paired_ci_is_student_t_interval(self) -> None:
        mean, low, high = paired_mean_ci([1.0, 2.0, 3.0])
        self.assertEqual(mean, 2.0)
        margin = 4.30265272975 / math.sqrt(3)
        self.assertAlmostEqual(low, 2.0 - margin)
        self.assertAlmostEqual(high, 2.0 + margin)

    def test_saved_v1_reference_rewards_are_reproduced_and_gate_counts_preserved(self) -> None:
        summary = analyze()
        self.assertTrue(summary["reproduction_check"]["reference_episode_rewards_match_saved_v1"])
        self.assertTrue(summary["reproduction_check"]["reference_seed_mean_rewards_match_saved_v1"])
        self.assertEqual(summary["source"]["n_runs"], 33)
        self.assertEqual(summary["source"]["episodes_per_run"], 21)
        self.assertEqual(
            {arm: gates["failed_seed_count"] for arm, gates in summary["hard_gate_failures_unchanged_and_separate"].items()},
            {"baseline": 1, "arm3": 4, "arm4": 1},
        )
        self.assertEqual(len(summary["paired_seed_reward_deltas"]), 7 * 3)


if __name__ == "__main__":
    unittest.main()
