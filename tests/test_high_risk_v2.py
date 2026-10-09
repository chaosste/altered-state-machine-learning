from __future__ import annotations

import unittest

from env import STOP
from high_risk_suite import (
    BASELINE_PRIOR_PRECISION,
    evaluate_belief_replay,
    evaluate_closed_loop,
    high_risk_cases,
)
from high_risk_v2_suite import (
    SCENARIO_SUITE_V2_ID,
    additional_heldout_cases,
    expanded_heldout_cases,
)


class HeldoutExpansionTests(unittest.TestCase):
    def test_v2_adds_distinct_heldout_templates_across_all_domains(self) -> None:
        original = high_risk_cases()
        additions = additional_heldout_cases()
        combined = expanded_heldout_cases()
        self.assertEqual(len(original), 21)
        self.assertEqual(len(additions), 12)
        self.assertEqual(len(combined), len(original) + len(additions))
        self.assertFalse({case.name for case in original} & {case.name for case in additions})
        self.assertTrue(all(case.name.startswith("v2_") and case.held_out for case in additions))
        self.assertEqual(
            {case.domain for case in additions},
            {"industrial_robot_workcell", "emergency_route_coordination", "abstract_status_handoff"},
        )

    def test_every_new_deterministic_hazard_template_forces_stop(self) -> None:
        for case in additional_heldout_cases():
            for event in case.events:
                if event.deterministic_hazard:
                    self.assertEqual(event.forced_cue, STOP, case.name)

    def test_replay_and_closed_loop_accept_a_versioned_custom_case_set(self) -> None:
        cases = additional_heldout_cases()[:1]
        replay, replay_trace = evaluate_belief_replay(
            "baseline",
            {"pe_gain": 0.35, "prior_precision": BASELINE_PRIOR_PRECISION},
            cases=cases,
            scenario_suite_id=SCENARIO_SUITE_V2_ID,
        )
        self.assertEqual(replay_trace["scenario_suite_id"], SCENARIO_SUITE_V2_ID)
        self.assertEqual(replay_trace["native_prior"][0]["scenario_suite_id"], SCENARIO_SUITE_V2_ID)
        self.assertEqual(replay["native_prior"]["n_cases"], 1)

        try:
            from train import BaselinePolicy, apply_variant
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        apply_variant("baseline")
        policy = BaselinePolicy(hidden_dim=8, device="cpu")
        task, traces = evaluate_closed_loop(policy, cases=cases, scenario_suite_id=SCENARIO_SUITE_V2_ID)
        self.assertEqual(task["scenario_suite_id"], SCENARIO_SUITE_V2_ID)
        self.assertEqual(traces[0]["scenario_suite_id"], SCENARIO_SUITE_V2_ID)
        self.assertEqual(traces[0]["name"], cases[0].name)


if __name__ == "__main__":
    unittest.main()

