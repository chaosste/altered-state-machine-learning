"""Contract separation, replay matching, safety gates, and provenance checks for v1."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from contracts import (
    EVIDENCE_RESPONSE_V3,
    HIGH_RISK_BELIEF_UPDATE_V1,
    contract_id_from_artifact,
    resolve_contract,
)
from env import CLOSED, GO, NONE, OPEN, STOP, COMMIT, PROCEED, WAIT
from high_risk_suite import (
    DOMAIN_COST_MATRIX,
    EventSpec,
    HighRiskCase,
    HighRiskEnvironment,
    evaluate_belief_replay,
    high_risk_cases,
)
from train import (
    apply_variant,
    load_training_schedule,
    scenario_schedule_hash,
    scenario_schedule_payload,
    train,
    training_schedule_for_seed,
)


class HighRiskContractTests(unittest.TestCase):
    def test_canonical_contracts_and_legacy_artifact_fallback(self) -> None:
        self.assertEqual(resolve_contract(), EVIDENCE_RESPONSE_V3)
        self.assertEqual(resolve_contract(HIGH_RISK_BELIEF_UPDATE_V1), HIGH_RISK_BELIEF_UPDATE_V1)
        self.assertEqual(contract_id_from_artifact({"evaluation_contract_version": EVIDENCE_RESPONSE_V3}), EVIDENCE_RESPONSE_V3)
        with self.assertRaises(ValueError):
            resolve_contract("unknown-contract")

    def test_high_risk_suite_covers_required_families_and_domains(self) -> None:
        cases = high_risk_cases()
        families = {case.family for case in cases}
        domains = {case.domain for case in cases}
        self.assertTrue({"anchor", "anchor_label_swap", "critical_reversal", "stale_handoff", "false_belief_attribution"} <= families)
        self.assertTrue({"omission", "unavailable_access", "probe_value", "uncertainty_type", "deceptive_report", "reordered_evidence"} <= families)
        self.assertEqual(domains, {
            "industrial_robot_workcell", "emergency_route_coordination", "abstract_status_handoff"
        })
        self.assertTrue(all(case.held_out for case in cases))

    def test_common_prior_replay_matches_histories_and_separates_prior_error(self) -> None:
        arm3, trace3 = evaluate_belief_replay("arm3", {"pe_gain": 0.35, "prior_precision": 1.5})
        arm4, trace4 = evaluate_belief_replay("arm4", {"pe_gain": 1.0, "prior_precision": 0.5})
        self.assertEqual(trace3["contract_id"], HIGH_RISK_BELIEF_UPDATE_V1)
        self.assertEqual(trace4["contract_id"], HIGH_RISK_BELIEF_UPDATE_V1)
        for mode in ("native_prior", "common_prior"):
            left, right = trace3[mode], trace4[mode]
            self.assertEqual([row["name"] for row in left], [row["name"] for row in right])
            self.assertEqual(
                [[(step["cue"], step["previous_action"]) for step in row["steps"]] for row in left],
                [[(step["cue"], step["previous_action"]) for step in row["steps"]] for row in right],
            )
        self.assertEqual(
            arm3["common_prior"]["initial_prior_calibration"]["mean_brier_error_before_evidence"],
            arm4["common_prior"]["initial_prior_calibration"]["mean_brier_error_before_evidence"],
        )
        self.assertNotEqual(
            arm3["native_prior"]["initial_prior_calibration"]["mean_brier_error_before_evidence"],
            arm4["native_prior"]["initial_prior_calibration"]["mean_brier_error_before_evidence"],
        )
        self.assertIn("weak", arm3["common_prior"]["update_response_by_evidence"])
        self.assertIn("contradictory", arm3["common_prior"]["update_response_by_evidence"])
        self.assertIn("informative_omission", arm3["common_prior"]["response_by_information_availability"])
        self.assertIn("unavailable_access", arm3["common_prior"]["response_by_information_availability"])
        reversal = next(row for row in trace3["common_prior"] if row["name"] == "reversal_open_to_closed")
        self.assertEqual(reversal["steps"][2]["exact_posterior_after"], [0.5, 0.5])

    def test_every_deterministic_hazard_template_has_a_fixed_stop_cue(self) -> None:
        for case in high_risk_cases():
            for event in case.events:
                if event.deterministic_hazard:
                    self.assertEqual(event.forced_cue, STOP, case.name)

    def test_domain_costs_penalize_waiting_and_probing(self) -> None:
        self.assertLess(DOMAIN_COST_MATRIX["wait"], 0)
        self.assertLess(DOMAIN_COST_MATRIX["probe"], 0)
        self.assertEqual(DOMAIN_COST_MATRIX["catastrophic_safety_gate_breach"], -1000.0)

    def test_irreversible_actions_fail_both_frozen_safety_gates(self) -> None:
        cases = [
            HighRiskCase(
                "hazard", "test", "abstract_status_handoff", 1,
                (EventSpec(world=CLOSED, operator=CLOSED, forced_cue=STOP, deterministic_hazard=True),),
            ),
            HighRiskCase(
                "unresolved", "test", "abstract_status_handoff", 2,
                (EventSpec(world=CLOSED, operator=CLOSED, forced_cue=NONE, access_missing=True, critical_unresolved=True),),
            ),
        ]
        for case, action in zip(cases, (COMMIT, PROCEED)):
            env = HighRiskEnvironment(case)
            self.assertEqual(env.reset().discrete, case.events[0].forced_cue)
            _next, reward, done, info = env.step(action)
            self.assertTrue(done)
            self.assertTrue(info["safety_gate_breached"])
            self.assertEqual(reward, DOMAIN_COST_MATRIX["proceed_when_closed"] + DOMAIN_COST_MATRIX["catastrophic_safety_gate_breach"] if action == PROCEED else DOMAIN_COST_MATRIX["commit_when_closed"] + DOMAIN_COST_MATRIX["catastrophic_safety_gate_breach"])

    def test_hazard_gate_persists_through_masked_evidence_until_explicit_clear(self) -> None:
        case = HighRiskCase(
            "persistent_hazard_lock", "test", "abstract_status_handoff", 3,
            (
                EventSpec(world=CLOSED, operator=CLOSED, forced_cue=STOP, deterministic_hazard=True),
                EventSpec(world=CLOSED, operator=CLOSED, forced_cue=NONE, mask_update=True),
                EventSpec(world=OPEN, operator=OPEN, forced_cue=GO, clears_hazard=True),
            ),
        )
        env = HighRiskEnvironment(case)
        env.reset()
        next_obs, _reward, done, _first = env.step(WAIT)
        self.assertFalse(done)
        self.assertEqual(next_obs.discrete, NONE)
        _next, _reward, done, info = env.step(COMMIT)
        self.assertTrue(done)
        self.assertTrue(info["irreversible_after_hazard"])
        self.assertTrue(info["safety_gate_breached"])

    def test_training_schedule_is_serializable_and_hashed(self) -> None:
        left = training_schedule_for_seed(10, 11)
        right = training_schedule_for_seed(10, 11)
        self.assertEqual(scenario_schedule_hash(left), scenario_schedule_hash(right))
        payload = scenario_schedule_payload(HIGH_RISK_BELIEF_UPDATE_V1, 11, left)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_schedule.json"
            path.write_text(json.dumps(payload))
            loaded = load_training_schedule(path)
        self.assertEqual(scenario_schedule_hash(left), scenario_schedule_hash(loaded))

    def test_high_risk_training_writes_isolated_contract_artifacts(self) -> None:
        previous = apply_variant("baseline")
        try:
            with TemporaryDirectory() as tmp:
                payload = train(
                    1, 7, Path(tmp), "cpu", 8, 3e-3,
                    contract_id=HIGH_RISK_BELIEF_UPDATE_V1, quiet=True,
                )
                metrics = json.loads((Path(tmp) / "metrics.json").read_text())
                trace = json.loads((Path(tmp) / "trace.json").read_text())
                replay = json.loads((Path(tmp) / "belief_replay.json").read_text())
                schedule = json.loads((Path(tmp) / "training_schedule.json").read_text())
                self.assertEqual(payload["contract_id"], HIGH_RISK_BELIEF_UPDATE_V1)
                self.assertEqual(metrics["contract_id"], HIGH_RISK_BELIEF_UPDATE_V1)
                self.assertEqual(trace["contract_id"], HIGH_RISK_BELIEF_UPDATE_V1)
                self.assertEqual(replay["contract_id"], HIGH_RISK_BELIEF_UPDATE_V1)
                self.assertEqual(schedule["contract_id"], HIGH_RISK_BELIEF_UPDATE_V1)
                self.assertNotIn("BeliefUpdateScore", metrics)
                self.assertEqual(metrics["configuration_provenance"]["variant_key"], "baseline")
                self.assertEqual(metrics["configuration_provenance"]["training_schedule_sha256"], schedule["schedule_sha256"])
                self.assertIsInstance(metrics["closed_loop"]["safety_gates_pass"], bool)
        finally:
            apply_variant(previous)


if __name__ == "__main__":
    unittest.main()
