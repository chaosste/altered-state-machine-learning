"""Exact filter and the anchor belief MDP."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from env import CLOSED, COMMIT, GO, NONE, OBS_DIM, OPEN, WAIT, YIELD, Observation
from oracle import anchor_action, anchor_value, evidence_update, softmax


def _obs(**kwargs) -> Observation:
    base = Observation(
        discrete=NONE,
        private_present=False,
        private_says_open=False,
        query_is_partner=False,
        requery=False,
        access_missing=False,
        mask_update=False,
        step_index=0,
        max_steps=6,
        intervening=False,
        clarity=0.0,
        generated_by=WAIT,
        public_informativeness=1.0,
        vector=np.zeros(OBS_DIM, dtype=np.float32),
        label=OPEN,
        world_type=OPEN,
        partner_belief=OPEN,
        post_reversal=False,
        pre_reversal_world=OPEN,
        family="anchor",
        scenario_name="unit",
        ambiguous=False,
        held_out=False,
        novel_reversal=False,
        score_from_step=0,
    )
    return replace(base, **kwargs)


class OracleTests(unittest.TestCase):
    def test_one_go_cue_from_a_uniform_prior(self) -> None:
        posterior = softmax(evidence_update(np.zeros(2), _obs(discrete=GO), WAIT, pe_gain=1.0))
        self.assertAlmostEqual(float(posterior[OPEN]), 0.75 / 0.85, places=6)

    def test_omission_supports_closed(self) -> None:
        posterior = softmax(evidence_update(np.zeros(2), _obs(discrete=NONE), WAIT, pe_gain=1.0))
        self.assertAlmostEqual(float(posterior[CLOSED]), 0.45 / 0.55, places=6)

    def test_requery_and_missing_access_do_not_move_the_filter(self) -> None:
        start = np.array([0.2, -0.2])
        self.assertTrue(np.allclose(evidence_update(start, _obs(requery=True, discrete=GO), WAIT, 1.0), start))
        self.assertTrue(
            np.allclose(evidence_update(start, _obs(access_missing=True, discrete=GO), WAIT, 1.0), start)
        )

    def test_private_evidence_can_oppose_a_public_go(self) -> None:
        logits = evidence_update(np.zeros(2), _obs(discrete=GO), WAIT, pe_gain=1.0)
        logits = evidence_update(
            logits,
            _obs(discrete=GO, private_present=True, private_says_open=False),
            WAIT,
            pe_gain=1.0,
        )
        self.assertGreater(float(softmax(logits)[CLOSED]), 0.5)

    def test_anchor_policy_commits_only_when_the_partner_is_almost_surely_open(self) -> None:
        self.assertEqual(anchor_action(0.99, steps_left=6), COMMIT)
        self.assertEqual(anchor_action(0.01, steps_left=6), YIELD)
        self.assertGreater(anchor_value(0.99, steps_left=6), anchor_value(0.01, steps_left=6))
        self.assertNotEqual(anchor_action(0.5, steps_left=6), COMMIT)


if __name__ == "__main__":
    unittest.main()
