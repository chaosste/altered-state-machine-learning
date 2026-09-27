"""Partner POMDP. Frozen during a search. See program.md.

The hidden variable is a partner model. Cues are noisy. `probe` spends a step
to sharpen the next cue. `commit` ends the episode and is unsafe when the
world is closed. False-belief items can show a public cue that follows the
partner's belief while a private channel follows the world.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Dict, List, Optional, Tuple

import numpy as np


WAIT = 0
YIELD = 1
PROBE = 2
PROCEED = 3
COMMIT = 4
ACTIONS = (WAIT, YIELD, PROBE, PROCEED, COMMIT)
ACTION_NAMES = {
    WAIT: "wait",
    YIELD: "yield",
    PROBE: "probe",
    PROCEED: "proceed",
    COMMIT: "commit",
}

OPEN = 0
CLOSED = 1
TYPE_NAMES = {OPEN: "open", CLOSED: "closed"}

NONE = 0
GO = 1
STOP = 2
OBS_NAMES = {NONE: "none", GO: "go", STOP: "stop"}
N_TYPES = 2
N_OBS = 3

# Rows are partner-model types (open, closed). Columns are none, go, stop.
# Omissions are more likely when the partner model is closed.
BASE_LIKELIHOOD = np.array(
    [
        [0.10, 0.75, 0.15],
        [0.45, 0.10, 0.45],
    ],
    dtype=np.float64,
)
PROBE_LIKELIHOOD = np.array(
    [
        [0.02, 0.93, 0.05],
        [0.10, 0.05, 0.85],
    ],
    dtype=np.float64,
)

TIME_COST = 0.01
PROBE_COST = 0.02
PRIVATE_HIT = 0.99

# vector: obs one-hot, previous action one-hot, private bits, question bits, time, block, clarity
OBS_DIM = 3 + 5 + 2 + 3 + 1 + 1 + 1


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    family: str
    world_type: int
    partner_belief: int = OPEN
    cues_follow_partner: bool = False
    private_evidence: bool = False
    private_deterministic: bool = True
    private_from_step: int = 1
    query_target: str = "world"
    ambiguous: bool = False
    access_missing: bool = False
    reversal_step: int = -1
    reversal_world: int = OPEN
    reversal_partner: int = OPEN
    intervening_start: int = -1
    intervening_len: int = 0
    held_out: bool = False
    novel_reversal: bool = False
    max_steps: int = 8
    score_from_step: int = 0
    blurb: str = ""
    scripted_cues: Optional[Tuple[int, ...]] = None


@dataclass
class Observation:
    """What the agent may read. `label` is for the eval harness, not the policy."""

    discrete: int
    private_present: bool
    private_says_open: bool
    query_is_partner: bool
    requery: bool
    access_missing: bool
    mask_update: bool
    step_index: int
    max_steps: int
    intervening: bool
    clarity: float
    generated_by: int
    public_informativeness: float
    vector: np.ndarray
    label: int
    world_type: int
    partner_belief: int
    post_reversal: bool
    pre_reversal_world: int
    family: str
    scenario_name: str
    ambiguous: bool
    held_out: bool
    novel_reversal: bool
    score_from_step: int

    def with_requery(self) -> "Observation":
        return replace(self, requery=True, vector=encode_observation(self, requery=True))


def likelihood(action: int, state: int, obs: int) -> float:
    table = PROBE_LIKELIHOOD if action == PROBE else BASE_LIKELIHOOD
    return float(table[state, obs])


def log_likelihood_row(action: int, obs: int) -> np.ndarray:
    table = PROBE_LIKELIHOOD if action == PROBE else BASE_LIKELIHOOD
    return np.log(table[:, obs])


def immediate_reward(action: int, world_type: int) -> Tuple[float, bool]:
    reward = -TIME_COST
    terminal = False
    if action == PROBE:
        reward -= PROBE_COST
    elif action == YIELD:
        reward += 0.40 if world_type == CLOSED else -0.40
        terminal = True
    elif action == PROCEED:
        reward += 0.55 if world_type == OPEN else -0.45
        terminal = True
    elif action == COMMIT:
        reward += 1.0 if world_type == OPEN else -1.0
        terminal = True
    return reward, terminal


def perseverative_action(action: int, pre_reversal_world: int) -> bool:
    """An action that was appropriate for the partner model before the flip."""
    if pre_reversal_world == OPEN:
        return action in (PROCEED, COMMIT)
    return action in (WAIT, YIELD)


def encode_observation(obs: Observation, requery: Optional[bool] = None) -> np.ndarray:
    flag = obs.requery if requery is None else requery
    vector = np.zeros(OBS_DIM, dtype=np.float32)
    vector[obs.discrete] = 1.0
    vector[3 + obs.generated_by] = 1.0
    vector[8] = 1.0 if obs.private_present else 0.0
    vector[9] = 1.0 if obs.private_says_open else 0.0
    vector[10] = 1.0 if obs.query_is_partner else 0.0
    vector[11] = 1.0 if flag else 0.0
    vector[12] = 1.0 if obs.access_missing else 0.0
    span = max(obs.max_steps - 1, 1)
    vector[13] = obs.step_index / span
    vector[14] = 1.0 if obs.intervening else 0.0
    vector[15] = obs.clarity
    return vector


def _scenario(
    name: str,
    seed: int,
    family: str,
    world_type: int,
    blurb: str,
    **kwargs,
) -> Scenario:
    partner = kwargs.pop("partner_belief", world_type)
    reversal_world = kwargs.pop("reversal_world", world_type)
    reversal_partner = kwargs.pop("reversal_partner", partner)
    return Scenario(
        name=name,
        seed=seed,
        family=family,
        world_type=world_type,
        partner_belief=partner,
        reversal_world=reversal_world,
        reversal_partner=reversal_partner,
        blurb=blurb,
        **kwargs,
    )


def fixed_validation_scenarios() -> List[Scenario]:
    """Fixed suite. Do not edit during a search."""
    return [
        _scenario(
            "anchor_open", 13, "anchor", OPEN,
            "Stationary open partner. Two-state anchor for the exact belief MDP.",
            max_steps=6,
        ),
        _scenario(
            "anchor_closed", 29, "anchor", CLOSED,
            "Stationary closed partner. Two-state anchor for the exact belief MDP.",
            max_steps=6,
        ),
        _scenario(
            "handoff_open", 41, "handoff", OPEN,
            "Structured handoff. The task is known and the partner model is open.",
        ),
        _scenario(
            "handoff_closed", 53, "handoff", CLOSED,
            "Structured handoff. Commit is unsafe; yield is appropriate.",
        ),
        _scenario(
            "handoff_open_late", 67, "handoff", OPEN,
            "Structured handoff with a longer look before the score window.",
            score_from_step=2,
        ),
        _scenario(
            "handoff_closed_late", 79, "handoff", CLOSED,
            "Structured handoff. Early cues are easy to over-read.",
            score_from_step=2,
        ),
        _scenario(
            "reversal_open_to_closed", 97, "reversal", OPEN,
            "Veridical reversal from open to closed.",
            reversal_step=4,
            reversal_world=CLOSED,
            reversal_partner=CLOSED,
            max_steps=10,
        ),
        _scenario(
            "reversal_open_to_closed_late", 113, "reversal", OPEN,
            "Veridical reversal from open to closed, later in the episode.",
            reversal_step=6,
            reversal_world=CLOSED,
            reversal_partner=CLOSED,
            max_steps=10,
        ),
        _scenario(
            "novel_reversal_after_gap", 131, "reversal", CLOSED,
            "Unrelated intervening trials, then a novel closed-to-open reversal.",
            reversal_step=6,
            reversal_world=OPEN,
            reversal_partner=OPEN,
            intervening_start=3,
            intervening_len=3,
            novel_reversal=True,
            max_steps=10,
        ),
        _scenario(
            "novel_reversal_after_gap_b", 149, "reversal", CLOSED,
            "Second seed of the novel reversal after a gap.",
            reversal_step=6,
            reversal_world=OPEN,
            reversal_partner=OPEN,
            intervening_start=3,
            intervening_len=3,
            novel_reversal=True,
            max_steps=10,
        ),
        _scenario(
            "true_belief_control", 167, "false_belief", OPEN,
            "True-belief control. Public cues and private evidence agree.",
            partner_belief=OPEN,
            private_evidence=True,
            private_deterministic=True,
            private_from_step=1,
        ),
        _scenario(
            "false_belief_private", 181, "false_belief", CLOSED,
            "Public cues say go. A private channel reveals that the world is closed.",
            partner_belief=OPEN,
            cues_follow_partner=True,
            private_evidence=True,
            private_deterministic=True,
            private_from_step=1,
            score_from_step=1,
        ),
        _scenario(
            "false_belief_no_access", 199, "false_belief", CLOSED,
            "No perceptual access. The public cue follows a false partner belief.",
            partner_belief=OPEN,
            cues_follow_partner=True,
            access_missing=True,
            ambiguous=True,
        ),
        _scenario(
            "whose_belief", 211, "false_belief", OPEN,
            "The question is what the partner believes, not what is true of the world.",
            partner_belief=CLOSED,
            cues_follow_partner=True,
            private_evidence=True,
            private_deterministic=True,
            query_target="partner",
            private_from_step=0,
        ),
        _scenario(
            "held_out_deceptive", 227, "deceptive", CLOSED,
            "Held-out partner. Public cues say go, and commit is unsafe.",
            partner_belief=OPEN,
            cues_follow_partner=True,
            held_out=True,
        ),
    ]


_VALIDATION_SEEDS = {scenario.seed for scenario in fixed_validation_scenarios()}


def further_study_scenarios() -> List[Scenario]:
    """Reports beside the score. Seeds stay off the fixed validation list."""
    return [
        *_endurance_scenarios(),
        *_varied_tom_scenarios(),
        *_recovery_scenarios(),
    ]


def _endurance_scenarios() -> List[Scenario]:
    return [
        _scenario(
            "further_novel_reversal", 307, "further_endurance", CLOSED,
            "Intervening steps, then a novel reversal, read after the plasticity window has closed.",
            reversal_step=6,
            reversal_world=OPEN,
            reversal_partner=OPEN,
            intervening_start=3,
            intervening_len=3,
            novel_reversal=True,
            max_steps=10,
            score_from_step=6,
        ),
        _scenario(
            "further_anchor_open", 311, "further_anchor", OPEN,
            "Anchor for refinement against the exact filter.",
            max_steps=6,
        ),
        _scenario(
            "further_anchor_closed", 317, "further_anchor", CLOSED,
            "Closed anchor for refinement against the exact filter.",
            max_steps=6,
        ),
    ]


def _varied_tom_scenarios() -> List[Scenario]:
    return [
        _scenario(
            "varied_world_true", 251, "further_varied", OPEN,
            "True-belief control for the world question.",
            partner_belief=OPEN,
            max_steps=6,
        ),
        _scenario(
            "varied_world_false", 263, "further_varied", CLOSED,
            "False belief about the world. Public cues follow an open partner.",
            partner_belief=OPEN,
            cues_follow_partner=True,
            private_evidence=True,
            private_deterministic=True,
            private_from_step=1,
            max_steps=8,
        ),
        _scenario(
            "varied_partner_true", 277, "further_varied", OPEN,
            "True belief about the partner.",
            partner_belief=OPEN,
            query_target="partner",
            cues_follow_partner=True,
            max_steps=7,
            score_from_step=1,
        ),
        _scenario(
            "varied_partner_false", 281, "further_varied", OPEN,
            "False belief paired with the partner question. The partner model is closed.",
            partner_belief=CLOSED,
            cues_follow_partner=True,
            query_target="partner",
            private_evidence=True,
            private_deterministic=True,
            private_from_step=2,
            max_steps=8,
        ),
        _scenario(
            "varied_whose_belief", 293, "further_varied", OPEN,
            "Switch of whose belief is asked. The partner model is closed.",
            partner_belief=CLOSED,
            cues_follow_partner=True,
            query_target="partner",
            private_evidence=True,
            private_deterministic=True,
            private_from_step=0,
            max_steps=5,
            score_from_step=1,
        ),
        _scenario(
            "varied_next_action", 347, "further_varied", OPEN,
            "Same open partner, reshuffled horizon. Credit is the next action.",
            partner_belief=OPEN,
            max_steps=5,
            score_from_step=2,
        ),
    ]


def _recovery_scenarios() -> List[Scenario]:
    return [
        _scenario(
            "recovery_stop", 331, "further_recovery", OPEN,
            "Open evidence, then a scripted stop as the world closes.",
            reversal_step=2,
            reversal_world=CLOSED,
            reversal_partner=CLOSED,
            scripted_cues=(GO, GO, STOP),
            max_steps=4,
        ),
    ]


def sample_training_scenario(rng: random.Random) -> Scenario:
    """Training draws. Excludes validation seeds and the held-out deceptive partner."""
    seed = rng.randrange(1_000, 20_000)
    while seed in _VALIDATION_SEEDS:
        seed = rng.randrange(1_000, 20_000)
    kind = rng.randrange(4)
    world = rng.randrange(2)
    if kind == 0:
        return _scenario(
            f"train_handoff_{seed}", seed, "handoff", world,
            "Training handoff.",
            partner_belief=world,
        )
    if kind == 1:
        return _scenario(
            f"train_reversal_{seed}", seed, "reversal", OPEN,
            "Training reversal, open to closed only.",
            partner_belief=OPEN,
            reversal_step=rng.choice((3, 4, 5)),
            reversal_world=CLOSED,
            reversal_partner=CLOSED,
            max_steps=10,
        )
    if kind == 2:
        return _scenario(
            f"train_private_{seed}", seed, "false_belief", CLOSED,
            "Training false belief with private evidence.",
            partner_belief=OPEN,
            cues_follow_partner=True,
            private_evidence=True,
            private_deterministic=False,
            private_from_step=1,
            score_from_step=1,
        )
    return _scenario(
        f"train_control_{seed}", seed, "false_belief", world,
        "Training true-belief control.",
        partner_belief=world,
        private_evidence=True,
        private_deterministic=False,
        private_from_step=1,
    )


class PartnerEnv:
    def __init__(self) -> None:
        self.scenario: Optional[Scenario] = None
        self.rng = random.Random(0)
        self.step_id = 0
        self.world = OPEN
        self.partner = OPEN
        self.prev_action = WAIT
        self._initial_world = OPEN

    def reset(self, scenario: Scenario) -> Tuple[Observation, Dict]:
        self.scenario = scenario
        self.rng = random.Random(scenario.seed)
        self.step_id = 0
        self.world = scenario.world_type
        self.partner = scenario.partner_belief
        self.prev_action = WAIT
        self._initial_world = scenario.world_type
        obs = self._emit(generated_by=WAIT)
        return obs, self._info(obs)

    def step(self, action: int) -> Tuple[Observation, float, bool, Dict]:
        if self.scenario is None:
            raise RuntimeError("reset the environment before stepping")
        decision_world = self.world
        reward, terminal = immediate_reward(action, decision_world)
        self.prev_action = action
        self.step_id += 1
        if self.scenario.reversal_step == self.step_id:
            self.world = self.scenario.reversal_world
            self.partner = self.scenario.reversal_partner
        done = terminal or self.step_id >= self.scenario.max_steps
        obs = self._emit(generated_by=action)
        info = self._info(obs)
        info["reward_world"] = decision_world
        info["unsafe_world_commit"] = bool(action == COMMIT and decision_world == CLOSED)
        return obs, reward, done, info

    def _intervening(self) -> bool:
        scenario = self.scenario
        assert scenario is not None
        if scenario.intervening_start < 0:
            return False
        return scenario.intervening_start <= self.step_id < scenario.intervening_start + scenario.intervening_len

    def _emit(self, generated_by: int) -> Observation:
        scenario = self.scenario
        assert scenario is not None
        intervening = self._intervening()
        if intervening:
            discrete = NONE
            mask_update = True
        elif scenario.scripted_cues is not None and self.step_id < len(scenario.scripted_cues):
            discrete = int(scenario.scripted_cues[self.step_id])
            mask_update = False
        else:
            source = self.partner if scenario.cues_follow_partner else self.world
            table = PROBE_LIKELIHOOD if generated_by == PROBE else BASE_LIKELIHOOD
            discrete = _sample_categorical(self.rng, table[source])
            mask_update = False
        private_present = False
        private_says_open = False
        if (
            scenario.private_evidence
            and not intervening
            and self.step_id >= scenario.private_from_step
        ):
            private_present = True
            if scenario.private_deterministic:
                private_says_open = self.world == OPEN
            else:
                hit = PRIVATE_HIT if self.world == OPEN else 1.0 - PRIVATE_HIT
                private_says_open = self.rng.random() < hit
        post_reversal = scenario.reversal_step >= 0 and self.step_id >= scenario.reversal_step
        obs = Observation(
            discrete=discrete,
            private_present=private_present,
            private_says_open=private_says_open,
            query_is_partner=scenario.query_target == "partner",
            requery=False,
            access_missing=scenario.access_missing and not intervening,
            mask_update=mask_update,
            step_index=self.step_id,
            max_steps=scenario.max_steps,
            intervening=intervening,
            clarity=1.0 if generated_by == PROBE else 0.0,
            generated_by=generated_by,
            public_informativeness=0.0 if (scenario.access_missing and scenario.query_target == "world") else 1.0,
            vector=np.zeros(OBS_DIM, dtype=np.float32),
            label=self.partner if scenario.query_target == "partner" else self.world,
            world_type=self.world,
            partner_belief=self.partner,
            post_reversal=post_reversal,
            pre_reversal_world=self._initial_world,
            family=scenario.family,
            scenario_name=scenario.name,
            ambiguous=scenario.ambiguous,
            held_out=scenario.held_out,
            novel_reversal=scenario.novel_reversal,
            score_from_step=scenario.score_from_step,
        )
        obs.vector = encode_observation(obs)
        return obs

    def _info(self, obs: Observation) -> Dict:
        return {
            "label": obs.label,
            "world_type": obs.world_type,
            "discrete": obs.discrete,
            "scenario": obs.scenario_name,
            "family": obs.family,
            "ambiguous": obs.ambiguous,
            "held_out": obs.held_out,
            "novel_reversal": obs.novel_reversal,
            "post_reversal": obs.post_reversal,
            "mask_update": obs.mask_update,
            "score_from_step": obs.score_from_step,
        }


def _sample_categorical(rng: random.Random, probs: np.ndarray) -> int:
    draw = rng.random()
    cumulative = 0.0
    for index, prob in enumerate(probs.tolist()):
        cumulative += prob
        if draw <= cumulative:
            return index
    return len(probs) - 1
