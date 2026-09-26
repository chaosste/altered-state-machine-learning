"""Exact filter and small belief-MDP solution for the anchor task.

Frozen during a search. The anchor has two hidden partner models, identity
transitions, and the likelihood tables in env.py. Chad?s et al. (2021): solve
the small problem before treating a larger model as the same update rule.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from env import (
    ACTIONS,
    BASE_LIKELIHOOD,
    CLOSED,
    N_OBS,
    N_TYPES,
    OPEN,
    PROBE_LIKELIHOOD,
    Observation,
    immediate_reward,
    likelihood,
    log_likelihood_row,
)


# Favours open. A sluggish learner has to be moved off this prior by evidence.
DEFAULT_PRIOR = np.array([1.0, -1.0], dtype=np.float64)
_ANCHOR_CACHE: Dict[Tuple[int, int, float], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def initial_logits(precision: float) -> np.ndarray:
    return precision * DEFAULT_PRIOR.copy()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def entropy(belief: np.ndarray) -> float:
    clipped = np.clip(belief, 1e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def evidence_update(
    logits: np.ndarray,
    obs: Observation,
    prev_action: int,
    pe_gain: float,
) -> np.ndarray:
    """Logit Bayes step. `pe_gain` 1 is exact. Requery and masked steps add nothing.

    Missing perceptual access blocks public cues when the question is about the
    world. Private evidence is about the world, so it is ignored when the
    question is about the partner's belief.
    """
    if obs.mask_update or obs.requery or pe_gain == 0.0:
        return logits.copy()
    updated = logits.astype(np.float64).copy()
    use_public = not (obs.access_missing and not obs.query_is_partner)
    if use_public:
        updated = updated + pe_gain * log_likelihood_row(prev_action, obs.discrete)
    if obs.private_present and not obs.query_is_partner:
        p_open = 0.99 if obs.private_says_open else 0.01
        private = np.log(np.array([p_open, 1.0 - p_open], dtype=np.float64))
        updated = updated + pe_gain * private
    return updated


def bayes_belief(belief: np.ndarray, action: int, obs: int) -> np.ndarray:
    """Exact one-step filter used by value iteration. Veridical public cues only."""
    lik = np.array([likelihood(action, state, obs) for state in range(N_TYPES)], dtype=np.float64)
    posterior = belief * lik
    total = float(posterior.sum())
    if total <= 0.0:
        return belief.copy()
    return posterior / total


def solve_anchor(
    max_steps: int = 6,
    n_grid: int = 51,
    gamma: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite-horizon value iteration on a grid over P(open).

    Returns grid, values[steps_left, grid], greedy[steps_left, grid].
    """
    key = (max_steps, n_grid, gamma)
    cached = _ANCHOR_CACHE.get(key)
    if cached is not None:
        return cached
    grid = np.linspace(0.0, 1.0, n_grid)
    values = np.zeros((max_steps + 1, n_grid), dtype=np.float64)
    greedy = np.zeros((max_steps + 1, n_grid), dtype=np.int64)
    for steps_left in range(1, max_steps + 1):
        for index, p_open in enumerate(grid):
            belief = np.array([p_open, 1.0 - p_open], dtype=np.float64)
            best_action = ACTIONS[0]
            best_q = -1e18
            for action in ACTIONS:
                reward_open, terminal_open = immediate_reward(action, OPEN)
                reward_closed, terminal_closed = immediate_reward(action, CLOSED)
                if terminal_open != terminal_closed:
                    raise RuntimeError("terminal flag must not depend on the hidden partner model")
                reward = belief[OPEN] * reward_open + belief[CLOSED] * reward_closed
                continuation = 0.0
                if not terminal_open and steps_left > 1:
                    for obs in range(N_OBS):
                        p_obs = float(
                            belief[OPEN] * likelihood(action, OPEN, obs)
                            + belief[CLOSED] * likelihood(action, CLOSED, obs)
                        )
                        if p_obs <= 0.0:
                            continue
                        posterior = bayes_belief(belief, action, obs)
                        continuation += p_obs * float(np.interp(posterior[OPEN], grid, values[steps_left - 1]))
                q_value = reward + gamma * continuation
                if q_value > best_q:
                    best_q = q_value
                    best_action = action
            values[steps_left, index] = best_q
            greedy[steps_left, index] = best_action
    solved = (grid, values, greedy)
    _ANCHOR_CACHE[key] = solved
    return solved


def anchor_action(p_open: float, steps_left: int, max_steps: int = 6) -> int:
    grid, _values, greedy = solve_anchor(max_steps=max_steps)
    steps = int(np.clip(steps_left, 1, max_steps))
    index = int(np.argmin(np.abs(grid - p_open)))
    return int(greedy[steps, index])


def anchor_value(p_open: float, steps_left: int, max_steps: int = 6) -> float:
    grid, values, _greedy = solve_anchor(max_steps=max_steps)
    steps = int(np.clip(steps_left, 1, max_steps))
    return float(np.interp(p_open, grid, values[steps]))


def rows_are_distributions() -> bool:
    return bool(
        np.allclose(BASE_LIKELIHOOD.sum(axis=1), 1.0)
        and         np.allclose(PROBE_LIKELIHOOD.sum(axis=1), 1.0)
    )
