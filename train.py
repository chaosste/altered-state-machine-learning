"""Baseline trainer. This is the only file a search should edit.

The belief is an explicit filter. Phenotype constants below are the placebo-like
baseline: an open-leaning prior, a low prediction-error gain, equal reward and
punishment learning rates, and stickiness on the previous action. They do not
add a bonus to proceed or commit.

Later arms change one block:
  2. Raise REWARD_LEARNING_RATE only.
  3. Raise both learning rates (reward more), lower STICKINESS, set
     PHASE_DEPENDENT_SENSITIVITY and the two betas.
  4. Lower PRIOR_PRECISION, raise BELIEF_PE_GAIN.
  5. Raise the rates and set PLASTICITY_UNTIL_EPISODE so eval, which uses the
     post-window constants, sees whether the change lasted.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from env import OBS_DIM, WAIT, Observation, PartnerEnv, sample_training_scenario
from eval import emit_eval_metrics, evaluate_policy
from oracle import evidence_update, initial_logits, softmax


VARIANT = "baseline"
NEUTRAL_LEARNING_RATE = 0.15
REWARD_LEARNING_RATE = 0.15
PUNISHMENT_LEARNING_RATE = 0.15
STICKINESS = 0.75
REINFORCEMENT_SENSITIVITY = 1.0
PRIOR_PRECISION = 1.5
BELIEF_PE_GAIN = 0.35
PHASE_DEPENDENT_SENSITIVITY = False
ACQUISITION_BETA = 1.0
REVERSAL_BETA = 1.0
SURPRISE_THRESHOLD = 0.35
PLASTICITY_UNTIL_EPISODE = None
CONSOLIDATED_PE_GAIN = 0.35
CONSOLIDATED_REWARD_LEARNING_RATE = 0.15
CONSOLIDATED_PUNISHMENT_LEARNING_RATE = 0.15
CONSOLIDATED_STICKINESS = 0.75
GAMMA = 0.95
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5


def phenotype_constants(episode: int | None = None) -> Dict[str, float | bool | None]:
    """Constants that actually drive the update. Eval passes a huge episode index."""
    use_window = (
        PLASTICITY_UNTIL_EPISODE is not None
        and episode is not None
        and episode >= PLASTICITY_UNTIL_EPISODE
    )
    if use_window:
        pe_gain = CONSOLIDATED_PE_GAIN
        reward_lr = CONSOLIDATED_REWARD_LEARNING_RATE
        punishment_lr = CONSOLIDATED_PUNISHMENT_LEARNING_RATE
        stickiness = CONSOLIDATED_STICKINESS
    else:
        pe_gain = BELIEF_PE_GAIN
        reward_lr = REWARD_LEARNING_RATE
        punishment_lr = PUNISHMENT_LEARNING_RATE
        stickiness = STICKINESS
    return {
        "pe_gain": pe_gain,
        "reward_lr": reward_lr,
        "punishment_lr": punishment_lr,
        "stickiness": stickiness,
        "prior_precision": PRIOR_PRECISION,
        "reinforcement_sensitivity": REINFORCEMENT_SENSITIVITY,
        "phase_dependent_sensitivity": PHASE_DEPENDENT_SENSITIVITY,
        "acquisition_beta": ACQUISITION_BETA,
        "reversal_beta": REVERSAL_BETA,
        "plasticity_until_episode": PLASTICITY_UNTIL_EPISODE,
    }


@dataclass
class StepDecision:
    action: int
    belief_before: np.ndarray
    belief_after: np.ndarray
    logp: torch.Tensor | None = None
    value: torch.Tensor | None = None
    entropy: torch.Tensor | None = None


class BaselinePolicy(nn.Module):
    def __init__(self, hidden_dim: int = 64, device: str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(OBS_DIM + 2, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, 5)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.to(self.device)
        self.h = torch.zeros(1, hidden_dim, device=self.device)
        self.logits = initial_logits(PRIOR_PRECISION)
        self.prev_action = WAIT
        self.pe_gain = BELIEF_PE_GAIN
        self.stickiness = STICKINESS
        self.beta = REINFORCEMENT_SENSITIVITY

    def reset_episode(self) -> None:
        self.h = torch.zeros(1, self.hidden_dim, device=self.device)
        constants = phenotype_constants(10**9)
        self.logits = initial_logits(float(constants["prior_precision"]))
        self.prev_action = WAIT
        self.pe_gain = float(constants["pe_gain"])
        self.stickiness = float(constants["stickiness"])
        self.beta = float(constants["reinforcement_sensitivity"])

    def begin_training_episode(self, episode: int) -> None:
        self.h = torch.zeros(1, self.hidden_dim, device=self.device)
        constants = phenotype_constants(episode)
        self.logits = initial_logits(float(constants["prior_precision"]))
        self.prev_action = WAIT
        self.pe_gain = float(constants["pe_gain"])
        self.stickiness = float(constants["stickiness"])
        self._reward_lr = float(constants["reward_lr"])
        self._punishment_lr = float(constants["punishment_lr"])

    def act(self, obs: Observation, deterministic: bool = True) -> StepDecision:
        before = softmax(self.logits)
        self.logits = evidence_update(self.logits, obs, self.prev_action, self.pe_gain)
        after = softmax(self.logits)
        self.beta = self._sensitivity(before, after)
        self._h_in = self.h
        self._stick_from = self.prev_action
        action, logp, value, entropy = self._forward(
            obs, after, deterministic=deterministic, write_state=True, h_in=self._h_in, stick_from=self._stick_from
        )
        self.prev_action = action
        return StepDecision(action, before, after, logp, value, entropy)

    def requery(self, obs: Observation) -> StepDecision:
        """Replay the same pre-action state with the confirmation flag. Does not learn."""
        after = softmax(self.logits)
        confirm = obs.with_requery()
        action, _logp, _value, _entropy = self._forward(
            confirm,
            after,
            deterministic=True,
            write_state=False,
            h_in=self._h_in,
            stick_from=self._stick_from,
        )
        return StepDecision(action, after.copy(), after.copy())

    def _sensitivity(self, before: np.ndarray, after: np.ndarray) -> float:
        if not PHASE_DEPENDENT_SENSITIVITY:
            return float(REINFORCEMENT_SENSITIVITY)
        surprise = 0.5 * float(np.abs(after - before).sum())
        if surprise >= SURPRISE_THRESHOLD:
            return float(REVERSAL_BETA)
        return float(ACQUISITION_BETA)

    def _forward(
        self,
        obs: Observation,
        belief: np.ndarray,
        deterministic: bool,
        write_state: bool,
        h_in: torch.Tensor,
        stick_from: int,
    ):
        vector = torch.tensor(obs.vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        belief_t = torch.tensor(np.asarray(belief, dtype=np.float32), device=self.device).unsqueeze(0)
        hidden = self.gru(torch.cat([vector, belief_t], dim=-1), h_in)
        if write_state:
            self.h = hidden
        action_logits = self.action_head(hidden).squeeze(0)
        sticky = torch.zeros_like(action_logits)
        sticky[stick_from] = self.stickiness
        action_logits = (action_logits + sticky) * self.beta
        value = self.value_head(hidden).squeeze()
        dist = Categorical(logits=action_logits)
        if deterministic:
            action = int(torch.argmax(action_logits).item())
        else:
            action = int(dist.sample().item())
        logp = dist.log_prob(torch.tensor(action, device=self.device))
        return action, logp, value, dist.entropy()


def _episode_loss(policy: BaselinePolicy, decisions: List[StepDecision], rewards: List[float]) -> torch.Tensor:
    returns: List[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + GAMMA * running
        returns.append(running)
    returns.reverse()
    policy_terms = []
    value_terms = []
    entropy_terms = []
    for decision, reward, target in zip(decisions, rewards, returns):
        assert decision.logp is not None and decision.value is not None and decision.entropy is not None
        advantage = target - float(decision.value.detach())
        weight = policy._reward_lr if reward >= 0.0 else policy._punishment_lr
        weight = weight / NEUTRAL_LEARNING_RATE
        policy_terms.append(-decision.logp * advantage)
        value_terms.append(weight * (decision.value - target) ** 2)
        entropy_terms.append(decision.entropy)
    return (
        torch.stack(policy_terms).mean()
        + VALUE_COEF * torch.stack(value_terms).mean()
        - ENTROPY_COEF * torch.stack(entropy_terms).mean()
    )


def train(episodes: int, seed: int, output_dir: Path, device: str, hidden_dim: int, lr: float) -> Dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    env = PartnerEnv()
    policy = BaselinePolicy(hidden_dim=hidden_dim, device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / "learning_curve.csv"
    with curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "return", "belief_match"])
        writer.writeheader()
        for episode in range(episodes):
            scenario = sample_training_scenario(rng)
            policy.begin_training_episode(episode)
            obs, info = env.reset(scenario)
            decisions: List[StepDecision] = []
            rewards: List[float] = []
            matches: List[float] = []
            done = False
            while not done:
                decision = policy.act(obs, deterministic=False)
                _next, reward, done, step_info = env.step(decision.action)
                decisions.append(decision)
                rewards.append(reward)
                matches.append(1.0 if int(np.argmax(decision.belief_after)) == int(info["label"]) else 0.0)
                if not done:
                    obs, info = _next, step_info
            optimizer.zero_grad()
            loss = _episode_loss(policy, decisions, rewards)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            policy.h = policy.h.detach()
            writer.writerow(
                {
                    "episode": episode,
                    "return": f"{sum(rewards):.6f}",
                    "belief_match": f"{float(np.mean(matches)):.6f}",
                }
            )

    policy.reset_episode()
    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        metrics = evaluate_policy(policy)
    if was_training:
        policy.train()
    traces = metrics.pop("traces")
    payload = {
        "variant": VARIANT,
        "seed": seed,
        "episodes": episodes,
        "constants": phenotype_constants(10**9),
        **metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output_dir / "trace.json").write_text(json.dumps(traces, indent=2) + "\n")
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "constants": phenotype_constants(None),
            "variant": VARIANT,
            "hidden_dim": hidden_dim,
        },
        output_dir / "model.pt",
    )
    print(emit_eval_metrics(payload))
    print(f"saved_checkpoint={output_dir / 'model.pt'}")
    print(f"learning_curve_csv={curve_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline partner-belief policy.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()
    train(
        episodes=args.episodes,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        device=args.device,
        hidden_dim=args.hidden,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
