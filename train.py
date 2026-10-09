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
  6. Combine arm 4 belief settings with arm 3 stickiness and phase sensitivity.
  7. Add the existing commit guard to arm 6.
  8. Use arm 4 prior precision with a lower belief prediction-error gain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from contracts import DEFAULT_CONTRACT, EVIDENCE_RESPONSE_V3, HIGH_RISK_BELIEF_UPDATE_V1, resolve_contract
from env import COMMIT, OBS_DIM, WAIT, Observation, PartnerEnv, Scenario, sample_training_scenario
from eval import emit_eval_metrics, evaluate_policy, gate_would_block
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

# Pre-registered variants. `rebus` and `arm4_rebus` are aliases for arm 4.
VARIANT_ALIASES = {
    "baseline": "baseline",
    "arm2": "arm2",
    "arm3": "arm3",
    "arm4": "arm4",
    "arm4_rebus": "arm4",
    "rebus": "arm4",
    "arm5": "arm5",
    "arm6": "arm6",
    "arm7": "arm7",
    "arm8": "arm8",
}
VARIANT_LABELS = {
    "baseline": "baseline",
    "arm2": "arm2_reward_lr",
    "arm3": "arm3_kanen_phase",
    "arm4": "arm4_rebus",
    "arm5": "arm5_plasticity_window",
    "arm6": "arm6_evidence_flexible_policy",
    "arm7": "arm7_evidence_flexible_policy_commit_guard",
    "arm8": "arm8_rebus_pe_gain_08",
}
VARIANT_OVERRIDES: Dict[str, Dict[str, object]] = {
    "baseline": {},
    "arm2": {"reward_lr": 0.30},
    "arm3": {
        "reward_lr": 0.30,
        "punishment_lr": 0.22,
        "stickiness": 0.25,
        "phase_dependent_sensitivity": True,
        "acquisition_beta": 0.75,
        "reversal_beta": 1.25,
    },
    "arm4": {"prior_precision": 0.5, "pe_gain": 1.0},
    "arm5": {"prior_precision": 0.5, "pe_gain": 1.0, "plasticity_until_episode": 25},
    "arm6": {
        "prior_precision": 0.5,
        "pe_gain": 1.0,
        "stickiness": 0.25,
        "phase_dependent_sensitivity": True,
        "acquisition_beta": 0.75,
        "reversal_beta": 1.25,
    },
    "arm7": {
        "prior_precision": 0.5,
        "pe_gain": 1.0,
        "stickiness": 0.25,
        "phase_dependent_sensitivity": True,
        "acquisition_beta": 0.75,
        "reversal_beta": 1.25,
    },
    "arm8": {"prior_precision": 0.5, "pe_gain": 0.8},
}
_ACTIVE_VARIANT = "baseline"


def resolve_variant(name: str) -> str:
    key = VARIANT_ALIASES.get(name.strip().lower())
    if key is None:
        known = ", ".join(sorted(VARIANT_ALIASES))
        raise ValueError(f"Unknown variant {name}. Choose from {known}.")
    return key


def apply_variant(name: str) -> str:
    global _ACTIVE_VARIANT
    _ACTIVE_VARIANT = resolve_variant(name)
    return _ACTIVE_VARIANT


def active_variant() -> str:
    return VARIANT_LABELS[_ACTIVE_VARIANT]


def active_variant_key() -> str:
    return _ACTIVE_VARIANT


def training_schedule_for_seed(episodes: int, seed: int) -> List[Scenario]:
    """Pre-generate the existing training template schedule for matched runs."""
    rng = random.Random(seed)
    return [sample_training_scenario(rng) for _ in range(episodes)]


def scenario_schedule_hash(schedule: Sequence[Scenario]) -> str:
    payload = [asdict(scenario) for scenario in schedule]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scenario_schedule_payload(contract_id: str, seed: int, schedule: Sequence[Scenario]) -> Dict[str, object]:
    return {
        "contract_id": contract_id,
        "seed": seed,
        "episodes": len(schedule),
        "schedule_sha256": scenario_schedule_hash(schedule),
        "generator": "env.sample_training_scenario(random.Random(seed))",
        "scenarios": [asdict(scenario) for scenario in schedule],
    }


def load_training_schedule(path: Path) -> List[Scenario]:
    payload = json.loads(Path(path).read_text())
    rows = payload.get("scenarios") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a scenario schedule.")
    schedule = []
    for row in rows:
        values = dict(row)
        if values.get("scripted_cues") is not None:
            values["scripted_cues"] = tuple(values["scripted_cues"])
        schedule.append(Scenario(**values))
    expected = payload.get("schedule_sha256") if isinstance(payload, dict) else None
    if expected and expected != scenario_schedule_hash(schedule):
        raise ValueError(f"Training schedule hash mismatch in {path}.")
    return schedule


def phenotype_constants(episode: int | None = None) -> Dict[str, float | bool | None]:
    """Constants that actually drive the update. Eval passes a huge episode index."""
    spec: Dict[str, float | bool | None] = {
        "pe_gain": BELIEF_PE_GAIN,
        "reward_lr": REWARD_LEARNING_RATE,
        "punishment_lr": PUNISHMENT_LEARNING_RATE,
        "stickiness": STICKINESS,
        "prior_precision": PRIOR_PRECISION,
        "reinforcement_sensitivity": REINFORCEMENT_SENSITIVITY,
        "phase_dependent_sensitivity": PHASE_DEPENDENT_SENSITIVITY,
        "acquisition_beta": ACQUISITION_BETA,
        "reversal_beta": REVERSAL_BETA,
        "plasticity_until_episode": PLASTICITY_UNTIL_EPISODE,
    }
    for key, value in VARIANT_OVERRIDES[_ACTIVE_VARIANT].items():
        spec[key] = value  # type: ignore[assignment]
    until = spec["plasticity_until_episode"]
    window_closed = until is not None and episode is not None and episode >= int(until)
    if window_closed:
        spec["pe_gain"] = CONSOLIDATED_PE_GAIN
        spec["reward_lr"] = CONSOLIDATED_REWARD_LEARNING_RATE
        spec["punishment_lr"] = CONSOLIDATED_PUNISHMENT_LEARNING_RATE
        spec["stickiness"] = CONSOLIDATED_STICKINESS
        spec["prior_precision"] = PRIOR_PRECISION
    return spec


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
        self._spec_episode: int | None = 10**9
        self.logits = initial_logits(PRIOR_PRECISION)
        self.prev_action = WAIT
        self.pe_gain = BELIEF_PE_GAIN
        self.stickiness = STICKINESS
        self.beta = REINFORCEMENT_SENSITIVITY

    def reset_episode(self) -> None:
        self.h = torch.zeros(1, self.hidden_dim, device=self.device)
        self._spec_episode = 10**9
        constants = phenotype_constants(10**9)
        self.logits = initial_logits(float(constants["prior_precision"]))
        self.prev_action = WAIT
        self.pe_gain = float(constants["pe_gain"])
        self.stickiness = float(constants["stickiness"])
        self.beta = float(constants["reinforcement_sensitivity"])

    def begin_training_episode(self, episode: int) -> None:
        self.h = torch.zeros(1, self.hidden_dim, device=self.device)
        self._spec_episode = episode
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
        spec = phenotype_constants(self._spec_episode)
        if not spec["phase_dependent_sensitivity"]:
            return float(spec["reinforcement_sensitivity"])
        surprise = 0.5 * float(np.abs(after - before).sum())
        if surprise >= SURPRISE_THRESHOLD:
            return float(spec["reversal_beta"])
        return float(spec["acquisition_beta"])

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
        if _ACTIVE_VARIANT == "arm7" and gate_would_block(COMMIT, belief):
            action_logits = action_logits.clone()
            action_logits[COMMIT] = torch.finfo(action_logits.dtype).min
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


def train(
    episodes: int,
    seed: int,
    output_dir: Path,
    device: str,
    hidden_dim: int,
    lr: float,
    contract_id: str = DEFAULT_CONTRACT,
    training_schedule: Optional[Sequence[Scenario]] = None,
    quiet: bool = False,
) -> Dict[str, object]:
    contract_id = resolve_contract(contract_id)
    if training_schedule is not None and len(training_schedule) != episodes:
        raise ValueError("The supplied training schedule length must equal episodes.")
    if contract_id == HIGH_RISK_BELIEF_UPDATE_V1 and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing high-risk v1 artifacts in {output_dir}.")
    schedule = list(training_schedule) if training_schedule is not None else None
    if contract_id == HIGH_RISK_BELIEF_UPDATE_V1 and schedule is None:
        schedule = training_schedule_for_seed(episodes, seed)
    schedule_hash = scenario_schedule_hash(schedule) if schedule is not None else None
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
        writer = csv.DictWriter(
            handle,
            fieldnames=["episode", "return", "belief_match"],
            lineterminator="\n",
        )
        writer.writeheader()
        for episode in range(episodes):
            scenario = schedule[episode] if schedule is not None else sample_training_scenario(rng)
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
        if contract_id == HIGH_RISK_BELIEF_UPDATE_V1:
            from high_risk_suite import evaluate_high_risk_policy

            evaluated = evaluate_high_risk_policy(
                policy,
                variant=active_variant_key(),
                constants=phenotype_constants(10**9),
                baseline_prior_precision=PRIOR_PRECISION,
            )
            replay_artifact = evaluated.pop("replay_artifact")
            trace_artifact = evaluated.pop("trace_artifact")
            metrics = evaluated
        else:
            metrics = evaluate_policy(policy)
            traces = metrics.pop("traces")
            trace_artifact = {
                "contract_id": EVIDENCE_RESPONSE_V3,
                "traces": traces,
            }
            replay_artifact = None
    if was_training:
        policy.train()
    payload = {
        "contract_id": contract_id,
        "variant": active_variant(),
        "seed": seed,
        "episodes": episodes,
        "constants": phenotype_constants(10**9),
        "training_device": device,
        "hidden_dim": hidden_dim,
        "optimizer_lr": lr,
        **metrics,
    }
    if schedule_hash is not None:
        payload["training_schedule_sha256"] = schedule_hash
        payload["training_schedule_source"] = "pre-generated and reused" if training_schedule is not None else "generated once before training"
    if contract_id == HIGH_RISK_BELIEF_UPDATE_V1:
        payload["configuration_provenance"] = {
            "variant_key": active_variant_key(),
            "constants": phenotype_constants(10**9),
            "training_device": device,
            "hidden_dim": hidden_dim,
            "optimizer_lr": lr,
            "episodes": episodes,
            "seed": seed,
            "training_schedule_sha256": schedule_hash,
        }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output_dir / "trace.json").write_text(json.dumps(trace_artifact, indent=2) + "\n")
    if replay_artifact is not None:
        (output_dir / "belief_replay.json").write_text(json.dumps(replay_artifact, indent=2) + "\n")
        if training_schedule is None:
            (output_dir / "training_schedule.json").write_text(
                json.dumps(scenario_schedule_payload(contract_id, seed, schedule or []), indent=2) + "\n"
            )
    torch.save(
        {
            "contract_id": contract_id,
            "state_dict": policy.state_dict(),
            "constants": phenotype_constants(None),
            "variant": active_variant(),
            "hidden_dim": hidden_dim,
        },
        output_dir / "model.pt",
    )
    if not quiet:
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
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--scenario-schedule", default="", help="JSON schedule produced for a matched comparison")
    args = parser.parse_args()
    try:
        apply_variant(args.variant)
    except ValueError as exc:
        parser.error(str(exc))
    train(
        episodes=args.episodes,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        device=args.device,
        hidden_dim=args.hidden,
        lr=args.lr,
        contract_id=args.contract,
        training_schedule=load_training_schedule(Path(args.scenario_schedule)) if args.scenario_schedule else None,
    )


if __name__ == "__main__":
    main()
