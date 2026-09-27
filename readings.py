"""Later readings on a finished belief trace.

Cue stance, cue polarity, and richness are reports. They do not enter the
reward, the belief filter, or BeliefUpdateScore. See program.md.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence

CONFIDENT_GAP = 0.05
RICHNESS_LOW = 0.15
DIFFUSE_ENTROPY = 0.9 * math.log(2.0)

STANCE_BY_CUE = {
    "go": "approach",
    "stop": "avoid",
    "none": "withhold",
}
POLARITY = {"approach": 1, "avoid": -1, "withhold": 0, "unavailable": 0}
BAND_LABELS = {
    "over_precise": "over-precise",
    "in_band": "in band",
    "diffuse": "diffuse",
    "unavailable": "unavailable",
}


def cue_stance(step: Mapping[str, object]) -> str:
    """Name the public cue. A masked update has no usable stance."""
    if bool(step.get("masked")):
        return "unavailable"
    return STANCE_BY_CUE.get(str(step.get("cue", "")), "unavailable")


def entropy_band(value: Optional[float]) -> str:
    if value is None:
        return "unavailable"
    entropy = float(value)
    if entropy < RICHNESS_LOW:
        return "over_precise"
    if entropy > DIFFUSE_ENTROPY:
        return "diffuse"
    return "in_band"


def binary_entropy(belief: Sequence[float]) -> float:
    total = 0.0
    for probability in belief:
        value = float(probability)
        if value > 0.0:
            total -= value * math.log(value)
    return total


def partner_call(belief: Sequence[float]) -> str:
    if len(belief) < 2:
        return "undetermined"
    open_p = float(belief[0])
    closed_p = float(belief[1])
    if abs(open_p - closed_p) < CONFIDENT_GAP or binary_entropy(belief) > DIFFUSE_ENTROPY:
        return "undetermined"
    if open_p > closed_p:
        return "open"
    return "closed"


def _pair(step: Mapping[str, object], key: str) -> Optional[List[float]]:
    raw = step.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    return [float(raw[0]), float(raw[1])]


def richness_step(step: Mapping[str, object]) -> Dict[str, object]:
    entropy_raw = step.get("epistemic_entropy")
    entropy = None if entropy_raw is None else float(entropy_raw)
    before = _pair(step, "belief_before")
    after = _pair(step, "belief_after")
    if before is None or after is None:
        movement = None
        call_before = "undetermined"
        call_after = "undetermined"
    else:
        movement = 0.5 * (abs(after[0] - before[0]) + abs(after[1] - before[1]))
        call_before = partner_call(before)
        call_after = partner_call(after)
    band = entropy_band(entropy)
    return {
        "epistemic_entropy": entropy,
        "band": band,
        "band_label": BAND_LABELS[band],
        "belief_movement": movement,
        "call_before": call_before,
        "call_after": call_after,
        "call_changed": call_before != call_after,
    }


def read_trace(trace: Mapping[str, object]) -> Dict[str, object]:
    steps = list(trace.get("steps") or [])
    stances = [cue_stance(step) for step in steps]
    polarities = [POLARITY[stance] for stance in stances]
    richness = [richness_step(step) for step in steps]
    in_band = sum(1 for row in richness if row["band"] == "in_band")
    call_changes = sum(1 for row in richness if row["call_changed"])
    return {
        "name": trace.get("name"),
        "cue_stance": stances,
        "cue_polarity_sum": int(sum(polarities)),
        "cue_polarity_used_as_reward": False,
        "steps_in_entropy_band": in_band,
        "n_steps": len(steps),
        "partner_call_changes": call_changes,
        "richness": richness,
        "changes_belief_update_score": False,
        "selection_metric": None,
    }


def read_traces(traces: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    scenarios = [read_trace(trace) for trace in traces]
    return {
        "instrument": "later_readings",
        "changes_belief_update_score": False,
        "selection_metric": None,
        "scenarios": scenarios,
    }
