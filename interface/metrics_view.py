"""Shared score list for the menu and the belief-trace page."""

from __future__ import annotations

from typing import Dict, List


SCORE_FIELDS = (
    "BeliefUpdateScore",
    "revision_accuracy",
    "revision_speed",
    "perseveration",
    "omission_sensitivity",
    "calibration_ece",
    "commitment_consistency",
    "unsafe_commit_rate",
    "ignored_evidence_rate",
)

TAGLINE = "Calibrated belief revision. BeliefUpdateScore is the only keep/discard metric."


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return str(value)


def format_metrics_list(payload: Dict[str, object]) -> str:
    """One field per line. BeliefUpdateScore comes first."""
    lines: List[str] = []
    seen = set()

    def add(key: str, value: object, indent: int = 0) -> None:
        pad = "  " * indent
        if isinstance(value, dict):
            lines.append(f"{pad}- {key}")
            for sub in sorted(value):
                add(str(sub), value[sub], indent + 1)
            return
        lines.append(f"{pad}- {key}: {format_value(value)}")

    if "BeliefUpdateScore" in payload:
        add("BeliefUpdateScore", payload["BeliefUpdateScore"])
        seen.add("BeliefUpdateScore")
    for key in SCORE_FIELDS:
        if key in payload and key not in seen:
            add(key, payload[key])
            seen.add(key)
    for key in sorted(payload):
        if key not in seen:
            add(key, payload[key])
    return "\n".join(lines)
