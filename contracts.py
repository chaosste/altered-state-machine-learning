"""Canonical experiment contract identifiers and artifact compatibility helpers."""

from __future__ import annotations

from typing import Mapping, Optional


EVIDENCE_RESPONSE_V3 = "evidence-response-v3"
HIGH_RISK_BELIEF_UPDATE_V1 = "high-risk-belief-update-v1"
DEFAULT_CONTRACT = EVIDENCE_RESPONSE_V3

CONTRACT_DESCRIPTIONS = {
    EVIDENCE_RESPONSE_V3: "Original belief-state experiment and historical scoring contract.",
    HIGH_RISK_BELIEF_UPDATE_V1: "Separate high-risk belief replay and closed-loop simulation suite.",
}


def resolve_contract(value: Optional[str] = None) -> str:
    contract_id = (value or DEFAULT_CONTRACT).strip().lower()
    if contract_id not in CONTRACT_DESCRIPTIONS:
        known = ", ".join(CONTRACT_DESCRIPTIONS)
        raise ValueError(f"Unknown contract {value!r}. Supported contracts: {known}.")
    return contract_id


def contract_id_from_artifact(payload: Mapping[str, object]) -> Optional[str]:
    """Read the canonical id, with a fallback for historical v3 artifacts."""
    value = payload.get("contract_id")
    if isinstance(value, str) and value:
        return value
    legacy = payload.get("evaluation_contract_version")
    if isinstance(legacy, str) and legacy:
        return legacy
    return None

