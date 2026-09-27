"""Language-model partner for a finished belief trace.

The partner speaks after the run. It does not choose actions and it is not
called from act or requery. See program.md.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Mapping, Optional

from readings import cue_stance, read_trace


PARAPHRASE_RULES = (
    "Paraphrase the reading in plain English. "
    "Do not invent a partner type, a BeliefUpdateScore, or a keep or discard decision. "
    "Do not add facts that are not in the reading."
)

Paraphraser = Callable[[str, str], str]


def narrate_scenario(trace: Mapping[str, object]) -> str:
    """Deterministic reading of one finished scenario."""
    name = str(trace.get("name", "scenario"))
    blurb = str(trace.get("blurb", "")).strip()
    outcome = str(trace.get("outcome", "undetermined"))
    steps = list(trace.get("steps") or [])
    if not steps:
        return (
            f"{name}. {blurb} This finished encounter has no steps. "
            "The partner is reading a saved trace. This reading does not change BeliefUpdateScore."
        )
    step = steps[-1]
    reading = read_trace(trace)
    richness = list(reading["richness"])
    row = richness[-1] if richness else {}
    cue = str(step.get("cue", ""))
    stance = cue_stance(step)
    action = str(step.get("action", ""))
    confirm = str(step.get("requery_action", ""))
    label = str(step.get("label", ""))
    epistemic = _num(step.get("epistemic_entropy"))
    aleatoric = _num(step.get("aleatoric"))
    band = str(row.get("band_label", "unavailable"))
    situation = blurb if blurb else name
    return (
        f"{situation} At the end of this finished encounter the belief call was {outcome}, "
        f"and the true label on the last step was {label}. "
        f"The last cue was {cue}, which is a {stance} stance. "
        f"The agent chose {action}, and when asked again with no new evidence it chose {confirm}. "
        f"Epistemic uncertainty was {epistemic}, which is doubt about the partner type. "
        f"Aleatoric uncertainty was {aleatoric}, which is noise in the cue. "
        f"Richness was {band}. "
        "The partner is reading a saved trace. This reading does not change BeliefUpdateScore."
    )


def draft_note(trace: Mapping[str, object]) -> str:
    """A note for the box. It is not a keep or discard decision."""
    name = str(trace.get("name", "scenario"))
    outcome = str(trace.get("outcome", "undetermined"))
    steps = list(trace.get("steps") or [])
    action = str(steps[-1].get("action", "")) if steps else "none"
    label = str(steps[-1].get("label", "")) if steps else "unknown"
    return (
        f"Draft for {name}. Final belief call {outcome}. Last action {action}, label {label}. "
        "This draft is not a keep or discard decision."
    )


def speak(
    trace: Mapping[str, object],
    model: Optional[str] = None,
    paraphraser: Optional[Paraphraser] = None,
) -> str:
    """Return the reading, or a labeled paraphrase when a local model is set."""
    reading = narrate_scenario(trace)
    if paraphraser is not None:
        return "Paraphrase. " + paraphraser(reading, PARAPHRASE_RULES).strip()
    chosen = os.environ.get("REBUS_PARTNER_MODEL", "").strip() if model is None else model.strip()
    if not chosen:
        return reading
    try:
        text = paraphrase_with_local_model(reading, chosen)
    except Exception as exc:
        return reading + f"\nLocal paraphrase was not run ({exc})."
    return "Paraphrase. " + text.strip()


def paraphrase_with_local_model(reading: str, model_path: str) -> str:
    """Paraphrase with a local model. Nothing is downloaded."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is not installed, and nothing was downloaded") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    prompt = f"{PARAPHRASE_RULES}\n\nReading:\n{reading}\n\nParaphrase:"
    encoded = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**encoded, max_new_tokens=180, do_sample=False)
    new_tokens = output[0][encoded["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("the local model returned an empty paraphrase")
    return text


def _num(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"
