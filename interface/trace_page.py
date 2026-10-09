"""Belief-trace page. A keep/discard mark is a note. It does not change the score."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from contracts import HIGH_RISK_BELIEF_UPDATE_V1, contract_id_from_artifact
from interface.metrics_view import TAGLINE, format_metrics_list
from partner import draft_note, narrate_scenario
from readings import cue_stance, read_trace


def decisions_path(trace_path: Path) -> Path:
    return trace_path.with_suffix(".decisions.jsonl")


def append_decision(trace_path: Path, mark: str, note: str, scenario: str) -> Path:
    if mark not in {"keep", "discard"}:
        raise ValueError("mark must be keep or discard")
    path = decisions_path(trace_path)
    record = {"mark": mark, "note": note, "scenario": scenario}
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    return path


def run_files(directory: Path) -> tuple[Optional[Path], Optional[Path], str]:
    """Resolve the files train.py writes into one output directory."""
    directory = Path(directory)
    trace = directory / "trace.json"
    metrics = directory / "metrics.json"
    if trace.is_file():
        return trace, metrics if metrics.is_file() else None, ""
    sides = sorted(
        (child for child in directory.iterdir() if child.is_dir() and (child / "trace.json").is_file()),
        key=lambda path: path.name,
    ) if directory.is_dir() else []
    if sides:
        lines = ["This directory holds a comparison. Open one side:"]
        lines.extend(f"  output={side}" for side in sides)
        return None, None, "\n".join(lines)
    return None, None, f"No trace at {trace}."


def render_trace_page(
    traces: Sequence[Dict],
    score: Optional[float] = None,
    metrics: Optional[Dict] = None,
    partner_lines: Optional[Sequence[str]] = None,
    contract_id: Optional[str] = None,
) -> str:
    options = []
    panels = []
    drafts = []
    for index, trace in enumerate(traces):
        name = str(trace.get("name", f"scenario-{index}"))
        options.append(f'<option value="{index}">{html.escape(name)}</option>')
        spoken = partner_lines[index] if partner_lines is not None else narrate_scenario(trace)
        note = draft_note(trace)
        drafts.append(note)
        panels.append(_panel(index, trace, spoken, note))
    first_draft = drafts[0] if drafts else ""
    if metrics is None and score is not None:
        metrics = {"BeliefUpdateScore": float(score)}
    if contract_id is None and metrics is not None:
        contract_id = contract_id_from_artifact(metrics)
    contract_line = contract_id or "unknown (artifact has no contract metadata)"
    score_block = "Score is not loaded." if metrics is None else format_metrics_list(metrics)
    if contract_id == HIGH_RISK_BELIEF_UPDATE_V1:
        lede = (
            "High-risk v1 keeps controlled belief replay separate from closed-loop task outcomes. "
            "The bars show the trained policy's P(open); details include the observed event and safety gates. "
            "A human mark is a note only and does not change saved metrics."
        )
    else:
        lede = (
            f"{TAGLINE} The list is the same one /score prints. Refresh after you train this directory again. "
            "A keep or discard mark is written beside the trace. It does not change the score. "
            "The automatic keep from /compare is selection.json. "
            "Cue stance, polarity, and richness are readings on the finished trace. The score ignores them. "
            "The filled bar is the agent's P(open). The brown tick is the exact filter."
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Belief trace</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0; font: 16px/1.45 "Iowan Old Style", Palatino, Georgia, serif; color: #1c1915; background: #f3efe4; }}
  main {{ max-width: 46rem; margin: 0 auto; padding: 1.5rem 1.2rem 3rem; }}
  h1 {{ font-size: 1.6rem; font-weight: 600; margin-bottom: 0.2rem; }}
  p.lede {{ margin-top: 0; color: #3f3a33; }}
  p.contract {{ margin: 0.2rem 0 0.7rem; color: #1f4e3d; font: 0.9rem ui-monospace, "SF Mono", Menlo, monospace; }}
  pre.score {{ font: 0.92rem/1.45 ui-monospace, "SF Mono", Menlo, monospace; white-space: pre-wrap; background: #fffdf8; border: 1px solid #d9d0c1; padding: 0.7rem 0.9rem; margin: 0.8rem 0 1rem; }}
  select, textarea, button {{ font: inherit; }}
  select {{ margin: 0.6rem 0 1rem; padding: 0.3rem 0.4rem; }}
  article {{ display: none; background: #fffdf8; border: 1px solid #d9d0c1; padding: 0.8rem 1rem 1rem; }}
  article.active {{ display: block; }}
  .step {{ display: grid; grid-template-columns: 7rem 1fr; gap: 0.35rem 0.8rem; padding: 0.7rem 0; border-top: 1px solid #eee6d8; }}
  .cue {{ font-variant: small-caps; letter-spacing: 0.04em; }}
  .track {{ position: relative; height: 0.85rem; background: #e4d8c4; margin: 0.2rem 0 0.35rem; }}
  .track .agent {{ display: block; height: 100%; background: #1f4e3d; }}
  .track .exact {{ position: absolute; top: -0.15rem; width: 2px; height: 1.15rem; background: #8a4b08; }}
  .partner {{ margin: 0.4rem 0 0.8rem; }}
  .meta {{ color: #4a453d; font-size: 0.92rem; }}
  .gate {{ color: #8d1d18; }}
  form {{ margin-top: 1.2rem; display: grid; gap: 0.5rem; }}
  textarea {{ min-height: 4.5rem; padding: 0.4rem; }}
  .actions {{ display: flex; gap: 0.5rem; }}
  button {{ background: #1f4e3d; color: #f7f3ea; border: 0; padding: 0.4rem 0.8rem; cursor: pointer; }}
  button.discard {{ background: #5c4636; }}
  #status {{ min-height: 1.2rem; }}
</style>
</head>
<body>
<main>
  <h1>Belief trace</h1>
  <p class="contract">Active contract: {html.escape(str(contract_line))}</p>
  <p class="lede">{html.escape(lede)}</p>
  <pre class="score">{html.escape(score_block)}</pre>
  <label for="scenario">Scenario</label>
  <select id="scenario">{''.join(options)}</select>
  {''.join(panels)}
  <form id="mark">
    <label for="note">Note for the next training hypothesis</label>
    <textarea id="note" name="note">{html.escape(first_draft)}</textarea>
    <div class="actions">
      <button type="button" id="keep" data-mark="keep">Keep</button>
      <button type="button" class="discard" id="discard" data-mark="discard">Discard</button>
    </div>
    <p id="status"></p>
  </form>
</main>
<script>
const select = document.getElementById("scenario");
function show(index) {{
  document.querySelectorAll("article").forEach((node) => node.classList.remove("active"));
  const panel = document.getElementById("panel-" + index);
  if (!panel) return;
  panel.classList.add("active");
  const note = document.getElementById("note");
  if (note && panel.dataset.draft) note.value = panel.dataset.draft;
}}
select.addEventListener("change", () => show(select.value));
show(select.value || "0");
async function mark(kind) {{
  const response = await fetch("/decision", {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{
      mark: kind,
      note: document.getElementById("note").value,
      scenario: select.selectedOptions[0] ? select.selectedOptions[0].text : ""
    }})
  }});
  const payload = await response.json();
  document.getElementById("status").textContent = payload.ok
    ? "Saved " + kind + ". BeliefUpdateScore was not changed."
    : "Could not save.";
}}
document.getElementById("keep").addEventListener("click", () => mark("keep"));
document.getElementById("discard").addEventListener("click", () => mark("discard"));
</script>
</body>
</html>
"""


def _panel(index: int, trace: Dict, spoken: str, note: str) -> str:
    blurb = html.escape(str(trace.get("blurb", "")))
    outcome = html.escape(str(trace.get("outcome", "")))
    outcome_status = html.escape(str(trace.get("outcome_status", "")))
    ambiguous = bool(trace.get("ambiguous", False))
    status = ""
    if outcome_status:
        context = "ambiguous" if ambiguous else "clear"
        status = f" Evaluation: {outcome_status} for this {context} scenario."
    reading = read_trace(trace)
    summary = (
        f"Cue polarity {int(reading['cue_polarity_sum']):+d}. "
        f"Steps in the entropy band: {reading['steps_in_entropy_band']} of {reading['n_steps']}. "
        f"Partner-call changes: {reading['partner_call_changes']}. "
        "Polarity is left out of the return."
    )
    steps = []
    richness = list(reading["richness"])
    for step, row in zip(trace.get("steps", []), richness):
        steps.append(_step(step, row))
    active = " active" if index == 0 else ""
    return (
        f'<article id="panel-{index}" class="{active.strip()}" data-draft="{html.escape(note, quote=True)}">'
        f"<p>{blurb} Outcome: {outcome}.{status}</p>"
        f"<p class='partner'>{html.escape(spoken)}</p>"
        f"<p class='meta'>{html.escape(summary)}</p>"
        f"{''.join(steps)}</article>"
    )


def _step(step: Dict, row: Dict) -> str:
    belief = step.get("belief_after") or [0.0, 0.0]
    p_open = float(belief[0])
    exact = step.get("exact_belief")
    has_exact = isinstance(exact, (list, tuple)) and len(exact) == 2
    exact_open = float(exact[0]) if has_exact else None
    exact_marker = f"<span class='exact' style='left:{exact_open * 100:.1f}%'></span>" if exact_open is not None else ""
    track_title = "Filled bar is P(open). Brown tick is the exact filter." if has_exact else "Filled bar is the trained policy's P(open). No exact-filter marker is included in this trace."
    gate = "<div class='gate'>Commit gate would block this action.</div>" if step.get("gate_would_block") else ""
    aleatoric = step.get("aleatoric")
    aleatoric_text = "n/a" if aleatoric is None else f"{float(aleatoric):.2f}"
    entropy = step.get("epistemic_entropy")
    entropy_text = "n/a" if entropy is None else f"{float(entropy):.2f}"
    movement = step.get("belief_movement")
    movement_text = "n/a" if movement is None else f"{float(movement):.3f}"
    evidence_result = ""
    if step.get("ignored_evidence"):
        evidence_result = " Strong evidence was followed by almost no belief movement."
    elif step.get("moved_but_uncertain"):
        evidence_result = " Belief moved after strong evidence but remained uncertain."
    elif step.get("strong_evidence"):
        evidence_result = " Belief moved and was no longer uncertain after strong evidence."
    high_risk_fields = []
    if step.get("query_target"):
        high_risk_fields.append(f"query target {html.escape(str(step['query_target']))}")
    if step.get("deterministic_hazard_cue"):
        high_risk_fields.append("deterministic hazard cue")
    if step.get("hazard_lock_active"):
        high_risk_fields.append("hazard lock active")
    if step.get("critical_evidence_unavailable_unresolved"):
        high_risk_fields.append("critical evidence unavailable and unresolved")
    if step.get("access_lock_active"):
        high_risk_fields.append("unresolved critical-access lock active")
    if step.get("safety_gate_breached"):
        high_risk_fields.append("SAFETY GATE BREACHED")
    if "task_reward" in step:
        high_risk_fields.append(f"task reward {float(step['task_reward']):.2f}")
    high_risk_text = (" High-risk: " + "; ".join(high_risk_fields) + ".") if high_risk_fields else ""
    label = step.get("label", step.get("world", "unavailable"))
    return (
        "<div class='step'>"
        f"<div class='cue'>{html.escape(str(step.get('cue', '')))}</div>"
        "<div>"
        f"<div class='track' title='{html.escape(track_title, quote=True)}'>"
        f"<span class='agent' style='width:{p_open * 100:.1f}%'></span>"
        f"{exact_marker}</div>"
        "<div class='meta'>"
        f"action {html.escape(str(step.get('action', '')))}, "
        f"confirm {html.escape(str(step.get('requery_action', '')))}, "
        f"label {html.escape(str(label))}, "
        f"epistemic {entropy_text}, movement {movement_text}, aleatoric {aleatoric_text}, "
        f"stance {html.escape(str(cue_stance(step)))}, "
        f"richness {html.escape(str(row.get('band_label', 'unavailable')))}."
        f"{html.escape(evidence_result)}"
        f"{html.escape(high_risk_text)}"
        f"</div>{gate}</div></div>"
    )


def load_traces(path: Path) -> List[Dict]:
    return load_trace_document(path)[1]


def load_trace_document(path: Path) -> tuple[Optional[str], List[Dict]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "traces" in payload:
        contract_id = contract_id_from_artifact(payload)
        return contract_id, list(payload["traces"])
    traces = list(payload)
    contract_id = contract_id_from_artifact(traces[0]) if traces and isinstance(traces[0], dict) else None
    return contract_id, traces


def make_server(
    trace_path: Path,
    score: Optional[float] = None,
    port: int = 0,
    metrics_path: Optional[Path] = None,
):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    def page_bytes() -> bytes:
        payload = None
        if metrics_path is not None and Path(metrics_path).is_file():
            payload = json.loads(Path(metrics_path).read_text())
        elif score is not None:
            payload = {"BeliefUpdateScore": float(score)}
        from partner import speak

        trace_contract, loaded = load_trace_document(trace_path)
        metrics_contract = contract_id_from_artifact(payload) if payload is not None else None
        if trace_contract and metrics_contract and trace_contract != metrics_contract:
            raise ValueError("Trace and metrics files use different canonical contracts.")
        spoken = [speak(trace) for trace in loaded]
        return render_trace_page(
            loaded, metrics=payload, partner_lines=spoken,
            contract_id=metrics_contract or trace_contract,
        ).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] not in {"/", "/index.html"}:
                self.send_error(404)
                return
            body = page_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/decision":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            try:
                append_decision(
                    trace_path,
                    str(payload.get("mark", "")),
                    str(payload.get("note", "")),
                    str(payload.get("scenario", "")),
                )
            except ValueError:
                body = json.dumps({"ok": False, "score_unchanged": True}).encode("utf-8")
                self.send_response(400)
            else:
                body = json.dumps({"ok": True, "score_unchanged": True}).encode("utf-8")
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            return

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
