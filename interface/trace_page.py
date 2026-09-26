"""Belief-trace page. A keep/discard mark is a note. It does not change the score."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


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


def render_trace_page(traces: Sequence[Dict], score: Optional[float] = None) -> str:
    options = []
    panels = []
    for index, trace in enumerate(traces):
        name = str(trace.get("name", f"scenario-{index}"))
        options.append(f'<option value="{index}">{html.escape(name)}</option>')
        panels.append(_panel(index, trace))
    score_line = "Score is not loaded." if score is None else f"BeliefUpdateScore {score:.3f}"
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
  select, textarea, button {{ font: inherit; }}
  select {{ margin: 0.6rem 0 1rem; padding: 0.3rem 0.4rem; }}
  article {{ display: none; background: #fffdf8; border: 1px solid #d9d0c1; padding: 0.8rem 1rem 1rem; }}
  article.active {{ display: block; }}
  .step {{ display: grid; grid-template-columns: 7rem 1fr; gap: 0.35rem 0.8rem; padding: 0.7rem 0; border-top: 1px solid #eee6d8; }}
  .cue {{ font-variant: small-caps; letter-spacing: 0.04em; }}
  .track {{ position: relative; height: 0.85rem; background: #e4d8c4; margin: 0.2rem 0 0.35rem; }}
  .track .agent {{ display: block; height: 100%; background: #1f4e3d; }}
  .track .exact {{ position: absolute; top: -0.15rem; width: 2px; height: 1.15rem; background: #8a4b08; }}
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
  <p class="lede">{html.escape(score_line)}. A keep or discard mark is written beside the trace. It does not change the score. The filled bar is the agent's P(open). The brown tick is the exact filter.</p>
  <label for="scenario">Scenario</label>
  <select id="scenario">{''.join(options)}</select>
  {''.join(panels)}
  <form id="mark">
    <label for="note">Note for the next training hypothesis</label>
    <textarea id="note" name="note"></textarea>
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
  if (panel) panel.classList.add("active");
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


def _panel(index: int, trace: Dict) -> str:
    blurb = html.escape(str(trace.get("blurb", "")))
    outcome = html.escape(str(trace.get("outcome", "")))
    steps = []
    for step in trace.get("steps", []):
        steps.append(_step(step))
    active = " active" if index == 0 else ""
    return (
        f'<article id="panel-{index}" class="{active.strip()}">'
        f"<p>{blurb} Outcome: {outcome}.</p>"
        f"{''.join(steps)}</article>"
    )


def _step(step: Dict) -> str:
    belief = step.get("belief_after") or [0.0, 0.0]
    exact = step.get("exact_belief") or [0.0, 0.0]
    p_open = float(belief[0])
    exact_open = float(exact[0])
    gate = "<div class='gate'>Commit gate would block this action.</div>" if step.get("gate_would_block") else ""
    aleatoric = step.get("aleatoric")
    aleatoric_text = "n/a" if aleatoric is None else f"{float(aleatoric):.2f}"
    entropy = step.get("epistemic_entropy")
    entropy_text = "n/a" if entropy is None else f"{float(entropy):.2f}"
    return (
        "<div class='step'>"
        f"<div class='cue'>{html.escape(str(step.get('cue', '')))}</div>"
        "<div>"
        "<div class='track' title='Filled bar is P(open). Brown tick is the exact filter.'>"
        f"<span class='agent' style='width:{p_open * 100:.1f}%'></span>"
        f"<span class='exact' style='left:{exact_open * 100:.1f}%'></span></div>"
        "<div class='meta'>"
        f"action {html.escape(str(step.get('action', '')))}, "
        f"confirm {html.escape(str(step.get('requery_action', '')))}, "
        f"label {html.escape(str(step.get('label', '')))}, "
        f"epistemic {entropy_text}, aleatoric {aleatoric_text}"
        f"</div>{gate}</div></div>"
    )


def load_traces(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "traces" in payload:
        return list(payload["traces"])
    return list(payload)


def make_server(trace_path: Path, score: Optional[float] = None, port: int = 0):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    page = render_trace_page(load_traces(trace_path), score=score)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] not in {"/", "/index.html"}:
                self.send_error(404)
                return
            body = page.encode("utf-8")
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
