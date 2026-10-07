"""Terminal menu for the belief-update suite.

`python rebus.py` prints a banner and a menu. Numbers and slash commands call
the same handlers. Training and comparison run in subprocesses. This file does
not write metrics.json and does not import a candidate train.py.
"""

from __future__ import annotations

import json
import select
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from interface.metrics_view import TAGLINE, format_metrics_list
from interface.trace_page import load_traces, run_files
from partner import speak


ROOT = Path(__file__).resolve().parent

BANNER = r"""
 █████╗ ███████╗███╗   ███╗██╗
██╔══██╗██╔════╝████╗ ████║██║
███████║███████╗██╔████╔██║██║
██╔══██║╚════██║██║╚██╔╝██║██║
██║  ██║███████║██║ ╚═╝ ██║███████╗
╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝
ALTERED STATE MACHINE LEARNING
""".strip("\n")

MENU = """
  1  /train      Train the baseline
  2  /compare    Compare a named arm, or a candidate file
  3  /trace      Open the belief-trace page, then return here
  4  /readings   Write cue stance, polarity, and richness
  5  /score      Show BeliefUpdateScore
  6  /test       Run the integrity tests
  7  /help       Show this menu
  8  /quit       Leave

/arms lists the edits. arm4 is the prior-precision edit.
/partner reads a finished trace. It does not train.
/compare episodes=50 seed=7 output=logs/seed7-arm4 candidate=arm4
/compare candidate=peft
/trace output=logs/baseline-seed7
/partner output=logs/baseline-seed7
""".strip("\n")

COMMANDS = {
    "1": "train",
    "2": "compare",
    "3": "trace",
    "4": "readings",
    "5": "score",
    "6": "test",
    "7": "help",
    "8": "quit",
}

ARM_CHOICES = {
    "arm2": "arm2",
    "arm3": "arm3",
    "arm4": "arm4",
    "arm4_rebus": "arm4",
    "rebus": "arm4",
    "arm5": "arm5",
}

ARMS_TEXT = """
arm2   Reward learning rate 0.30. Everything else stays at the baseline.
arm3   Reward learning rate 0.30, punishment learning rate 0.22, stickiness 0.25, sensitivity down on quiet updates and up after a large belief move.
arm4   Prior precision 0.5 and prediction-error gain 1, on the belief only.
arm5   The arm4 settings for the first 25 episodes, then the baseline constants. Evaluation reads the closed window.
""".strip("\n")

Runner = Callable[[List[str]], int]
Ask = Callable[[str], str]


class Outcome:
    def __init__(self, text: str = "", quit: bool = False) -> None:
        self.text = text
        self.quit = quit


def banner() -> str:
    return BANNER


def menu() -> str:
    return MENU


def parse_line(line: str) -> Tuple[str, Dict[str, str]]:
    parts = line.strip().split()
    if not parts:
        return "", {}
    head = parts[0]
    if head in COMMANDS:
        name = COMMANDS[head]
        tokens = parts[1:]
    elif head.startswith("/"):
        name = head[1:].lower()
        tokens = parts[1:]
    else:
        name = head.lower()
        tokens = parts[1:]
    spec: Dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        spec[key.strip().lower()] = value.strip()
    return name, spec


def _field(
    spec: Dict[str, str],
    key: str,
    ask: Optional[Ask],
    label: str,
    default: Optional[str] = None,
) -> Optional[str]:
    if key in spec:
        return spec[key]
    if ask is None:
        return default
    hint = f" [{default}]" if default is not None else ""
    answer = ask(f"{label}{hint}: ").strip()
    if answer:
        return answer
    return default


def _integer(value: str) -> Optional[int]:
    try:
        return int(value)
    except ValueError:
        return None


def project_python() -> str:
    """Prefer the repo virtualenv so training can import PyTorch."""
    for parts in (("bin", "python"), ("Scripts", "python.exe")):
        candidate = ROOT.joinpath(".venv", *parts)
        if candidate.is_file():
            return str(candidate)
    return sys.executable


_TRACE_PAGES: Dict[str, subprocess.Popen[str]] = {}


def present_command_output(text: str) -> str:
    rendered: List[str] = []
    for line in text.splitlines():
        if line.startswith("eval_metrics="):
            try:
                payload = json.loads(line[len("eval_metrics=") :])
            except json.JSONDecodeError:
                rendered.append(line)
            else:
                rendered.append(format_metrics_list(payload))
        else:
            rendered.append(line)
    if not rendered:
        return ""
    return "\n".join(rendered) + "\n"


def _run(cmd: List[str], runner: Runner) -> Outcome:
    try:
        code = runner(cmd)
    except KeyboardInterrupt:
        return Outcome("Returned to the menu.")
    if code == 0:
        return Outcome("Done.")
    return Outcome(f"Command exited {code}. Ran with {cmd[0]}.")


def default_runner(cmd: List[str]) -> int:
    try:
        completed = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    except KeyboardInterrupt:
        return 130
    rendered = present_command_output(completed.stdout)
    if rendered:
        sys.stdout.write(rendered)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    return completed.returncode


def show_score(path: Path) -> str:
    payload = json.loads(path.read_text())
    if "BeliefUpdateScore" not in payload:
        return f"{path} has no BeliefUpdateScore."
    return format_metrics_list(payload)


def dispatch(line: str, ask: Optional[Ask] = None, runner: Optional[Runner] = None) -> Outcome:
    name, spec = parse_line(line)
    if not name:
        return Outcome()
    run = runner or default_runner
    if name in {"help", "menu"}:
        return Outcome(menu() + "\n\n" + ARMS_TEXT)
    if name == "arms":
        return Outcome(ARMS_TEXT)
    if name == "quit":
        return Outcome(quit=True)
    if name == "train":
        return _train(spec, ask, run)
    if name == "compare":
        return _compare(spec, ask, run)
    if name == "trace":
        return _trace(spec, ask, run)
    if name == "readings":
        return _readings(spec, ask, run)
    if name == "score":
        return _score(spec, ask)
    if name == "partner":
        return _partner(spec, ask)
    if name == "test":
        return _run([project_python(), "-m", "unittest", "discover", "-s", "tests", "-t", "."], run)
    return Outcome(menu())


def _train(spec: Dict[str, str], ask: Optional[Ask], run: Runner) -> Outcome:
    episodes = _field(spec, "episodes", ask, "Episodes", "50") or "50"
    seed = _field(spec, "seed", ask, "Seed", "7") or "7"
    output = _field(spec, "output", ask, "Output directory", "logs/baseline-seed7") or "logs/baseline-seed7"
    if _integer(episodes) is None or _integer(seed) is None:
        return Outcome("Episodes and seed are integers.")
    cmd = [
        project_python(),
        str(ROOT / "train.py"),
        "--episodes",
        episodes,
        "--seed",
        seed,
        "--output-dir",
        output,
    ]
    device = spec.get("device")
    if device:
        cmd.extend(["--device", device])
    variant = spec.get("variant", "")
    if variant:
        cmd.extend(["--variant", variant])
    return _run(cmd, run)


def _compare(spec: Dict[str, str], ask: Optional[Ask], run: Runner) -> Outcome:
    episodes = _field(spec, "episodes", ask, "Episodes", "50") or "50"
    seed = _field(spec, "seed", ask, "Seed", "7") or "7"
    output = _field(spec, "output", ask, "Output root", "logs/local-run") or "logs/local-run"
    if "candidate" in spec:
        candidate = spec["candidate"]
    elif ask is None:
        candidate = ""
    else:
        candidate = ask("Candidate train.py (empty for baseline only): ").strip()
    if _integer(episodes) is None or _integer(seed) is None:
        return Outcome("Episodes and seed are integers.")
    cmd = [
        project_python(),
        str(ROOT / "scripts" / "local_runner.py"),
        "--episodes",
        episodes,
        "--seed",
        seed,
        "--output-root",
        output,
    ]
    if candidate == "peft":
        cmd.extend(["--candidate-train-py", str(ROOT / "train_peft.py")])
    elif candidate in ARM_CHOICES:
        cmd.extend(["--candidate-variant", ARM_CHOICES[candidate]])
    elif candidate:
        cmd.extend(["--candidate-train-py", candidate])
    device = spec.get("device")
    if device:
        cmd.extend(["--device", device])
    return _run(cmd, run)


def stop_trace_pages() -> None:
    for proc in list(_TRACE_PAGES.values()):
        if proc.poll() is None:
            proc.terminate()
    _TRACE_PAGES.clear()


def _open_trace_page(cmd: List[str], port: str) -> Outcome:
    existing = _TRACE_PAGES.get(port)
    if existing is not None and existing.poll() is None:
        return Outcome(
            f"trace_page=http://127.0.0.1:{port}/\n"
            "That page is already open. The menu is free."
        )
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    url = ""
    if proc.stdout is not None:
        ready, _, _ = select.select([proc.stdout], [], [], 5)
        if ready:
            url = proc.stdout.readline().strip()
    if proc.poll() is not None and proc.returncode not in {0, None}:
        err = ""
        if proc.stderr is not None:
            err = proc.stderr.read().strip()
        return Outcome(err or f"Trace page exited {proc.returncode}.")
    _TRACE_PAGES[port] = proc
    if not url:
        url = f"trace_page=http://127.0.0.1:{port}/"
    return Outcome(
        f"{url}\n"
        "The menu is free. /score prints this same list. /quit closes a page this menu opened."
    )


def _trace(spec: Dict[str, str], ask: Optional[Ask], run: Runner) -> Outcome:
    output = spec.get("output", "")
    trace = spec.get("trace", "")
    if not output and not trace:
        if ask is None:
            return Outcome("An output directory or a trace path is required.")
        output = ask("Output directory [logs/baseline-seed7]: ").strip() or "logs/baseline-seed7"
    if output and not trace:
        _, _, error = run_files(Path(output))
        if error:
            return Outcome(error)
    elif trace and not Path(trace).is_file():
        return Outcome(f"No trace at {trace}.")
    metrics = spec.get("metrics", "")
    port = spec.get("port", "8765")
    if _integer(port) is None:
        return Outcome("Port is an integer.")
    cmd = [project_python(), str(ROOT / "scripts" / "serve_trace.py"), "--port", port]
    if output and not trace:
        cmd.extend(["--output-dir", output])
    if trace:
        cmd.extend(["--trace", trace])
    if metrics:
        cmd.extend(["--metrics", metrics])
    if run is default_runner:
        return _open_trace_page(cmd, port)
    return _run(cmd, run)


def _readings(spec: Dict[str, str], ask: Optional[Ask], run: Runner) -> Outcome:
    trace = _field(spec, "trace", ask, "Trace path")
    if not trace:
        return Outcome("A trace path is required.")
    if not Path(trace).is_file():
        return Outcome(f"No trace at {trace}.")
    cmd = [project_python(), str(ROOT / "scripts" / "read_trace.py"), "--trace", trace]
    output = spec.get("output", "")
    if output:
        cmd.extend(["--output", output])
    return _run(cmd, run)


def _partner(spec: Dict[str, str], ask: Optional[Ask]) -> Outcome:
    output = spec.get("output", "")
    trace = spec.get("trace", "")
    if not output and not trace:
        if ask is None:
            return Outcome("An output directory or a trace path is required.")
        output = ask("Output directory [logs/baseline-seed7]: ").strip() or "logs/baseline-seed7"
    if output and not trace:
        found, _, error = run_files(Path(output))
        if error or found is None:
            return Outcome(error or f"No trace at {output}.")
        trace = str(found)
    path = Path(trace)
    if not path.is_file():
        return Outcome(f"No trace at {trace}.")
    blocks = []
    for item in load_traces(path):
        blocks.append(f"{item.get('name', 'scenario')}\n{speak(item)}")
    return Outcome("\n\n".join(blocks))


def _score(spec: Dict[str, str], ask: Optional[Ask]) -> Outcome:
    metrics = _field(spec, "metrics", ask, "Metrics path")
    if not metrics:
        return Outcome("A metrics path is required.")
    path = Path(metrics)
    if not path.is_file():
        return Outcome(f"No metrics at {metrics}.")
    before = path.read_bytes()
    try:
        text = show_score(path)
    except json.JSONDecodeError:
        return Outcome(f"{path} is not a metrics file.")
    if path.read_bytes() != before:
        return Outcome("Refused to show a score that rewrote metrics.json.")
    return Outcome(text)


def main() -> None:
    print(banner())
    print(TAGLINE)
    print(menu())
    try:
        while True:
            try:
                line = input("asml> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            outcome = dispatch(line, ask=input)
            if outcome.text:
                print(outcome.text)
            if outcome.quit:
                return
    finally:
        stop_trace_pages()


if __name__ == "__main__":
    main()
