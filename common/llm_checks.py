"""LLM gates for one problem (a few Opus calls each, no evolution).

  blindness: show the exact iteration-1 prompt of each arm and ask what
             combined_score measures. Blind arm should not know; hinted should.
  memory:    describe the task cold and ask for the best known value.

Usage: llm_checks.py <problem_dir> [samples]
Answers are saved under <problem_dir>/checks/.
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness
from openevolve.config import load_config
from openevolve.prompt.sampler import PromptSampler

BLIND_Q = """

---
Do NOT modify the program. In at most 90 words, answer:
(a) What task is this program solving?
(b) What exactly does `combined_score` measure? Give your single best-guess formula.
"""

MEMORY_Q = """Here is an optimisation task:

{desc}

In at most 60 words: is this a known benchmark instance, and what is the best known objective value for exactly this instance? If you do not know a value, say "unknown" rather than estimating."""


def ask(model, system, users, cwd):
    procs = []
    for user in users:
        cmd = ["claude", "-p", "--model", model, "--no-session-persistence",
               "--output-format", "text", "--max-budget-usd", "1"]
        if system:
            cmd += ["--system-prompt", system]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, cwd=cwd)
        p.stdin.write(user)
        p.stdin.close()
        procs.append(p)
    outs = []
    for p in procs:
        out = p.stdout.read()
        p.wait()
        outs.append(out.strip() or "[no output] " + p.stderr.read()[:300])
    return outs


def main(d, samples):
    d = Path(d).resolve()
    spec = importlib.util.spec_from_file_location("task", d / "task.py")
    task = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(d))
    spec.loader.exec_module(task)
    cfg = load_config(str(HERE.parent / "config.yaml"))
    model = cfg.llm.models[0].name
    cwd = tempfile.mkdtemp()
    (d / "checks").mkdir(exist_ok=True)
    for arm in ("baseline", "hinted"):
        path = d / arm / "initial_program.py"
        code = path.read_text()
        out, _ = harness.run_program(str(path), task.TIME_LIMIT_S)
        metrics = {"combined_score": task.score(out)[0]}
        initial = {"id": "initial", "code": code, "metrics": metrics, "language": "python"}
        prompt = PromptSampler(cfg.prompt).build_prompt(
            current_program=code, parent_program=code, program_metrics=metrics,
            previous_programs=[initial], top_programs=[initial], inspirations=[],
            language="python", evolution_round=1,
            diff_based_evolution=cfg.diff_based_evolution,
            feature_dimensions=cfg.database.feature_dimensions,
        )
        (d / "checks" / f"{arm}_prompt.txt").write_text(prompt["system"] + "\n=====\n" + prompt["user"])
        for k, ans in enumerate(ask(model, prompt["system"], [prompt["user"] + BLIND_Q] * samples, cwd)):
            (d / "checks" / f"{arm}_blindness_{k + 1}.txt").write_text(ans)
            print(f"--- {task.NAME} {arm} blindness {k + 1}\n{ans}\n")
    for k, ans in enumerate(ask(model, None, [MEMORY_Q.format(desc=task.DESCRIPTION)] * 2, cwd)):
        (d / "checks" / f"memory_{k + 1}.txt").write_text(ans)
        print(f"--- {task.NAME} memory {k + 1}\n{ans}\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
