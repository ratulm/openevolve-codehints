"""Blindness check: show the model the exact iteration-1 prompt OpenEvolve
builds for an arm, and ask what it thinks the task and the score are."""
import subprocess
import sys
import tempfile
from pathlib import Path

from openevolve.config import load_config
from openevolve.prompt.sampler import PromptSampler

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import evaluator

QUESTION = """

---
Do NOT modify the program yet. Answer these three questions briefly instead:
1. What is this program for?
2. What exactly do you believe `combined_score` measures? Give your best-guess formula.
3. What approach would you take to raise the score?
"""


def build_prompt(arm, cfg):
    code = (HERE / arm / "initial_program.py").read_text()
    metrics = evaluator.evaluate(str(HERE / arm / "initial_program.py"))
    initial = {"id": "initial", "code": code, "metrics": metrics, "language": "python"}
    return PromptSampler(cfg.prompt).build_prompt(
        current_program=code,
        parent_program=code,
        program_metrics=metrics,
        previous_programs=[initial],
        top_programs=[initial],
        inspirations=[],
        language="python",
        evolution_round=1,
        diff_based_evolution=cfg.diff_based_evolution,
        feature_dimensions=cfg.database.feature_dimensions,
    )


if __name__ == "__main__":
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    cfg = load_config(str(HERE / "config.yaml"))
    model = cfg.llm.models[0].name
    workdir = tempfile.mkdtemp()  # empty cwd: the model cannot read the repo
    for arm in ("baseline", "hinted"):
        prompt = build_prompt(arm, cfg)
        (HERE / "blindness" / f"{arm}_prompt.txt").write_text(
            "=== SYSTEM ===\n" + prompt["system"] + "\n=== USER ===\n" + prompt["user"]
        )
        procs = [
            subprocess.Popen(
                ["claude", "-p", "--model", model, "--no-session-persistence",
                 "--output-format", "text", "--system-prompt", prompt["system"],
                 "--max-budget-usd", "1"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=workdir,
            )
            for _ in range(samples)
        ]
        for p in procs:
            p.stdin.write(prompt["user"] + QUESTION)
            p.stdin.close()
        for k, p in enumerate(procs):
            out = p.stdout.read()
            p.wait()
            (HERE / "blindness" / f"{arm}_answer_{k + 1}.txt").write_text(out or p.stderr.read())
            print(f"===== {arm} sample {k + 1} =====\n{out}\n")
