"""Shared evaluation harness for the hints study.

Each problem directory has a `task.py` exposing:
    NAME          short identifier
    DESCRIPTION   plain-language statement of the full task (used for the memory check)
    TIME_LIMIT_S  wall-clock limit for one call of the program's run()
    score(output) -> (score, message)
        score   float, higher is better, 1.0 = the offline reference solution,
                0.0 for anything invalid
        message None when valid, otherwise a specific explanation of the failure

The program under test must define run() with no arguments.

evaluate_blind  returns only {"combined_score": x}: no error text, no other metrics.
evaluate_hinted returns the same score plus the failure message as an artifact,
                which OpenEvolve shows to the model.
"""
import os
import pickle
import subprocess
import sys
import tempfile

_RUNNER = """
import importlib.util, pickle, sys
spec = importlib.util.spec_from_file_location("program", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(sys.argv[2], "wb") as fh:
    pickle.dump(m.run(), fh)
"""


def run_program(program_path, time_limit):
    """Run program.run() in a subprocess. Returns (output, error_message)."""
    workdir = tempfile.mkdtemp()
    out = os.path.join(workdir, "out.pkl")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, os.path.abspath(program_path), out],
            capture_output=True, text=True, timeout=time_limit, cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return None, f"The program ran longer than the {time_limit} second limit."
    if proc.returncode != 0:
        return None, "The program raised an exception:\n" + proc.stderr[-1500:]
    try:
        with open(out, "rb") as fh:
            return pickle.load(fh), None
    except Exception as exc:
        return None, f"Could not read the value returned by run(): {exc}"


def _score(program_path, task):
    output, error = run_program(program_path, task.TIME_LIMIT_S)
    if error:
        return 0.0, error
    try:
        value, message = task.score(output)
    except Exception as exc:
        return 0.0, f"The value returned by run() could not be scored: {exc}"
    if message is not None:
        return 0.0, message
    return float(value), None


def evaluate_blind(program_path, task):
    return {"combined_score": _score(program_path, task)[0]}


def evaluate_hinted(program_path, task):
    from openevolve.evaluation_result import EvaluationResult

    value, message = _score(program_path, task)
    if message is None:
        return EvaluationResult(metrics={"combined_score": value})
    return EvaluationResult(
        metrics={"combined_score": 0.0}, artifacts={"error": "Scored 0. " + message}
    )
