"""Offline gates for one problem directory (no LLM calls).

  1. blind lint on baseline/initial_program.py
  2. matched start: both arms return identical output and the same score
  3. reference_output.pkl scores ~1.0 and standard_output.pkl (best solution to
     the standard, un-twisted version of the problem) scores <= 0.90
  4. headroom: initial score <= 0.60
  5. the hinted evaluator explains an invalid output
"""
import importlib.util
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness
import lint_blind


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same(a, b):
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return np.array_equal(np.asarray(a), np.asarray(b))


def main(d):
    d = os.path.abspath(d)
    sys.path.insert(0, d)
    task = load(os.path.join(d, "task.py"), "task")
    ok = True

    def gate(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")

    found = lint_blind.lint(os.path.join(d, "baseline", "initial_program.py"))
    gate("blind lint", not found, "; ".join(found[:5]))

    outs, scores = {}, {}
    for arm in ("baseline", "hinted"):
        out, err = harness.run_program(os.path.join(d, arm, "initial_program.py"), task.TIME_LIMIT_S)
        if err:
            gate(f"{arm} runs", False, err[:300])
            return False
        outs[arm] = out
        scores[arm] = task.score(out)
    gate("matched start", same(outs["baseline"], outs["hinted"]) and scores["baseline"] == scores["hinted"],
         f"baseline={scores['baseline']} hinted={scores['hinted']}")
    init = scores["baseline"][0]
    gate("initial is valid", scores["baseline"][1] is None and init > 0, f"score={init:.4f}")
    gate("headroom (initial <= 0.60)", init <= 0.60, f"score={init:.4f}")

    ref = task.score(pickle.load(open(os.path.join(d, "reference_output.pkl"), "rb")))
    gate("reference scores ~1.0", ref[1] is None and abs(ref[0] - 1.0) < 1e-6, f"score={ref[0]:.6f} {ref[1] or ''}")
    std = task.score(pickle.load(open(os.path.join(d, "standard_output.pkl"), "rb")))
    gate("standard guess is valid and <= 0.90", std[1] is None and std[0] <= 0.90, f"score={std[0]:.4f} {std[1] or ''}")
    gate("standard guess beats the start", std[0] > init, f"{std[0]:.4f} > {init:.4f}")

    blind = load(os.path.join(d, "evaluator.py"), "ev_blind").evaluate(os.path.join(d, "baseline", "initial_program.py"))
    gate("blind evaluator returns only combined_score", list(blind) == ["combined_score"], str(blind))
    hinted = load(os.path.join(d, "evaluator_hinted.py"), "ev_hinted").evaluate(os.path.join(d, "hinted", "initial_program.py"))
    gate("hinted evaluator matches", abs(hinted.metrics["combined_score"] - blind["combined_score"]) < 1e-12)
    gate("DESCRIPTION present", len(getattr(task, "DESCRIPTION", "")) > 100)
    print("RESULT:", "ALL GATES PASS" if ok else "SOME GATES FAIL",
          f"| initial={init:.4f} standard={std[0]:.4f} reference={ref[0]:.4f}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
