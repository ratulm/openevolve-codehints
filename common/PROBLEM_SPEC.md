# Problem specification for the hints study

## What the study is

We measure whether hints in a program's comments, names and error messages help
OpenEvolve (an LLM-driven evolutionary code optimiser) improve that program.
Each problem has two arms whose initial programs behave identically:

- **blind** (`baseline/`): no natural-language cues at all.
- **hinted** (`hinted/`): comments, descriptive names and evaluator error
  messages that state the task.

The mutation model is strong and has memorised standard benchmark problems, so
every problem must be **non-standard**: a known problem family with a changed
instance and a changed scoring rule, such that a solver who assumes the standard
objective does clearly worse than one who knows the real one.

`p01_circles_lshape/` is a complete, working example. Read every
file in it before you start, and `common/harness.py`, `common/lint_blind.py` and
`common/check_problem.py`.

## Files to produce in your problem directory `<name>/`

| File | Content |
|---|---|
| `task.py` | `NAME`, `DESCRIPTION`, `TIME_LIMIT_S = 30`, and `score(output) -> (score, message)` |
| `evaluator.py` | exact copy of `common/evaluator_blind_template.py` |
| `evaluator_hinted.py` | exact copy of `common/evaluator_hinted_template.py` |
| `baseline/initial_program.py` | blind arm |
| `hinted/initial_program.py` | hinted arm |
| `reference.py` | offline solver that writes `reference_output.pkl` and `standard_output.pkl` |
| `reference_output.pkl`, `standard_output.pkl` | produced by `reference.py` |

## `task.py`

- `score(output)` receives whatever the program's `run()` returned.
- Valid output: return `(value, None)` where higher is better and the reference
  solution scores exactly 1.0 (store the normalising constant in `task.py`).
  For minimisation problems use `reference_cost / cost` or similar. Scores above
  1.0 must be possible if someone beats the reference.
- Invalid output: return `(0.0, message)` with a specific, helpful message
  (which element broke which rule, with numbers). Check shapes, types, NaN/inf
  and every constraint. Never raise.
- Deterministic and fast (well under 1 second). No randomness unless seeded.
- `DESCRIPTION`: a complete plain-language statement of the real task, including
  the instance and the exact objective. It is used to ask the model whether it
  knows this problem.
- Guard against cheap exploits: `score` must recompute everything itself and not
  trust any value the program reports about its own quality.

## Both initial programs

- Python with numpy and scipy only. Deterministic. `run()` takes no arguments and
  finishes in under 2 seconds.
- One `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` pair. The evolved block holds
  a deliberately naive but valid solution method. After the block, a fixed tail
  defines `run()` (and holds any instance data the program needs).
- The two arms must return **identical output**: same algorithm, same constants,
  same data, same order of operations. Only comments and identifiers differ.
- The program must **not compute its own score**. The true scoring rule lives
  only in `task.py`. The initial code should make the constraints inferable
  (as the wall and notch terms do in problem 1) but must not contain the true
  objective.
- Keep it small: roughly 30–80 lines.
- Initial score must be valid, above 0, and at most 0.60, so there is headroom.

## Blind arm rules (enforced by `common/lint_blind.py`)

- No comments except the two EVOLVE-BLOCK markers. No docstrings. No string
  literals at all.
- Every identifier the file defines (functions, arguments, variables, constants)
  is at most 2 characters, chosen to be meaningless (`f`, `g`, `p`, `q`, `a1`).
  The only exception is the entry point `run`.
- Library names (`np.hypot`, `scipy.optimize`) are unavoidable and fine.

## Hinted arm rules

- A comment block at the top of the evolved block stating: the goal, the exact
  scoring formula with all constants and how it is normalised, every validity
  rule, the tolerance, and the 30-second limit. Say explicitly what the score is
  NOT if a standard assumption would be wrong.
- Descriptive names for every function, argument and variable, and short
  comments describing what each part of the current code does.
- **No solving advice.** Do not name or suggest algorithms, libraries or
  strategies for improving the program, and no TODOs. Describing what the
  existing naive code does is fine; saying how to do better is not.
- The hinted evaluator's error messages come from `task.score`, so nothing more
  is needed there.

## `reference.py` and the two pickles

- `reference_output.pkl`: the best solution you can find for the **true**
  objective, in the same format `run()` returns. Make it strong (spend up to a
  few minutes of compute, many restarts); the study normalises by it. It must
  score 1.0.
- `standard_output.pkl`: the best solution you can find for the **standard**
  version of the problem, meaning the objective a knowledgeable person would
  assume from the blind program alone, on the same instance and constraints. It
  must be valid under the true rules, score at most 0.90, and beat the initial
  program. If it scores above 0.90, strengthen the twist and try again.
- Running `reference.py` from the problem directory must regenerate both files.

## Acceptance

From the repository root, this must end with `ALL GATES PASS`:

    python common/check_problem.py <name>

## Hard limits

- Use `python` for everything.
- Do not call any LLM (no `claude` command), do not run OpenEvolve, and do not
  edit anything outside your own problem directory.
- You may adjust the details of the twist to pass the gates, but keep the
  problem family and the kind of twist you were given.

## What to report back

1. The final task and twist in two or three sentences.
2. The `check_problem.py` output line with initial, standard and reference scores.
3. How `standard_output.pkl` was computed and what "standard objective" you assumed.
4. How strong you believe the reference is (restarts, time, any sign it is not near-optimal).
5. Anything you are unsure about, including any way the hinted arm could game the score.
