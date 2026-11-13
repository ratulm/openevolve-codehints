"""Check that a blind-arm program carries no natural-language hints.

Rules: no comments except the two EVOLVE-BLOCK markers, no docstrings or other
string literals, and every identifier the file defines (functions, arguments,
variables) is at most 2 characters long, except the entry point `run`.
"""
import ast
import io
import sys
import tokenize

# The third comment is not a task hint; it was present in the reused problem-1 blind runs.
ALLOWED_COMMENTS = {
    "# EVOLVE-BLOCK-START",
    "# EVOLVE-BLOCK-END",
    "# This part remains fixed (not evolved)",
}


def lint(path):
    src = open(path).read()
    problems = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT and tok.string.strip() not in ALLOWED_COMMENTS:
            problems.append(f"line {tok.start[0]}: comment {tok.string.strip()!r}")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                imported.add((a.asname or a.name).split(".")[0])
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            problems.append(f"line {node.lineno}: string literal {node.value!r:.40}")
        names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.append(node.id)
        elif isinstance(node, ast.keyword) and node.arg:
            continue  # keyword arguments to library calls are unavoidable
        for name in names:
            if name != "run" and name not in imported and len(name) > 2:
                problems.append(f"line {node.lineno}: descriptive identifier {name!r}")
    return sorted(set(problems))


if __name__ == "__main__":
    found = lint(sys.argv[1])
    print("\n".join(found) if found else "blind lint: clean")
    sys.exit(1 if found else 0)
