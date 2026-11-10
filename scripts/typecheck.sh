#!/usr/bin/env bash
# The type check: mypy --strict ([tool.mypy]) against the pinned real dependencies. Errors in
# .mypy-baseline.txt pass; a new one fails, and so does a fixed one left in the baseline.
#   scripts/typecheck.sh          check
#   scripts/typecheck.sh prune    drop fixed errors from the baseline (it never grows)
#   scripts/typecheck.sh lock     re-pin typecheck-requirements.txt from typecheck-requirements.in
set -euo pipefail
cd "$(dirname "$0")/.."
req=scripts/typecheck-requirements
venv=.venv-typecheck

if [ "${1:-check}" = lock ]; then
  exec uv pip compile -q --python-version 3.12 --python-platform x86_64-manylinux_2_28 --no-header --no-annotate "$req.in" -o "$req.txt"
fi
if ! cmp -s "$req.txt" "$venv/requirements.txt"; then
  uv venv -q --clear --python 3.12 "$venv"
  uv pip install -q --python "$venv/bin/python" -r "$req.txt"
  cp "$req.txt" "$venv/requirements.txt"
fi
out=$(mktemp)
trap 'rm -f "$out"' EXIT
# mypy exits 1 when it reports errors (the baseline judges those) and 2 when it cannot run.
status=0
"$venv/bin/mypy" --python-executable "$venv/bin/python" --no-error-summary --no-pretty --show-error-codes > "$out" || status=$?
if [ "$status" -gt 1 ]; then cat "$out" >&2; exit "$status"; fi
"$venv/bin/python" scripts/typecheck.py "${1:-check}" < "$out"
