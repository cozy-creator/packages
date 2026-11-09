#!/usr/bin/env bash
# The type check: basedpyright ([tool.basedpyright]) against the pinned real dependencies. Errors recorded in
# .basedpyright/baseline.json pass; a new one fails.
#   scripts/typecheck.sh            check
#   scripts/typecheck.sh baseline   rewrite the baseline, after removing errors
#   scripts/typecheck.sh lock       re-pin typecheck-requirements.txt from typecheck-requirements.in
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
case "${1:-check}" in
  check) exec "$venv/bin/basedpyright" --pythonpath "$venv/bin/python" ;;
  baseline) exec "$venv/bin/basedpyright" --pythonpath "$venv/bin/python" --writebaseline ;;
  *) echo "usage: scripts/typecheck.sh [check|baseline|lock]" >&2; exit 2 ;;
esac
