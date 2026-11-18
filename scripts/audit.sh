#!/usr/bin/env bash
# Advisories in the packages' locked dependencies (xs-196): every uv.lock's pins, nothing installed, judged against OSV by
# scripts/osv-audit.py (cl-1212). One not listed in advisories-known.txt fails.
set -euo pipefail
cd "$(dirname "$0")/.."
for lock in $(git ls-files '*/uv.lock'); do
  (cd "$(dirname "$lock")" && uv export --frozen --format requirements-txt --no-hashes --no-emit-project --all-extras --quiet)
done | python3 scripts/osv-audit.py advisories-known.txt
