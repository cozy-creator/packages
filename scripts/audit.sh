#!/usr/bin/env bash
# Advisories in the packages' locked dependencies (xs-196): pip-audit over every uv.lock's pins, nothing installed (a git pin
# it can't look up is left out). One not listed in pip-audit-known.txt fails. Each pin is asked once, in parallel batches with one
# version of a name each.
set -euo pipefail
cd "$(dirname "$0")/.."
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for lock in $(git ls-files '*/uv.lock'); do
  (cd "$(dirname "$lock")" && uv export --frozen --format requirements-txt --no-hashes --no-emit-project --all-extras --quiet)
done | grep -E '^[A-Za-z0-9][A-Za-z0-9._-]*==' | sed 's/ *;.*//' | sort -u \
  | awk -F== -v dir="$work" '{ print > (dir "/" ++seen[tolower($1)] ".txt") }'
ignore=$(grep -oE '^[A-Z]+-[A-Za-z0-9-]+' pip-audit-known.txt | sed 's/^/--ignore-vuln /' | tr '\n' ' ')
uvx pip-audit@2.9.0 --version > /dev/null
for batch in "$work"/*.txt; do
  # shellcheck disable=SC2086
  uvx pip-audit@2.9.0 --disable-pip --no-deps --progress-spinner off --cache-dir "${batch%.txt}.cache" -r "$batch" $ignore > "${batch%.txt}.out" 2>&1 &
done
status=0
for job in $(jobs -p); do wait "$job" || status=1; done
grep -hv '^WARNING' "$work"/*.out | grep -v '^No known vulnerabilities found' || true
exit "$status"
