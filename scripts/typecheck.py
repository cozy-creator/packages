"""The type check's verdict: mypy's errors (stdin) against `.mypy-baseline.txt`.

A known error passes and a new one fails. A baseline line that no longer occurs fails too, so the
baseline is always exact; `prune` drops those lines and never adds one: the baseline only shrinks.
Errors are keyed without line numbers, so moving code does not churn it.
    mypy … | python scripts/typecheck.py check|prune|init
"""

import re
import sys
from collections import Counter
from pathlib import Path

BASELINE = Path(".mypy-baseline.txt")
ERROR = re.compile(r"^(?P<path>[^:\n]+):\d+(?::\d+)?: error: (?P<message>.*?)\s+\[(?P<code>[a-z-]+)\]$")


def keys(lines: list[str]) -> Counter[str]:
    found: Counter[str] = Counter()
    for line in lines:
        m = ERROR.match(line.rstrip("\n"))
        if m:
            found[f"{m['path']}: [{m['code']}] {m['message']}"] += 1
        elif ": error: " in line:
            sys.exit(f"typecheck: an error without a code, or unreadable: {line.strip()}")
    return found


def write(counts: Counter[str]) -> None:
    BASELINE.write_text("".join(f"{key}\n" for key in sorted(counts.elements())))


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    output = sys.stdin.readlines()
    errors = keys(output)
    if mode == "init":
        if BASELINE.exists():
            sys.exit(f"typecheck: {BASELINE} exists; it only shrinks (prune)")
        write(errors)
        print(f"typecheck: baseline of {sum(errors.values())} errors")
        return
    known = Counter(BASELINE.read_text().splitlines()) if BASELINE.exists() else Counter()
    new, gone = errors - known, known - errors
    if mode == "prune":
        write(known - gone)
        print(f"typecheck: pruned {sum(gone.values())}; baseline {sum((known - gone).values())}")
    for line in output:
        m = ERROR.match(line.rstrip("\n"))
        if m and new[f"{m['path']}: [{m['code']}] {m['message']}"]:
            print(line.rstrip("\n"))
    if new:
        print(f"typecheck: {sum(new.values())} new error(s) above; fix them (the baseline never grows)")
    if gone and mode == "check":
        print(f"typecheck: {sum(gone.values())} baseline error(s) fixed; run `scripts/typecheck.sh prune` and commit .mypy-baseline.txt")
        for key in sorted(gone.elements()):
            print(f"  fixed: {key}")
    if new or (gone and mode == "check"):
        sys.exit(1)
    print(f"typecheck: ok, {sum(errors.values())} known error(s) left")


if __name__ == "__main__":
    main()
