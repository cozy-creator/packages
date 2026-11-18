"""Advisories in locked pins (xs-196, cl-1212).

One OSV querybatch asks for every `name==version` line on stdin (uv export's format; a URL or git
pin has no version to ask and is left out), then each advisory found is read by its ID. Any
advisory, whatever its severity, fails unless the known file lists its ID or an alias; a withdrawn
one doesn't count. A timeout, reset, 408, 429 or 5xx is asked again with backoff, shown as attempt
N/5; after the fifth, or on any other refusal, it fails with one line naming the fault, never a
pass. The same file is in cozy-creator/cozy-runtime and cozy-creator/packages.
    uv export ... | python3 scripts/osv-audit.py advisories-known.txt
"""

import http.client
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://api.osv.dev/v1"
ATTEMPTS = 5
TIMEOUT = 30
BATCH = 1000  # OSV's most queries in one querybatch
PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)")
KNOWN = re.compile(r"^([A-Z]+-[A-Za-z0-9-]+)", re.M)


class Fault(Exception):
    """Nothing was judged; the message says why."""


def ask(path: str, body: object = None) -> dict[str, object]:
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "osv-audit"}
    attempt = 0
    while True:
        attempt += 1
        wait = 0.0
        # A new request each attempt: urllib's proxy handler rewrites the one it's given.
        request = urllib.request.Request(f"{API}/{path}", data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return obj(json.loads(response.read()), path)
        except urllib.error.HTTPError as e:
            if e.code not in (408, 429) and e.code < 500:
                raise Fault(f"api.osv.dev refused {path}: HTTP {e.code} {e.read()[:200]!r}") from e
            reason = f"HTTP {e.code}"
            after = e.headers.get("Retry-After", "")
            wait = min(float(after), 60) if after.isdigit() else 0
        except urllib.error.URLError as e:
            timed_out = isinstance(e.reason, TimeoutError)
            reason = f"no answer in {TIMEOUT} s" if timed_out else str(e.reason)
        except TimeoutError:
            reason = f"no answer in {TIMEOUT} s"
        except (OSError, http.client.HTTPException, ValueError) as e:
            reason = f"{type(e).__name__}: {e}"
        if attempt == ATTEMPTS:
            raise Fault(f"infrastructure fault: api.osv.dev {path}: {reason} ({ATTEMPTS} attempts)")
        delay = wait or 2 ** (attempt - 1) * (1 + random.random() / 2)
        print(
            f"osv-audit: {path}: {reason}; attempt {attempt + 1}/{ATTEMPTS} in {delay:.0f} s",
            file=sys.stderr,
        )
        time.sleep(delay)


def obj(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise Fault(f"api.osv.dev {where}: an object was expected, got {type(value).__name__}")
    return {str(k): v for k, v in value.items()}


def arr(value: object, where: str) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise Fault(f"api.osv.dev {where}: a list was expected, got {type(value).__name__}")
    return list(value)


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def found(pins: list[tuple[str, str]]) -> list[set[str]]:
    """The advisory IDs OSV lists for each pin, every page."""
    ids: list[set[str]] = [set() for _ in pins]
    pending: list[tuple[int, object]] = [(i, None) for i in range(len(pins))]
    while pending:
        chunk, pending = pending[:BATCH], pending[BATCH:]
        queries = [
            {"package": {"name": pins[i][0], "ecosystem": "PyPI"}, "version": pins[i][1]}
            | ({"page_token": token} if token else {})
            for i, token in chunk
        ]
        results = arr(ask("querybatch", {"queries": queries}).get("results"), "results")
        if len(results) != len(chunk):
            raise Fault(f"api.osv.dev querybatch: {len(results)} results for {len(chunk)} queries")
        for (i, _), result in zip(chunk, results, strict=True):
            answer = obj(result, "result")
            for vuln in arr(answer.get("vulns"), "vulns"):
                ids[i].add(str(obj(vuln, "vuln")["id"]))
            if answer.get("next_page_token"):
                pending.append((i, answer["next_page_token"]))
    return ids


def fixes(advisory: dict[str, object], name: str) -> str:
    out: set[str] = set()
    for item in arr(advisory.get("affected"), "affected"):
        affected = obj(item, "affected")
        package = obj(affected.get("package", {}), "package")
        if canonical(str(package.get("name", ""))) != name:
            continue
        for r in arr(affected.get("ranges"), "ranges"):
            for e in arr(obj(r, "range").get("events"), "events"):
                if "fixed" in (event := obj(e, "event")):
                    out.add(str(event["fixed"]))
    return ", ".join(sorted(out)) or "none"


def main() -> int:
    known_path = sys.argv[1]
    known = set(KNOWN.findall(Path(known_path).read_text()))
    pins = sorted({(canonical(m[1]), m[2]) for line in sys.stdin if (m := PIN.match(line))})
    if not pins:
        raise Fault("no locked pins on stdin")
    hits = found(pins)
    ids = sorted(set[str]().union(*hits))
    with ThreadPoolExecutor(8) as pool:
        advisories = dict(zip(ids, pool.map(lambda i: ask(f"vulns/{i}"), ids), strict=True))
    unknown: list[str] = []
    held: list[str] = []
    for (name, version), pin_ids in zip(pins, hits, strict=True):
        seen: set[str] = set()
        for i in sorted(pin_ids):
            advisory = advisories[i]
            if i in seen or advisory.get("withdrawn"):
                continue
            names = {i} | {str(a) for a in arr(advisory.get("aliases"), "aliases")}
            seen |= names
            if listed := sorted(known & names):
                held.append(f"{name} {version} {', '.join(listed)}")
                continue
            summary = advisory.get("summary") or str(advisory.get("details", "")).split("\n")[0]
            also = f" ({', '.join(sorted(names - {i}))})" if names - {i} else ""
            fixed = fixes(advisory, name)
            unknown.append(f"{name} {version}: {i}{also} {str(summary)[:120]}; fixed in {fixed}")
    note = f" (known: {'; '.join(held)})" if held else ""
    for line in unknown:
        print(line)
    if unknown:
        print(
            f"osv-audit: {len(unknown)} unknown advisories in {len(pins)} locked pins{note}: "
            f"move each pin past it, or list it in {known_path} with why it stays"
        )
        return 1
    print(f"osv-audit: no unknown advisory in {len(pins)} locked pins{note}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fault as e:
        print(f"osv-audit: nothing judged: {e}", file=sys.stderr)
        sys.exit(2)
