"""A PR names the requested features it removes (tracker xs-199).

Lists each definition the net diff BASE..HEAD deletes from the tree whose adding commit (`git blame -w` at BASE, then
past later edits of its line) names a tracker issue or added a changelog.d entry: a Rust fn, static, const or type, a
Python module-level def or class or a class's method, an exported JS/TS function, class or const, a clap flag or verb,
an axum route, a cell file or shell function under e2e/, release-gate/ or checks/. A name still defined anywhere at
HEAD was moved or renamed, not deleted. With --pr (the PR's `gh pr view --json title,headRefName,body`), it fails
unless the body's `## Removed` names each such issue or entry on a line; the PR's own issues (in its title or branch)
need none. Needs full history.
    python3 scripts/removed-features.py BASE HEAD [--pr pr.json]
"""

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import PurePosixPath

ISSUE = re.compile(r"(?<![\w-])(?:cl|xs|cr|se|tfs|th|var|h3a)-\d+[a-z]?(?!\w)")
SKIP = ("vendor/", "target/", "node_modules/")
CELL_ROOTS = ("e2e/", "release-gate/", "checks/")
CODE = (".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".sh")
WHY = "<why: replaced by …, dropped because …, or Paul's ruling>"
VIS = r"(?:pub(?:\([^)]*\))?\s+)?"
RUST = [
    re.compile(rf"^\s*{VIS}(?:(?:default|async|unsafe|const|extern(?:\s+\"[^\"]*\")?)\s+)*(fn)\s+(\w+)"),
    re.compile(rf"^\s*{VIS}(static)\s+(?:mut\s+)?(\w+)\s*:"),
    re.compile(rf"^\s*{VIS}(const)\s+(\w+)\s*:"),
    re.compile(rf"^\s*{VIS}(struct|enum|trait|type|union)\s+(\w+)"),
]
PY = re.compile(r"^((?:    )?)(?:async\s+)?(def|class)\s+(\w+)")
JS = re.compile(r"^export\s+(?:default\s+)?(?:async\s+)?(function|class|const|let)\*?\s+([A-Za-z_$][\w$]*)")
SH = re.compile(r"^(?:function\s+)?([A-Za-z_][\w-]*)\s*\(\)\s*\{")
ROUTE = re.compile(r"\.route\(\s*\"([^\"]+)\"")

# (kind, name) -> (line, label)
Defs = dict[tuple[str, str], tuple[int, str]]


def git(*args: str, stdin: str | None = None) -> str:
    return subprocess.run(["git", *args], input=stdin, capture_output=True, text=True, check=True).stdout


def kebab(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", name).replace("_", "-").lower()


def clap(lines: list[str], defs: Defs) -> None:
    """Flags (`#[arg(long…)] field`) and verbs (a `#[derive(…Subcommand…)]` enum's variants)."""
    attr, open_attr, subcommand, depth, name = "", False, False, 0, ""
    for no, line in enumerate(lines, 1):
        s = line.strip()
        if depth:
            delta = s.count("{") - s.count("}")
            m = re.match(r"#\[command\(.*\bname\s*=\s*\"([^\"]+)\"", s)
            if m:
                name = m[1]
            m = re.match(r"([A-Z]\w*)\s*(?:[({,]|$)", s)
            if m and depth == 1:
                verb = name or kebab(m[1])
                defs[("verb", verb)] = (no, f"verb {verb}")
                name = ""
            depth += delta
            continue
        if s.startswith("#[derive(") and "Subcommand" in s:
            subcommand = True
        elif subcommand and re.match(rf"{VIS}enum\s+\w+", s):
            depth, subcommand = max(s.count("{") - s.count("}"), 1), False
            continue
        if open_attr or s.startswith(("#[arg(", "#[clap(")):
            attr += s
            open_attr = attr.count("(") > attr.count(")") or not attr.endswith("]")
            continue
        if attr and s and not s.startswith(("#[", "//")):
            m = re.match(r"(?:pub(?:\([^)]*\))?\s+)?(\w+)\s*:", s)
            if m and re.search(r"\blong\b", attr):
                long = re.search(r"\blong\s*=\s*\"([^\"]+)\"", attr)
                flag = "--" + (long[1] if long else m[1].replace("_", "-"))
                defs[("flag", flag)] = (no, f"flag {flag}")
            attr = ""


def definitions(path: str, text: str) -> Defs:
    defs: Defs = {}
    lines = text.splitlines()
    if path.endswith(".rs"):
        for no, line in enumerate(lines, 1):
            for pattern in RUST:
                m = pattern.match(line)
                if m:
                    defs.setdefault(("item", m[2]), (no, f"{m[1]} {m[2]}"))
                    break
        if "clap" in text or "#[arg(" in text:
            clap(lines, defs)
        for m in ROUTE.finditer(text):
            defs[("route", m[1])] = (text.count("\n", 0, m.start()) + 1, f"route {m[1]}")
    elif path.endswith(".py"):
        in_class = False
        for no, line in enumerate(lines, 1):
            if line[:1].isalpha():
                in_class = line.startswith("class ")
            m = PY.match(line)
            if m and (not m[1] or in_class):
                defs.setdefault(("item", m[3]), (no, f"{m[2]} {m[3]}"))
    elif path.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs")):
        for no, line in enumerate(lines, 1):
            m = JS.match(line)
            if m:
                defs.setdefault(("item", m[2]), (no, f"export {m[1]} {m[2]}"))
    elif path.endswith(".sh") and path.startswith(CELL_ROOTS):
        for no, line in enumerate(lines, 1):
            m = SH.match(line)
            if m:
                defs.setdefault(("cell", m[1]), (no, f"cell {m[1]}"))
    if path.startswith(CELL_ROOTS) and text.strip():
        name = PurePosixPath(path).name
        defs[("file", name)] = (1, f"cell file {name}")
    return defs


def blobs(rev: str, paths: list[str]) -> dict[str, str]:
    """Each path's text at rev, through one `git cat-file --batch`."""
    if not paths:
        return {}
    out = subprocess.run(["git", "cat-file", "--batch"], input="".join(f"{rev}:{p}\n" for p in paths).encode(),
                         capture_output=True, check=True).stdout
    texts: dict[str, str] = {}
    at = 0
    for path in paths:
        end = out.index(b"\n", at)
        header = out[at:end].split()
        at = end + 1
        if header[-1] == b"missing":
            continue
        size = int(header[2])
        texts[path] = out[at:at + size].decode("utf-8", "replace")
        at += size + 1
    return texts


def needles(kind: str, name: str) -> list[str]:
    if kind == "flag":
        return [name[2:], name[2:].replace("-", "_")]
    if kind == "verb":
        return [name, name.replace("-", "")]
    return [name]


def removed(base: str, head: str) -> list[tuple[str, str, int, str]]:
    """(kind:name, path, line at base, label) for each definition at base that HEAD defines nowhere."""
    changed = [line.split("\t") for line in git("diff", "--no-renames", "--name-status", base, head).splitlines()]
    paths = [p for status, p in changed if status in ("M", "D") and p.endswith(CODE) and not p.startswith(SKIP)]
    before, after = blobs(base, paths), blobs(head, paths)
    gone: dict[tuple[str, str], tuple[str, int, str]] = {}
    for path, text in before.items():
        kept = definitions(path, after.get(path, ""))
        for key, (no, label) in definitions(path, text).items():
            if key not in kept:
                gone.setdefault(key, (path, no, label))
    if not gone:
        return []
    files = set(git("ls-tree", "-r", "--name-only", head).splitlines())
    names = {PurePosixPath(f).name for f in files}
    search = sorted({n for kind, name in gone if kind != "file" for n in needles(kind, name)})
    found: set[tuple[str, str]] = {key for key in gone if key[0] == "file" and key[1] in names}
    if search:
        args = ["grep", "-l", "-F", *[a for n in search for a in ("-e", n)], head, "--"]
        hits = subprocess.run(["git", *args, *[f"*{e}" for e in CODE]], capture_output=True, text=True).stdout
        mentioned = [h.split(":", 1)[1] for h in hits.splitlines() if not h.split(":", 1)[1].startswith(SKIP)]
        for path, text in blobs(head, mentioned).items():
            found |= definitions(path, text).keys()
    return sorted((f"{k}:{n}", p, no, label) for (k, n), (p, no, label) in gone.items() if (k, n) not in found)


def origins(base: str, rows: list[tuple[str, str, int, str]]) -> dict[str, str]:
    """The commit that added each row's definition: `git blame -w` at base, then again from the blamed commit's
    parent while the definition was already there (a later edit of its line is not its adding commit)."""
    found: dict[str, str] = {}
    pending = [(f"{key}@{path}:{no}", key, base, path, no) for key, path, no, _ in rows]
    for _ in range(12):
        by_file: dict[tuple[str, str], list[tuple[str, str, int]]] = defaultdict(list)
        for row, key, rev, path, no in pending:
            by_file[(rev, path)].append((row, key, no))
        blamed: list[tuple[str, str, str, str, str]] = []
        for (rev, path), items in by_file.items():
            ranges = [a for no in sorted({no for _, _, no in items}) for a in ("-L", f"{no},{no}")]
            line_commit: dict[int, tuple[str, str, str]] = {}
            sha, final, prev, prev_path = "", 0, "", ""
            blame = subprocess.run(["git", "blame", "-w", "--line-porcelain", *ranges, rev, "--", path],
                                   capture_output=True, text=True)
            if blame.returncode:
                print(f"removed-features: can't blame {path} at {rev[:9]}: {blame.stderr.strip()}", file=sys.stderr)
            for line in blame.stdout.splitlines():
                if m := re.match(r"^([0-9a-f]{40}) \d+ (\d+)", line):
                    sha, final, prev, prev_path = m[1], int(m[2]), "", ""
                elif line.startswith("previous "):
                    prev, prev_path = line.split(" ", 2)[1:]
                elif line.startswith("\t"):
                    line_commit[final] = (sha, prev, prev_path)
            for row, key, no in items:
                if no not in line_commit:
                    continue
                sha, prev, prev_path = line_commit[no]
                found[row] = sha
                if prev:
                    blamed.append((row, key, prev, prev_path, sha))
        parents: dict[tuple[str, str], Defs] = {}
        for rev, path in {(r, p) for _, _, r, p, _ in blamed}:
            text = blobs(rev, [path]).get(path)
            parents[(rev, path)] = definitions(path, text) if text is not None else {}
        pending = []
        for row, key, rev, path, _ in blamed:
            k, n = key.split(":", 1)
            if (k, n) in parents[(rev, path)]:
                pending.append((row, key, rev, path, parents[(rev, path)][(k, n)][0]))
        if not pending:
            break
    return found


def requests(shas: set[str]) -> dict[str, tuple[list[str], str]]:
    """Per commit: the issues its message names and the changelog.d entries it added, and its subject."""
    if not shas:
        return {}
    info: dict[str, tuple[list[str], str]] = {}
    log = git("log", "--no-walk=unsorted", "--format=%H%x00%s%x00%B%x01", *sorted(shas))
    for record in log.split("\x01"):
        if "\x00" not in record:
            continue
        sha, subject, body = record.strip("\n").split("\x00", 2)
        info[sha] = (list(dict.fromkeys(ISSUE.findall(body))), subject)
    added = git("diff-tree", "--stdin", "-r", "--root", "--name-only", "--diff-filter=A", "--", "changelog.d/",
                stdin="".join(f"{s}\n" for s in sorted(shas)))
    sha = ""
    for line in added.splitlines():
        if re.fullmatch(r"[0-9a-f]{40}", line):
            sha = line
        elif line.startswith("changelog.d/") and sha in info:
            info[sha][0].append(PurePosixPath(line).stem)
    return info


def removed_section(body: str) -> list[str]:
    lines: list[str] = []
    inside = False
    for line in body.splitlines():
        if re.match(r"^#{1,6}\s", line):
            inside = bool(re.match(r"^#{1,6}\s*Removed\b", line, re.I))
        elif inside and line.strip():
            lines.append(line)
    return lines


def main() -> int:
    args = sys.argv[1:]
    pr: dict[str, str] = {}
    if "--pr" in args:
        i = args.index("--pr")
        with open(args[i + 1]) as f:
            pr = {k: str(v) for k, v in json.load(f).items()}
        del args[i:i + 2]
    if len(args) != 2:
        sys.exit(__doc__)
    base, head = [git("rev-parse", "--verify", f"{r}^{{commit}}").strip() for r in args]
    rows = removed(base, head)
    commit = origins(base, rows)
    info = requests(set(commit.values()))
    own = set(ISSUE.findall(f"{pr.get('title', '')} {pr.get('headRefName', '')}"))
    groups: dict[str, list[tuple[str, str, int, str, str]]] = defaultdict(list)
    alias: dict[str, set[str]] = defaultdict(set)
    for key, path, no, label in rows:
        sha = commit.get(f"{key}@{path}:{no}", "")
        keys, subject = info.get(sha, ([], ""))
        if not keys or own & set(keys):
            continue
        groups[keys[0]].append((label, path, no, sha[:9], subject))
        alias[keys[0]] |= set(keys)
    named = removed_section(pr.get("body", ""))
    missing = [k for k in groups
               if not any(re.search(rf"(?<![\w-]){re.escape(a)}(?![\w-])", line) and "<why" not in line
                          for a in alias[k] for line in named)]
    print(f"removed-features: {len(rows)} definitions deleted from the tree, {len(groups)} requests behind them"
          + (f" (own: {', '.join(sorted(own))})" if own else ""))
    for k, items in groups.items():
        print(f"  {k}{' (named)' if k not in missing else ''}")
        for label, path, no, sha, subject in items[:6]:
            print(f"    {label}  {path}:{no}  ({sha} {subject[:70]})")
        if len(items) > 6:
            print(f"    … and {len(items) - 6} more")
    if not pr or not missing:
        return 0
    print("\nThis PR removes definitions that tracker issues or changelog.d entries added. Add this to the PR body,"
          " fill in each reason, and queue it again:\n\n## Removed")
    for k in missing:
        print(f"- {k}: {WHY}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
