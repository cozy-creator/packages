"""Capture current H3 with an exact one-, four- or eight-GPU model ladder.

Only deployment metadata changes: the H3 callable and sampling math stay intact.
Run `cozy package lock` and `cozy run ... --describe` on each emitted directory.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--account-index", required=True,
                        help="The account_index returned by cozy package lock for the selected Hub")
    parser.add_argument("--degrees", type=int, nargs="+", choices=(1, 4, 8), default=(1, 4, 8),
                        help="Only generate these exact widths; existing captures are never replaced")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / "minimax-h3"
    text = (source / "h3.py").read_text()
    syntax = ast.parse(text)
    targets = {
        "_DEFAULT_MODEL_LADDER": "minimax-h3@1.0.0-rc.3/fp8-pruned",
        "_DEFAULT_TURBO_LORA_LADDER": "minimax-h3-turbo-lora@1.0.0-audit.1/pdd8",
    }
    nodes = [node for node in syntax.body if isinstance(node, ast.AnnAssign)
             and isinstance(node.target, ast.Name) and node.target.id in targets]
    assert len(nodes) == len(targets)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if len(set(args.degrees)) != len(args.degrees):
        parser.error("degrees must be unique")
    for degree in args.degrees:
        dest = args.output / f"package-degree-{degree}"
        if dest.exists():
            raise FileExistsError(dest)
        shutil.copytree(source, dest)
        lines = text.splitlines(keepends=True)
        for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
            name = node.target.id
            rows = [{"gpu": gpu, "gpus": degree, "lane": targets[name]} for gpu in ("H100", "5090")]
            replacement = f"{name}: list[dict[str, str | int]] = " + json.dumps(rows, indent=4) + "\n"
            lines[node.lineno - 1:node.end_lineno] = [replacement]
        (dest / "h3.py").write_text("".join(lines))
        project = (dest / "pyproject.toml").read_text()
        project = project.replace('name = "minimax-h3"', f'name = "h3-scaling-degree-{degree}"', 1)
        project = project.replace("cozy-runtime[media,minimax-h3]>=0.21.8", "cozy-runtime[media,minimax-h3]==0.22.0")
        assert "cozy-runtime[media,minimax-h3]==0.22.0" in project
        project += ("\n[[tool.uv.index]]\nname = \"tensorhub\"\nurl = "
                    + json.dumps(args.account_index) + "\nexplicit = true\n")
        (dest / "pyproject.toml").write_text(project)
        # The CLI regenerates this interface from the changed source. Retaining the old
        # generated ladder would misrepresent degree8 as a degree4 capture.
        (dest / "metadata" / "package-interface.json").unlink(missing_ok=True)
        proof = {"source_sha": source_sha, "source": str(source), "degree": degree,
                 "generator_sha256": sha(Path(__file__)),
                 "base_lane": targets["_DEFAULT_MODEL_LADDER"],
                 "turbo_lora_lane": targets["_DEFAULT_TURBO_LORA_LADDER"],
                 "source_files": {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()},
                 "captured_files": {str(p.relative_to(dest)): sha(p) for p in dest.rglob("*") if p.is_file()}}
        (args.output / f"package-degree-{degree}-source.json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    main()
