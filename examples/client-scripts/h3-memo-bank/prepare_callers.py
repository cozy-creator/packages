"""Generate callers from a real adoption receipt and the selected H3 source/interface."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sdk(*arguments: str) -> bytes:
    return subprocess.check_output([sys.executable, "-m", "cozy_runtime.cli.main", *arguments])


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("adoption", type=Path)
parser.add_argument("--tools-source", type=Path, required=True, help="exact H3 checkout used by the callers")
parser.add_argument("--interface", type=Path, required=True, help="canonical cozy-runtime describe --json for that checkout")
parser.add_argument("--out", type=Path, default=Path.home() / ".cozy/outputs/h3-bank-qualification")
args = parser.parse_args()
reply = json.loads(args.adoption.read_text())
if reply.get("status") != "completed":
    raise SystemExit("adoption must be an actual completed ordinary CLI run")
artifact = reply["result"]
if set(artifact) != {"producer_request_id", "output_slot", "checkpoint", "transaction"}:
    raise SystemExit("adoption did not return one exact ModelArtifact")
if not artifact["producer_request_id"] or artifact["output_slot"] != "original":
    raise SystemExit("adoption has no genuine producer/output provenance")
for value in (artifact["checkpoint"], artifact["transaction"]):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise SystemExit("adoption lacks exact native digest provenance")

selected = args.tools_source.resolve()
project = tomllib.loads((selected / "pyproject.toml").read_text())["project"]
if project["name"] != "minimax-h3":
    raise SystemExit("tools source must be the minimax-h3 package")
interface = args.interface.read_bytes()
described = sdk("--dir", str(selected), "--json", "describe")
if json.loads(interface) != json.loads(described):
    raise SystemExit("interface does not match the selected H3 source; describe it again")

helper_path = Path("minimax_h3/preparation/operations.py")
operations = (selected / helper_path).read_text()
helper = next(
    node for node in ast.parse(operations).body
    if isinstance(node, ast.AsyncFunctionDef) and node.name == "precompute_adaln"
)
orchestration = ast.get_source_segment(operations, helper)
assert orchestration is not None
source_files = subprocess.check_output([
    "git", "-C", str(selected), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", ".",
]).decode().split("\0")
source_hashes = {name: digest((selected / name).read_bytes()) for name in sorted(set(source_files)) if name}
source_identity = digest(json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode())
source_commit = subprocess.check_output(["git", "-C", str(selected), "rev-parse", "HEAD"], text=True).strip()

# Keep the source-derived orchestration intact; only the imported targets change to Operations.
code = '''import msgspec
from cozy_runtime.author import App, ModelArtifact, ModelChoice, UnsupportedInput, call
from ._h3 import (
    AdalnInput, ComputeAdalnInput, SelectAdalnInput,
    apply_adaln, compute_adaln_tables, retable_adaln, select_adaln_weights,
)

app = App()
SOURCE = msgspec.json.decode(BODY, type=ModelArtifact)

HELPER

class Nothing(msgspec.Struct):
    pass

@app.entrypoint(accelerator=False)
async def main(payload: Nothing) -> ModelArtifact:
    return await precompute_adaln(model=SOURCE, timesteps=30)
'''.replace("BODY", repr(json.dumps(artifact, sort_keys=True, separators=(",", ":")))).replace("HELPER", orchestration)
negative = code.replace(
    "    return await precompute_adaln(model=SOURCE, timesteps=30)",
    "    selected = await call(select_adaln_weights, SelectAdalnInput(task='fl2va'), models={'source': ModelChoice(ref=SOURCE)})\n"
    "    if selected.ready or selected.projection is None:\n"
    "        raise ValueError('positive bank test requires generating weights')\n"
    "    return await call(compute_adaln_tables, ComputeAdalnInput(task='fl2va', plan_digest='sha256:' + '0' * 64), models={'source': ModelChoice(ref=selected.projection)})",
)
callers = {
    "compute": code,
    "caller_edited": code + "\n# Same library operations, new caller.\n",
    "schedule40": code.replace("model=SOURCE, timesteps=30", "model=SOURCE, timesteps=40"),
    "wrong_plan": negative,
    "compute_fl": negative.replace("plan_digest='sha256:' + '0' * 64", "plan_digest=selected.plan_digest"),
}
args.out.mkdir(parents=True, exist_ok=False)
canonical = args.out / "h3-interface.json"
canonical.write_bytes(interface)
generated = args.out / "h3_calls.py"
sdk("--package-interface", str(canonical), "calls", "minimax-h3", str(generated))
references = generated.read_bytes()
for name, caller in callers.items():
    root = args.out / name
    module = root / "bank_caller"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text(caller)
    (module / "_h3.py").write_bytes(references)
    (root / "pyproject.toml").write_text(f'''[project]
name = "h3-bank-{name.replace('_', '-')}"
version = "0.0.1"
requires-python = ">=3.12,<3.13"
dependencies = ["cozy-runtime>=0.48.0.dev0,<0.49", "msgspec>=0.19"]

[project.optional-dependencies]
cozy-calls = ["minimax-h3[cozy-calls]=={project['version']}"]

[project.entry-points."cozy.application"]
default = "bank_caller:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["bank_caller"]

[tool.uv.sources]
minimax-h3 = {{ path = {json.dumps(str(selected))}, editable = true }}
''')
provenance = {
    "source": str(selected),
    "source_commit": source_commit,
    "source_digest": source_identity,
    "source_files": source_hashes,
    "version": project["version"],
    "helper": str(helper_path),
    "helper_digest": digest(orchestration.encode()),
    "interface_digest": digest(interface),
    "references_digest": digest(references),
    "adoption_digest": digest(args.adoption.read_bytes()),
    "generator_digest": digest(Path(__file__).read_bytes()),
    "sdk": json.loads(sdk("--json", "version")),
    "callers": {name: digest(caller.encode()) for name, caller in callers.items()},
}
(args.out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
print(json.dumps({"out": str(args.out), "source": artifact, "schedule40_expectation": "same approved union bank; reuse, not invalidation"}, sort_keys=True))
