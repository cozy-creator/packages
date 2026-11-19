"""Generate caller packages from a real adoption receipt and the selected H3 checkout."""

import argparse
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
parser.add_argument("--tools-source", type=Path, required=True, help="exact H3 checkout the callers depend on")
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

# The bank: two selections, two table computations and one attach, each a managed H3 operation.
code = '''import msgspec
from cozy_runtime.author import App, ModelArtifact, ModelChoice, UnsupportedInput, call

from ._h3 import (
    apply_adaln, apply_adalnRequest, compute_adaln_tables, compute_adaln_tablesRequest, select_adaln_weights,
    select_adaln_weightsRequest,
)

app = App()
SOURCE = msgspec.json.decode(BODY, type=ModelArtifact)


class Nothing(msgspec.Struct):
    pass


async def bank(model: ModelArtifact, timesteps: int) -> ModelArtifact:
    """`model` with its AdaLN generators replaced by the shared tables of the plan both tasks serve."""
    source = {"source": ModelChoice(ref=model)}
    fl = await call(select_adaln_weights, select_adaln_weightsRequest(task="fl2va"), models=source)
    ref = await call(select_adaln_weights, select_adaln_weightsRequest(task="ref2va"), models=source)
    if timesteps not in fl.steps or timesteps not in ref.steps:
        raise UnsupportedInput("H3 timestep count is not supported by both source model plans", code="adaln_plan")
    if fl.ready and ref.ready:
        return model
    if fl.projection is None or ref.projection is None:
        raise UnsupportedInput("H3 source lacks both tasks' generating weights", code="adaln_generating_weights")
    tables = {}
    for task, selection in (("fl2va", fl), ("ref2va", ref)):
        tables[task] = await call(
            compute_adaln_tables,
            compute_adaln_tablesRequest(task=task, plan_digest=selection.plan_digest),
            models={"source": ModelChoice(ref=selection.projection)},
        )
    attached = {"source": ModelChoice(ref=model)} | {task: ModelChoice(ref=table) for task, table in tables.items()}
    return await call(apply_adaln, apply_adalnRequest(), models=attached)


@app.entrypoint
async def main(payload: Nothing) -> ModelArtifact:
    return await bank(SOURCE, 30)
'''.replace("BODY", repr(json.dumps(artifact, sort_keys=True, separators=(",", ":"))))
negative = code.replace(
    "    return await bank(SOURCE, 30)",
    "    selected = await call(select_adaln_weights, select_adaln_weightsRequest(task='fl2va'), models={'source': ModelChoice(ref=SOURCE)})\n"
    "    if selected.ready or selected.projection is None:\n"
    "        raise ValueError('positive bank test requires generating weights')\n"
    "    return await call(compute_adaln_tables, compute_adaln_tablesRequest(task='fl2va', plan_digest='sha256:' + '0' * 64), "
    "models={'source': ModelChoice(ref=selected.projection)})",
)
callers = {
    "compute": code,
    "caller_edited": code + "\n# Same library operations, new caller.\n",
    "schedule40": code.replace("bank(SOURCE, 30)", "bank(SOURCE, 40)"),
    "wrong_plan": negative,
    "compute_fl": negative.replace("plan_digest='sha256:' + '0' * 64", "plan_digest=selected.plan_digest"),
}
args.out.mkdir(parents=True, exist_ok=False)
interface = sdk("--dir", str(selected), "--json", "describe")
(args.out / "h3-interface.json").write_bytes(interface)
generated = args.out / "h3_calls.py"
sdk("--package-interface", str(args.out / "h3-interface.json"), "calls", "minimax-h3", str(generated))
references = generated.read_bytes()
for name, caller in callers.items():
    module = args.out / name / "bank_caller"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text(caller)
    (module / "_h3.py").write_bytes(references)
    (args.out / name / "pyproject.toml").write_text(f'''[project]
name = "h3-bank-{name.replace('_', '-')}"
version = "0.0.1"
requires-python = ">=3.12,<3.13"
dependencies = ["cozy-runtime>=0.48.0.dev0,<0.50", "minimax-h3=={project['version']}", "msgspec>=0.19"]

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
files = subprocess.check_output(["git", "-C", str(selected), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "."])
hashes = {name: digest((selected / name).read_bytes()) for name in sorted(set(files.decode().split("\0"))) if name}
provenance = {
    "source": str(selected),
    "source_commit": subprocess.check_output(["git", "-C", str(selected), "rev-parse", "HEAD"], text=True).strip(),
    "source_digest": digest(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()),
    "version": project["version"],
    "interface_digest": digest(interface),
    "references_digest": digest(references),
    "adoption_digest": digest(args.adoption.read_bytes()),
    "generator_digest": digest(Path(__file__).read_bytes()),
    "sdk": json.loads(sdk("--json", "version")),
    "callers": {name: digest(caller.encode()) for name, caller in callers.items()},
}
(args.out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
print(json.dumps({"out": str(args.out), "source": artifact, "schedule40_expectation": "same approved union bank; reuse, not invalidation"}, sort_keys=True))
