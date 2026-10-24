"""Generate callers only from a real successful ordinary-CLI adoption receipt."""
import argparse
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("adoption", type=Path)
parser.add_argument("--tools-source", type=Path, help="private same-version tools checkout for controlled code/plan qualification; never published")
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
for digest in (artifact["checkpoint"], artifact["transaction"]):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise SystemExit("adoption lacks exact native digest provenance")
body = json.dumps(artifact, sort_keys=True, separators=(",", ":"))
code = '''# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.30.0,<1", "minimax-h3-tools", "msgspec>=0.19"]
# ///
import msgspec
from cozy_runtime.author import App, ModelArtifact
from h3_tables.operations import precompute_adaln

app = App()
SOURCE = msgspec.json.decode(BODY, type=ModelArtifact)

class Nothing(msgspec.Struct):
    pass

@app.entrypoint(accelerator=False)
async def main(payload: Nothing) -> ModelArtifact:
    return await precompute_adaln(model=SOURCE, timesteps=30)
'''.replace("BODY", repr(body))
if args.tools_source is not None:
    selected = args.tools_source.resolve()
    if not (selected / "pyproject.toml").is_file():
        raise SystemExit("tools source must be one real private package directory")
    code = code.replace("# ///\nimport msgspec", "# [tool.uv.sources]\n# minimax-h3-tools = {path = " + json.dumps(str(selected)) + ", editable = true}\n# ///\nimport msgspec")
args.out.mkdir(parents=True, exist_ok=True)
(args.out / "compute.py").write_text(code)
(args.out / "caller_edited.py").write_text(code + "\n# Same library operations, new caller.\n")
(args.out / "schedule40.py").write_text(code.replace("timesteps=30", "timesteps=40"))
negative = code.replace("from h3_tables.operations import precompute_adaln", "from h3_tables.adaln_operations import select_adaln_weights, compute_adaln_tables")
negative = negative.replace("    return await precompute_adaln(model=SOURCE, timesteps=30)", "    selected = await select_adaln_weights(source=SOURCE, task='fl2va')\n    if selected.ready or selected.projection is None:\n        raise ValueError('positive bank test requires generating weights')\n    return await compute_adaln_tables(source=selected.projection, task='fl2va', plan_digest='sha256:' + '0' * 64)")
(args.out / "wrong_plan.py").write_text(negative)
(args.out / "compute_fl.py").write_text(negative.replace("plan_digest='sha256:' + '0' * 64", "plan_digest=selected.plan_digest"))
print(json.dumps({"out": str(args.out), "source": artifact, "schedule40_expectation": "same approved union bank; reuse, not invalidation"}, sort_keys=True))
