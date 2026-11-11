"""Derive a private 30-step coverage fixture from a source model's plan; never a production plan."""
import argparse
import json
from pathlib import Path
import tensorfs

from cozy_runtime.author import canonical_json
from minimax_h3.preparation.plans import parse_plan
from minimax_h3.preparation.source import timestep_plan

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path, help="local migrated source checkpoint")
parser.add_argument("--task", choices=("fl2va", "ref2va"), default="fl2va")
parser.add_argument("--out", type=Path, default=Path.home() / ".cozy/outputs/h3-bank-qualification/coverage-30.json")
args = parser.parse_args()
source = tensorfs.open(args.checkpoint, local_files_only=True)
production = timestep_plan(source, args.task)
document = json.loads(production.canonical_bytes)
document["schedules"] = [schedule for schedule in document["schedules"] if schedule["transformer_evaluations"] == 30]
if len(document["schedules"]) != 1:
    raise SystemExit("the source must contain exactly one reviewed 30-step schedule")
blocks: list[dict[str, object]] = []
final: list[dict[str, object]] = []
seen_blocks, seen_final = set(), set()
for evaluation in document["schedules"][0]["evaluations"]:
    for item in evaluation["modulation_classes"]:
        pair = (item["timestep"], item["modality_tag"])
        if pair not in seen_blocks:
            seen_blocks.add(pair)
            blocks.append({"index": len(blocks), "timestep": pair[0], "modality": item["modality"], "modality_tag": pair[1]})
        if pair[0] not in seen_final:
            seen_final.add(pair[0])
            final.append({"index": len(final), "timestep": pair[0]})
document["table_keys"] = {"block_modulation": blocks, "final_normalization": final}
raw = canonical_json.encode(document)
plan = parse_plan(raw, task=args.task)
if plan.digest == production.digest:
    raise SystemExit("qualification variant unexpectedly equals the source model's plan")
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_bytes(raw)
print(json.dumps({"task": args.task, "digest": plan.digest, "steps": plan.steps, "timesteps": len(plan.timesteps), "block_rows": len(plan.block_rows), "file": str(args.out), "source_plan": production.digest}, sort_keys=True))
