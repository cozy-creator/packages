"""Source-derived geometry only; this does not execute a model or prove memo reuse."""
import argparse
import json
from pathlib import Path
import tensorfs
import math
import torch
from minimax_h3.preparation.adaln_operations import _plan, _topology
from minimax_h3.preparation.kernel import source_shapes, table_shapes

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path, help="local migrated full model checkpoint")
args = parser.parse_args()
checkpoint = tensorfs.open(args.checkpoint, local_files_only=True)

rows = []
for task in ("fl2va", "ref2va"):
    plan, topology = _plan(checkpoint, task), _topology(task)
    source = {key: math.prod(shape) * torch.empty((), dtype=dtype).element_size()
              for key, (dtype, shape) in source_shapes(topology).items()}
    tables = {key: math.prod(shape) * 2 for key, shape in table_shapes(topology, plan).items()}
    rows.append({"task": task, "plan_digest": plan.digest, "supported_steps": plan.steps,
                 "timesteps": len(plan.timesteps), "block_rows": len(plan.block_rows),
                 "source_tensors": len(source), "source_bytes": sum(source.values()),
                 "largest_source_tensor_bytes": max(source.values()),
                 "table_tensors": len(tables), "table_bytes": sum(tables.values()),
                 "checkpoint_unit_max_bytes": max(tables.values()),
                 "time_embedding_read_bytes": sum(value for key, value in source.items() if key.startswith("time_embedder.")),
                 "first_block_read_bytes": sum(value for key, value in source.items() if key.startswith("transformer_blocks.0."))})
print(json.dumps(rows, indent=2))
