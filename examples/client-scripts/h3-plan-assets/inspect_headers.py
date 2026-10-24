# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.30,<0.31", "tensorfs>=0.13,<0.14", "msgspec>=0.19"]
# ///
"""Read a granted checkpoint's header and plan references without reading tensor data."""

import json
from typing import Any

import msgspec

from cozy_runtime.author import App, Context, Weights

app = App()


class Nothing(msgspec.Struct):
    pass


class Header(msgspec.Struct):
    header: str
    tree: str
    variant: str
    plans: str
    tensor_chunk_count: int
    missing_tensor_chunk_count: int


@app.entrypoint(accelerator=False)
def main(ctx: Context, payload: Nothing, source: Weights[Any]) -> Header:
    checkpoint = ctx.tensorfs_source(source)
    plans = {
        name: component.assets["timestep-plan.json"]
        for name, component in checkpoint.components.items()
        if "timestep-plan.json" in component.assets
    }
    missing = set(checkpoint.missing())
    return Header(
        header=checkpoint.id,
        tree=checkpoint.tree or "",
        variant=checkpoint.variant or "",
        plans=json.dumps(plans, sort_keys=True),
        tensor_chunk_count=len(checkpoint.chunks),
        missing_tensor_chunk_count=sum(chunk in missing for chunk in checkpoint.chunks),
    )
