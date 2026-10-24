# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.30.0,<1", "tensorfs>=0.13", "msgspec>=0.19"]
# ///
"""Return genuine native custody before a separate caller delegates H3 bank work."""
from typing import Any

import msgspec

from cozy_runtime.author import App, Context, ModelArtifact, Weights, WeightsOutput
from cozy_runtime.weights_store import Derivation

app = App()


class Nothing(msgspec.Struct):
    pass


@app.entrypoint(accelerator=False, weights=(WeightsOutput("original", max_new_bytes=1),))
def main(ctx: Context, payload: Nothing, source: Weights[Any]) -> ModelArtifact:
    opened = ctx.tensorfs_source(source)
    tensors = {name: component.tensors for name, component in opened.components.items()}
    for component in ("transformer", "transformer_ref"):
        if "transformer_blocks.0.adaln_proj.linear.weight" not in tensors[component]:
            raise ValueError("positive bank qualification requires original generating weights")
        if "transformer_blocks.0.adaln_proj.table" in tensors[component]:
            raise ValueError("an already pruned model is not a positive bank source")
    order = [(name, key) for name, rows in tensors.items() for key in rows]
    with ctx.output("original", Derivation(base=opened, order=order)) as out:
        return ctx.adopt_model(out)
