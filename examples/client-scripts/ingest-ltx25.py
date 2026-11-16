# /// script
# requires-python = ">=3.12"
# dependencies = ["tensorfs>=0.25.1,<0.26", "numpy>=1.26"]
# ///
"""Prepare the pinned LTX-2.5 Distilled CozyTensors checkpoint on a CPU machine.

HF_TOKEN supplies source access. This writes a local alias only; publish separately with
`cozy model upload local/<name> lightricks/ltx-2.5 --lane bf16 --on <machine>`.
Use that machine's TensorFS store as --cache-dir so the alias has the right custody.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from pathlib import Path
from typing import cast

import numpy as np
import tensorfs

REPO = "Lightricks/LTX-2.5-Diffusers"
REVISION = "4791a1100e0cf97f7ed5c55c12dd71dce13f9dd3"
LICENSE = "https://raw.githubusercontent.com/Lightricks/LTX-2/f0f2f8607189c78a72420b2b63cb1975b5eca179/LICENSE-2_x"
COMPONENTS = ("transformer", "text_encoder", "connectors", "vae", "audio_vae", "vocoder", "latent_upsampler")
WEIGHTS = (
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "connectors/diffusion_pytorch_model.safetensors.index.json",
    "text_encoder/model.safetensors.index.json",
    "vae/diffusion_pytorch_model.safetensors",
    "audio_vae/diffusion_pytorch_model.safetensors",
    "vocoder/diffusion_pytorch_model.safetensors",
    "latent_upsampler/diffusion_pytorch_model.safetensors",
)


def get(url: str, token: str = "") -> bytes:
    headers = {"Authorization": "Bearer " + token} if token else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as reply:
        return cast(bytes, reply.read())


def write(root: Path, name: str, data: bytes) -> None:
    dest = root / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.read_bytes() != data:
            raise ValueError(f"{dest} differs; preserve it and choose another --work-dir")
        return
    with dest.open("xb") as output:
        output.write(data)


def renamed(key: str) -> str:
    if key.startswith("model.vision_embedder."):
        return key.replace("model.vision_embedder.", "model.embed_vision.", 1)
    return key.replace("model.embed_vision.embedding_projection.", "model.embed_vision.multimodal_embedder.embedding_projection.", 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--local-name", default="ltx25-distilled-ready")
    parser.add_argument("--token-stdin", action="store_true", help="Read HF token from stdin instead of HF_TOKEN")
    args = parser.parse_args()
    token = sys.stdin.readline().strip() if args.token_stdin else os.environ["HF_TOKEN"]
    base = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
    metadata = args.work_dir / "metadata"
    for component in (*COMPONENTS, "scheduler"):
        filename = "scheduler_config.json" if component == "scheduler" else "config.json"
        data = json.loads(get(base + component + "/" + filename, token))
        if component == "text_encoder":
            original = data.pop("ltx_source_checkpoint", None)
            if original is not None:
                write(metadata, component + "/upstream-training-provenance.json", json.dumps(original, indent=2).encode())
        write(metadata, component + "/" + filename, json.dumps(data, indent=2).encode())
    for filename in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        original = get(base + "tokenizer/" + filename, token)
        if filename == "tokenizer_config.json":
            write(metadata, "tokenizer/upstream-tokenizer-config.json", original)
            data = json.loads(original)
            data["model_max_length"] = 1024
            original = json.dumps(data, indent=2).encode()
        write(metadata, "tokenizer/" + filename, original)
    index = json.loads(get(base + "model_index.json", token))
    index = {key: value for key, value in index.items() if key.startswith("_") or key in (*COMPONENTS, "scheduler", "tokenizer")}
    index["latent_upsampler"] = ["diffusers", "LTX2LatentUpsamplerModel"]
    write(metadata, "model_index.json", json.dumps(index, indent=2).encode())
    write(metadata, "LICENSE", get(LICENSE))
    source = tensorfs.convert_to_cozytensors(
        [str(metadata), *(f"hf://{REPO}@{REVISION}/{name}" for name in WEIGHTS)],
        cache_dir=args.cache_dir, token=token,
    )
    changes = {key: renamed(key) for key in source.components["text_encoder"].tensors if key != renamed(key)}
    with tensorfs.Writer(model_id="local/" + args.local_name, lane="bf16", cache_dir=args.cache_dir) as writer:
        writer.metadata = dict(source.metadata)
        writer.metadata["cozy_inference_precision"] = {
            "transformer": "BF16 as published in the pinned Diffusers export",
            "audio_vae": "F32", "vocoder": "F32",
        }
        if source.pipeline is not None:
            writer.pipeline(source.pipeline.cls, config=source.pipeline.config)
        for name, part in source.components.items():
            output = writer.component(name, cls=part.cls, config=part.config, config_file=part.config_file)
            for key, spec in part.tensors.items():
                if name in ("audio_vae", "vocoder"):
                    if spec.dtype.lower() != "bf16":
                        raise ValueError(f"the pinned {name}'s {key} is {spec.dtype}, expected BF16")
                    bf16_data = bytearray(2 * math.prod(spec.shape))
                    part.tensors.read_into(key, bf16_data)
                    # BF16 is the upper sixteen bits of IEEE F32: widening changes no value.
                    words = np.frombuffer(bf16_data, dtype="<u2").astype("<u4") << 16
                    value = words.view("<f4").reshape(spec.shape)
                    output.put(key, value.tobytes(), dtype="F32", shape=spec.shape)
                else:
                    output.copy(key, part, as_key=renamed(key) if name == "text_encoder" else key)
            for asset in part.assets:
                output.asset(asset, part.asset(asset))
            if name == "text_encoder":
                output.asset("cozy-import-normalization.json", json.dumps({
                    "source": f"{REPO}@{REVISION}", "source_header": source.id,
                    "tensor_key_renames": changes, "tokenizer_model_max_length": 1024,
                    "audio_vae_dtype": "f32", "vocoder_dtype": "f32",
                    "other_numerical_values": "unchanged",
                }, indent=2).encode())
        writer.component("model").asset("LICENSE", (metadata / "LICENSE").read_bytes())
        checkpoint = writer.commit()
    print(json.dumps({"local_model": "local/" + args.local_name, "checkpoint": checkpoint.id}))


if __name__ == "__main__":
    main()
