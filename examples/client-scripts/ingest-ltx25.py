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
import hashlib
import json
import math
import os
import struct
import sys
import urllib.request
from pathlib import Path
from typing import cast

import numpy as np
import tensorfs

REPO = "Lightricks/LTX-2.5-Diffusers"
REVISION = "4791a1100e0cf97f7ed5c55c12dd71dce13f9dd3"
NATIVE_REVISION = "49b093a39c19169a8a81721dcc541e07e945232f"
NATIVE_TRANSFORMER = f"https://huggingface.co/Lightricks/LTX-2.5/resolve/{NATIVE_REVISION}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
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


def get(url: str, token: str = "", byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {"Authorization": "Bearer " + token} if token else {}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as reply:
        if byte_range is not None:
            if reply.status != 206:
                raise ValueError("source ignored bounded metadata/payload range")
            return cast(bytes, reply.read(byte_range[1] - byte_range[0] + 1))
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
    # The Diffusers export rounded native F32 modulation tables to BF16. Restore
    # the author's exact 19 MB prefix instead of widening those rounded values.
    raw = get(NATIVE_TRANSFORMER, token, (0, (1 << 20) - 1))
    length = struct.unpack("<Q", raw[:8])[0]
    native = json.loads(raw[8:8 + length])
    tables = {}
    for key, info in native.items():
        if key == "__metadata__" or info["dtype"] != "F32":
            continue
        target = key.removeprefix("model.diffusion_model.")
        target = target.replace("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table")
        target = target.replace("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table")
        tables[target] = info
    end = max(info["data_offsets"][1] for info in tables.values())
    if len(tables) != 290 or not set(tables) <= set(source.components["transformer"].tensors):
        raise ValueError("the native modulation table schema differs from the pinned model")
    payload = get(NATIVE_TRANSFORMER, token, (8 + length, 8 + length + end - 1))
    if hashlib.sha256(payload).hexdigest() != "ffa9d298ef556dc2668826f1af0eea723196ee418570f9a5d18092763ac14303":
        raise ValueError("the pinned native F32 table prefix did not match its verified digest")
    changes = {key: renamed(key) for key in source.components["text_encoder"].tensors if key != renamed(key)}
    with tensorfs.Writer(model_id="local/" + args.local_name, lane="bf16", cache_dir=args.cache_dir) as writer:
        writer.metadata = dict(source.metadata)
        writer.metadata["cozy_native_precision"] = {
            "source": NATIVE_TRANSFORMER, "tables": len(tables),
            "table_sha256": hashlib.sha256(payload).hexdigest(), "audio_dtype": "f32",
        }
        if source.pipeline is not None:
            writer.pipeline(source.pipeline.cls, config=source.pipeline.config)
        for name, part in source.components.items():
            output = writer.component(name, cls=part.cls, config=part.config, config_file=part.config_file)
            for key, spec in part.tensors.items():
                if name == "transformer" and key in tables:
                    info = tables[key]
                    a, b = info["data_offsets"]
                    output.put(key, payload[a:b], dtype="f32", shape=info["shape"])
                elif name in ("audio_vae", "vocoder"):
                    if spec.dtype.lower() != "bf16":
                        raise ValueError(f"the pinned {name}'s {key} is {spec.dtype}, expected BF16")
                    bf16_data = bytearray(2 * math.prod(spec.shape))
                    part.tensors.read_into(key, bf16_data)
                    # BF16 is the upper sixteen bits of IEEE F32: widening changes no value.
                    words = np.frombuffer(bf16_data, dtype="<u2").astype("<u4") << 16
                    value = words.view("<f4").reshape(spec.shape)
                    output.put(key, value.tobytes(), dtype="f32", shape=spec.shape)
                else:
                    output.copy(key, part, as_key=renamed(key) if name == "text_encoder" else key)
            for asset in part.assets:
                output.asset(asset, part.asset(asset))
            if name == "text_encoder":
                output.asset("cozy-import-normalization.json", json.dumps({
                    "source": f"{REPO}@{REVISION}", "source_header": source.id,
                    "tensor_key_renames": changes, "tokenizer_model_max_length": 1024,
                    "audio_vae_dtype": "f32", "vocoder_dtype": "f32",
                    "native_f32_tables": {"revision": NATIVE_REVISION, "count": len(tables), "sha256": hashlib.sha256(payload).hexdigest()},
                    "other_numerical_values": "unchanged",
                }, indent=2).encode())
            if name == "model":
                output.asset("LICENSE", (metadata / "LICENSE").read_bytes())
        checkpoint = writer.commit()
    print(json.dumps({"local_model": "local/" + args.local_name, "checkpoint": checkpoint.id}))


if __name__ == "__main__":
    main()
