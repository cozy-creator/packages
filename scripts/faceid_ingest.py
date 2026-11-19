"""SDXL's FaceID-PlusV2 weights, converted once into the layout `sdxl` reads (se-175). TensorFS reads neither source:
insightface ships buffalo_l as ONNX graphs and h94 ships FaceID-PlusV2 as a pickle, which it refuses.

    uv run --project sdxl --with torch --with onnx --with onnxruntime scripts/faceid_ingest.py buffalo <dir>
    cozy model upload <dir> deepinsight/buffalo-l --lane f32 --source-format as-is --mirror <page>
    uv run --project sdxl --with torch scripts/faceid_ingest.py faceid <dir>
    cozy model upload <dir> h94/ip-adapter-faceid-plusv2-sdxl --lane f16 --source-format as-is --mirror <page>

- `buffalo`: `det_10g.onnx` and `w600k_r50.onnx` from insightface's v0.7 release, each graph's initializers in its
  convolutions' order onto `sdxl.faces`' port (`SCRFD`, `IResNet`): the folded convolutions as they are, the other batch
  norms as their scale and shift, PReLU slopes flat. The port must give onnxruntime's outputs on random inputs.
- `faceid`: `ip-adapter-faceid-plusv2_sdxl.bin` through Diffusers' own loader conversion
  (`_convert_ip_adapter_image_proj_to_diffusers`), as TensorFS's `ipadapter.sdxl` lays out Plus: `image_proj` is
  Diffusers' `IPAdapterFaceIDPlusImageProjection`, its arguments as config; `ip_adapter` each cross-attention's
  `N.to_k_ip`/`N.to_v_ip`. Its LoRA is h94's own safetensors, which TensorFS's `sdxl.lora` converts
  (`cozy model upload hf://h94/IP-Adapter-FaceID@<rev>/ip-adapter-faceid-plusv2_sdxl_lora.safetensors`); the .bin's
  copy of it (`N.to_*_lora`) is checked identical there and left out here.

Each source is pinned by its digest. Both are for non-commercial research use only (insightface's models; h94's card).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.torch import save_file
from torch import nn

from sdxl.faces import SCRFD, IResNet

INSIGHTFACE = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
BUFFALO = {
    "buffalo_l.zip": "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}
H94 = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/43907e6f44d079bf1a9102d9a6e56aef7a219bae/"
FACEID = ("ip-adapter-faceid-plusv2_sdxl.bin", "c6945d82b543700cc3ccbb98d363b837e9c596281607857c74b713a876daf5fb")
LICENSE = ("For non-commercial research use only: insightface's pretrained models are, and h94 releases IP-Adapter-FaceID "
           "'exclusively for research purposes and not intended for commercial use'.")


def fetched(url: str, name: str, digest: str, given: Path | None) -> bytes:
    data = given.read_bytes() if given else urllib.request.urlopen(url).read()  # noqa: S310  # a pinned https source
    found = hashlib.sha256(data).hexdigest()
    if found != digest:
        raise SystemExit(f"{name}: sha256 {found}, not the pinned {digest}")
    return data


def written(folder: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str], name: str = "model") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(folder / f"{name}.safetensors"), metadata)


# ---------------------------------------------------------------- buffalo_l


def _convolutions(net: nn.Module) -> Iterator[str]:
    """The port's convolutions in the order its ONNX graph runs them."""
    if isinstance(net, SCRFD):
        yield from (f"stem.{i}" for i in range(3))
        for s, stage in enumerate(net.stages):
            for b, block in enumerate(stage):
                yield from (f"stages.{s}.{b}.conv1", f"stages.{s}.{b}.conv2")
                if block.downsample is not None:
                    yield f"stages.{s}.{b}.downsample.1"
        yield from (f"lateral.{i}" for i in range(3))
        yield from (f"fpn.{i}" for i in range(3))
        yield from ("down.0", "down.1", "pafpn.0", "pafpn.1")
        for h in range(len(net.heads)):
            yield from (f"heads.{h}.convs.{i}" for i in range(3))
            yield from (f"heads.{h}.cls", f"heads.{h}.reg", f"heads.{h}.kps")
        return
    assert isinstance(net, IResNet)
    yield "conv1"
    for s, layer in enumerate(net.layers):
        for b, block in enumerate(layer):
            yield from (f"layers.{s}.{b}.conv1", f"layers.{s}.{b}.conv2")
            if block.downsample is not None:
                yield f"layers.{s}.{b}.downsample"


def _ported(graph: bytes, net: nn.Module) -> dict[str, torch.Tensor]:
    """`graph`'s initializers under `net`'s names, by node: convolutions, batch norms (scale and shift), PReLUs, the
    linear and SCRFD's box scales, each in graph order."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load_model_from_string(graph)
    values: dict[str, NDArray[np.float32]] = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    convolutions = iter(_convolutions(net))
    norms = iter(n for n, m in net.named_modules() if type(m).__name__ == "_Affine")
    prelus = iter(n for n, m in net.named_modules() if isinstance(m, nn.PReLU))
    scales = iter(f"heads.{h}.scale" for h in range(len(getattr(net, "heads", ()))))
    out: dict[str, NDArray[np.float32]] = {}
    for node in model.graph.node:
        inputs = [values[i] for i in node.input if i in values]
        if node.op_type == "Conv":
            name = next(convolutions)
            out[f"{name}.weight"], out[f"{name}.bias"] = inputs
        elif node.op_type == "BatchNormalization":
            weight, bias, mean, variance = inputs
            epsilon = next(a.f for a in node.attribute if a.name == "epsilon")
            scale = weight / np.sqrt(variance + epsilon)
            name = next(norms)
            out[f"{name}.weight"], out[f"{name}.bias"] = scale, bias - mean * scale
        elif node.op_type == "PRelu":
            out[f"{next(prelus)}.weight"] = inputs[0].reshape(-1)
        elif node.op_type == "Gemm":
            out["fc.weight"], out["fc.bias"] = inputs
        elif node.op_type == "Mul" and inputs:
            out[f"{next(scales)}"] = inputs[0].reshape(())
    for left in (convolutions, norms, prelus, scales):
        if (extra := next(left, None)) is not None:
            raise SystemExit(f"{type(net).__name__}: the graph fills no {extra}")
    state = net.state_dict()
    if set(out) != set(state):
        raise SystemExit(f"{type(net).__name__}: {sorted(set(state) ^ set(out))[:4]} unmatched")
    for key, array in out.items():
        if tuple(array.shape) != tuple(state[key].shape):
            raise SystemExit(f"{key}: {array.shape} from the graph, {tuple(state[key].shape)} in the port")
    return {key: torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)) for key, array in out.items()}


def _agrees(graph: bytes, net: nn.Module, tensors: dict[str, torch.Tensor], size: int) -> float:
    """The port's largest difference from onnxruntime's outputs, relative to their scale, on a random input."""
    import onnxruntime

    net.load_state_dict(tensors)
    x = torch.from_numpy(np.random.default_rng(0).uniform(-1, 1, (1, 3, size, size)).astype(np.float32))
    session = onnxruntime.InferenceSession(graph, providers=["CPUExecutionProvider"])
    theirs = cast("list[NDArray[np.float32]]", session.run(None, {session.get_inputs()[0].name: x.numpy()}))
    with torch.no_grad():
        result = net.eval()(x)
    if isinstance(net, SCRFD):
        levels = cast("list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]", result)
        ours = [t.permute(0, 2, 3, 1).reshape(-1, t.shape[1] // 2) for part in zip(*levels, strict=True) for t in part]
    else:
        ours = [cast("torch.Tensor", result)]
    worst = 0.0
    for mine, reference in zip(ours, theirs, strict=True):
        diff = np.abs(mine.numpy().reshape(reference.shape) - reference).max() / max(np.abs(reference).max(), 1e-6)
        worst = max(worst, float(diff))
    return worst


def buffalo(out: Path, given: Path | None) -> None:
    pack = zipfile.ZipFile(io.BytesIO(fetched(INSIGHTFACE, "buffalo_l.zip", BUFFALO["buffalo_l.zip"], given)))
    readme = [f"insightface's buffalo_l ({INSIGHTFACE}, sha256 {BUFFALO['buffalo_l.zip']}), as `sdxl.faces` ports it.",
              "", LICENSE, ""]  # fmt: skip
    for component, file, net, size in (("detection", "det_10g.onnx", SCRFD(), 640),
                                       ("recognition", "w600k_r50.onnx", IResNet(), 112)):  # fmt: skip
        graph = pack.read(file)
        if (digest := hashlib.sha256(graph).hexdigest()) != BUFFALO[file]:
            raise SystemExit(f"{file}: sha256 {digest}, not the pinned {BUFFALO[file]}")
        tensors = _ported(graph, net)
        worst = _agrees(graph, net, tensors, size)
        print(f"{component}: {file}, {len(tensors)} tensors, largest relative difference from onnxruntime {worst:.2e}")
        if worst > 1e-4:
            raise SystemExit(f"{component}: the port differs from {file} by {worst:.2e}")
        written(out / component, tensors, {"source": f"{INSIGHTFACE}#{file}", "sha256": digest})
        readme.append(f"- `{component}`: `{file}` (sha256 {digest}), {len(tensors)} tensors.")
    (out / "README.md").write_text("\n".join(readme) + "\n")


# ---------------------------------------------------------------- FaceID-PlusV2


def faceid(out: Path, given: Path | None) -> None:
    from diffusers import UNet2DConditionModel

    name, digest = FACEID
    state = torch.load(io.BytesIO(fetched(H94 + name, name, digest, given)), map_location="cpu", weights_only=True)
    with torch.device("meta"):
        unet = UNet2DConditionModel()  # its converter reads only the file
    image_proj: dict[str, torch.Tensor] = state["image_proj"]
    projection: nn.Module = unet._convert_ip_adapter_image_proj_to_diffusers(image_proj, low_cpu_mem_usage=False)
    # The arguments Diffusers' converter builds it with, read off the file as it reads them.
    resampler = "perceiver_resampler"
    config = {
        "_class_name": type(projection).__name__,
        "embed_dims": image_proj[f"{resampler}.proj_in.weight"].shape[0],
        "output_dims": image_proj[f"{resampler}.proj_out.weight"].shape[0],
        "hidden_dims": image_proj[f"{resampler}.proj_in.weight"].shape[1],
        "heads": image_proj[f"{resampler}.layers.0.0.to_q.weight"].shape[0] // 64,
        "id_embeddings_dim": image_proj["proj.0.weight"].shape[1],
    }
    tensors = {k: v.to(torch.float16) for k, v in projection.state_dict().items()}
    metadata = {"source": H94 + name, "sha256": digest}
    written(out / "image_proj", tensors, metadata, "diffusion_pytorch_model")
    (out / "image_proj" / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    sites = {k: v for k, v in state["ip_adapter"].items() if k.endswith(("to_k_ip.weight", "to_v_ip.weight"))}
    written(out / "ip_adapter", sites, metadata, "diffusion_pytorch_model")
    left = len(state["ip_adapter"]) - len(sites)
    print(f"image_proj: {config}, {len(tensors)} tensors; ip_adapter: {len(sites)} tensors ({left} LoRA factors left out)")
    (out / "README.md").write_text(
        f"h94's IP-Adapter-FaceID-PlusV2 for SDXL ({H94}{name}, sha256 {digest}), in Diffusers' loader layout; its LoRA "
        f"is h94/ip-adapter-faceid-plusv2-sdxl-lora.\n\n{LICENSE}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", choices=("buffalo", "faceid"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--from", dest="given", type=Path, help="the source file, already downloaded")
    args = parser.parse_args()
    (buffalo if args.source == "buffalo" else faceid)(args.out, args.given)


if __name__ == "__main__":
    main()
