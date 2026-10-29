"""HunyuanImage 3.0: the package's model code against Tencent's, on CPU, with a small random-weight model.

    python scripts/hunyuan-image-3-model-proof.py <HunyuanImage-3.0 checkout> <checkpoint files dir>

The checkout is `Tencent-Hunyuan/HunyuanImage-3.0` at 6e9113a6; the files dir holds the Instruct
checkpoint's `config.json`, `generation_config.json`, `tokenizer.json` and `tokenizer_config.json`.
Run it in Tencent's environment (torch, transformers 4.57.1, diffusers 0.35.2, einops). Both models are
built from one shrunken config and share weights; each case compares Tencent's `generate_image` with
the package's `generate_cot` / `denoise` / `decode_latents`: the chain of thought must be identical
(same seeds) and the images equal within float32 noise. It also checks that the package's full-size
module tree names exactly the checkpoint's tensors.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer

UPSTREAM, FILES = Path(sys.argv[1]), Path(sys.argv[2])
PORT = Path(__file__).resolve().parent.parent / "hunyuan-image-3" / "hunyuan_image_3" / "tencent"


def load_package(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Tencent's eager MoE pins the CUDA device and marks NVTX ranges.
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.nvtx.range = lambda *a, **k: contextlib.nullcontext()


class CpuAutocast(torch.autocast):  # type: ignore[misc]
    """Both models wrap their forwards in `torch.autocast("cuda", ...)`; on this CPU the same regions
    autocast on the CPU, so the bfloat16 modules take float32 latents as they do on a GPU. The VAE's float16
    regions run in bfloat16: CPU group norm takes no float16 input beside bfloat16 weights."""

    def __init__(self, device_type: str, dtype: Any = None, *args: Any, **kwargs: Any) -> None:
        if device_type == "cuda":
            device_type, dtype = "cpu", torch.bfloat16 if dtype == torch.float16 else dtype
        super().__init__(device_type, dtype, *args, **kwargs)


torch.autocast = CpuAutocast


@contextlib.contextmanager
def cpu_empty() -> Iterator[None]:
    """Tencent's VAE allocates an empty CUDA tensor in its constructor; build it on CPU instead. Only
    for that construction: `torch.device("meta")` recognizes the real `torch.empty` only."""
    empty = torch.empty
    torch.empty = lambda *a, **k: empty(*a, **{**k, "device": "cpu"} if k.get("device") == "cuda" else k)
    try:
        yield
    finally:
        torch.empty = empty


# Both trees are packages named `hunyuan_image_3`, so each is loaded under its own name.
sys.path.insert(0, str(UPSTREAM))
load_package("tencent_upstream", UPSTREAM / "hunyuan_image_3")
load_package("hunyuan_port", PORT)
Reference: Any = importlib.import_module("tencent_upstream.modeling_hunyuan_image_3").HunyuanImage3ForCausalMM
HunyuanImage3Config: Any = importlib.import_module("tencent_upstream.configuration_hunyuan_image_3").HunyuanImage3Config
Port: Any = importlib.import_module("hunyuan_port.modeling_hunyuan_image_3").HunyuanImage3ForCausalMM
PortConfig: Any = importlib.import_module("hunyuan_port.configuration_hunyuan_image_3").HunyuanImage3Config
get_system_prompt: Any = importlib.import_module("hunyuan_port.system_prompt").get_system_prompt
PortTokenizer: Any = importlib.import_module("hunyuan_port.tokenization_hunyuan_image_3").HunyuanImage3TokenizerFast
GenerationConfig: Any = importlib.import_module("transformers").GenerationConfig
_routing_spec = importlib.util.spec_from_file_location("hunyuan_routing", PORT.parent / "routing.py")
assert _routing_spec is not None and _routing_spec.loader is not None
routing_module = importlib.util.module_from_spec(_routing_spec)
_routing_spec.loader.exec_module(routing_module)

FULL = json.loads((FILES / "config.json").read_text())
GENERATION = json.loads((FILES / "generation_config.json").read_text())
TINY = {
    "hidden_size": 64, "num_attention_heads": 4, "attention_head_dim": 16, "head_dim": 16,
    "num_key_value_heads": 2, "num_hidden_layers": 2, "intermediate_size": 48,
    "moe_intermediate_size": [32, 32], "moe_topk": [2, 2], "num_experts": 4, "num_shared_expert": [1, 1],
    "image_base_size": 256, "patch_embed_hidden_dim": 32,
    "vae": {**FULL["vae"], "block_out_channels": [32] * 5, "latent_channels": 16, "layers_per_block": 1},
    "vit": {**FULL["vit"], "hidden_size": 32, "intermediate_size": 64, "num_attention_heads": 2,
            "num_hidden_layers": 1},
    "vit_aligner": {**FULL["vit_aligner"], "input_dim": 32, "n_embed": 64},
}


def tiny(distilled: bool) -> dict[str, Any]:
    return {**FULL, **TINY, "cfg_distilled": distilled, "use_meanflow": distilled}


def build(distilled: bool) -> tuple[Any, Any]:
    torch.manual_seed(0)
    config = HunyuanImage3Config(**tiny(distilled))
    config._attn_implementation = "sdpa"
    with cpu_empty():
        reference = Reference(config).eval()
    reference.load_tokenizer(str(FILES))
    reference.generation_config = GenerationConfig(**GENERATION)
    port = Port(PortConfig(**tiny(distilled))).eval()
    # The served dtypes, as `from_pretrained(torch_dtype="auto")` and the package's Graph make them:
    # bfloat16 throughout; Tencent's expert routers are float32 modules of bfloat16 values, the
    # package's stay bfloat16 (as stored) and route in float32.
    for model in (reference, port):
        model.to(torch.bfloat16)
        if model is reference:
            for layer in model.model.layers:
                layer.mlp.gate.wg.float()
        # CPU autocast has no float32 rule for group norm, so the VAE keeps float32 weights here (on a GPU it
        # is bfloat16). Tencent's `dtype` is its first parameter's, the VAE's: report the backbone's as served.
        model.vae.float()
    type(reference).dtype = property(lambda self: self.model.wte.weight.dtype)
    port.load_state_dict(reference.state_dict(), strict=True)
    tokenizer_config = json.loads((FILES / "tokenizer_config.json").read_text())
    tokenizer_config.pop("tokenizer_class")
    port._tokenizer = PortTokenizer(
        tokenizer_object=Tokenizer.from_str((FILES / "tokenizer.json").read_text()), **tokenizer_config)
    port.generation_config = SimpleNamespace(**GENERATION)
    return reference, port


def full_tree_keys() -> None:
    """The package's full-size construction names exactly the checkpoint's tensors."""
    index = json.loads((FILES / "model.safetensors.index.json").read_text())["weight_map"]
    with torch.device("meta"):
        port = Port(PortConfig(**FULL))
    built = set(port.state_dict())
    missing, extra = sorted(set(index) - built), sorted(built - set(index))
    assert not missing and not extra, (missing[:5], extra[:5])
    print(f"full tree: {len(built)} tensors, identical to the checkpoint index")


def picture(seed: int, size: tuple[int, int]) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


@contextlib.contextmanager
def biased_ends(*models: Any) -> Iterator[None]:
    """With random weights a thought never ends: make `</think>` and `</recaption>` the likely tokens, alike in
    both models, so the stage transitions and the size pick after the recaption run."""
    tokenizer = models[1]._tokenizer
    ends = [tokenizer.end_of_think_token_id, tokenizer.end_of_recaption_token_id]

    def hook(module: Any, inputs: Any, output: Any) -> Any:
        output = output.clone()
        output[..., ends] += 1e4
        return output

    handles = [model.lm_head.register_forward_hook(hook) for model in models]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def case(name: str, reference: Any, port: Any, *, images: list[Image.Image], bot_task: str,
         image_size: str, steps: int, seed: int, ends: bool = False) -> None:
    started = time.monotonic()
    prompt = f"A red fox in fresh snow, case {name}"
    reference.generation_config.diff_infer_steps = steps
    port.generation_config.max_new_tokens = 48
    random.seed(seed)
    torch.manual_seed(seed + 1)
    # Tencent's postprocess resizes an edit's output to its reference's exact ratio; the package keeps
    # the model's own size (the lead's ruling), so the comparison is before that resize.
    reference.image_processor.postprocess_outputs = lambda outputs, **_: outputs
    with biased_ends(reference, port) if ends else contextlib.nullcontext():
        cot_ref, (image_ref,) = reference.generate_image(
            prompt=prompt, image=images or None, seed=seed, image_size=image_size,
            use_system_prompt="en_unified", bot_task=bot_task, infer_align_image_size=True, max_new_tokens=48,
        )
        random.seed(seed)
        text_seed = random.randint(0, 10_000_000) if bot_task != "image" else None
        system_prompt = get_system_prompt("en_unified", bot_task).strip()
        cond = [port.image_processor.build_cond_images(image_list=images, infer_align_image_size=True)] if images else None
        cot, ratio = None, None
        if bot_task != "image":
            generator = torch.Generator().manual_seed(seed + 1)
            cot, ratio = port.generate_cot(prompt, cond, system_prompt, bot_task, seed=text_seed,
                                           generator=generator, need_ratio=image_size == "auto")
            assert [cot] == cot_ref, (cot[:200], cot_ref[0][:200] if cot_ref else None)
        elif image_size == "auto":
            ratio = port.predict_ratio(prompt, cond, system_prompt, seed=seed)
    if ratio is not None:
        size = port.image_processor.vae_reso_group[ratio]
        image_size = f"{size.height}x{size.width}"
    latents = port.denoise(prompt, cond, system_prompt, cot, image_size=image_size,
                           steps=steps, guidance_scale=GENERATION["diff_guidance_scale"], seed=seed)
    image = port.decode_latents(latents)
    assert image.size == image_ref.size, (image.size, image_ref.size)
    diff = np.abs(np.asarray(image, dtype=np.int16) - np.asarray(image_ref, dtype=np.int16)).max()
    assert diff <= 1, f"{name}: max pixel difference {diff}"
    picked = f", picked {image_size}" if ratio is not None else ""
    print(f"{name}: cot {len(cot or '')} chars identical{picked}, image {image.size} max |Δ| {diff} "
          f"[{time.monotonic() - started:.0f}s]", flush=True)


def brute_lru(sequence: list[int], size: int) -> int:
    cache: list[int] = []
    hits = 0
    for key in sequence:
        if key in cache:
            hits += 1
            cache.remove(key)
        elif len(cache) == size:
            cache.pop(0)
        cache.append(key)
    return hits


def brute_min(sequence: list[int], size: int) -> int:
    cache: set[int] = set()
    hits = 0
    for i, key in enumerate(sequence):
        if key in cache:
            hits += 1
            continue
        if len(cache) == size:
            uses = {k: next((j for j in range(i + 1, len(sequence)) if sequence[j] == k), len(sequence)) for k in cache}
            cache.remove(max(sorted(cache), key=uses.__getitem__))
        cache.add(key)
    return hits


def routing_replay(port: Any) -> None:
    """The routing record against the experts' own calls: forward hooks see every expert a forward runs and its
    rows; the record must match them, and its LRU and MIN replays must match brute-force ones."""
    experts = port.config.num_experts
    calls: list[tuple[int, int, int]] = []
    hooks = [expert.register_forward_hook(
        lambda module, inputs, output, layer=layer, index=index: calls.append((layer, index, inputs[0].shape[0])))
        for layer, decoder in enumerate(port.model.layers) for index, expert in enumerate(decoder.mlp.experts)]
    forwards = [0]
    hooks.append(port.register_forward_pre_hook(lambda module, args, kwargs: forwards.__setitem__(0, forwards[0] + 1),
                                                with_kwargs=True))
    caches = (2, 3, 5)
    try:
        for stage in ("denoise", "think"):
            calls.clear()
            forwards[0] = 0
            record = routing_module.Routing(experts=experts, caches=caches)
            for decoder in port.model.layers:
                decoder.mlp.trace = record
            prompt = f"A lantern festival on a river, routing replay {stage}"
            if stage == "denoise":
                port.denoise(prompt, None, get_system_prompt("en_unified", "image").strip(), None,
                             image_size="256x256", steps=3, guidance_scale=2.5, seed=21)
            else:
                port.generation_config.max_new_tokens = 12
                port.generate_cot(prompt, None, get_system_prompt("en_unified", "think_recaption").strip(),
                                  "think_recaption", seed=22, generator=torch.Generator().manual_seed(23))
            for decoder in port.model.layers:
                decoder.mlp.trace = None
            sequence = [layer * experts + index for layer, index, _ in calls]
            assert list(record.sequence) == sequence, stage
            assert list(record.tokens_per_hit) == [rows for _, _, rows in calls], stage
            assert record.forwards == forwards[0], (record.forwards, forwards[0])
            summary = record.summary()
            for size in caches:
                assert round(brute_lru(sequence, size) / len(sequence), 4) == summary[f"lru_{size}_hit_rate"], size
                assert round(brute_min(sequence, size) / len(sequence), 4) == summary[f"min_{size}_hit_rate"], size
            print(f"routing replay {stage}: {summary}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()


def main() -> None:
    torch.set_num_threads(4)
    if (FILES / "model.safetensors.index.json").exists():
        full_tree_keys()
    if "--routing" in sys.argv:
        routing_replay(build(True)[1])
        return
    for distilled in (False, True):
        reference, port = build(distilled)
        label = "distil" if distilled else "instruct"
        case(f"{label} t2i 2:1", reference, port, images=[], bot_task="image", image_size="352x176", steps=3,
             seed=8)
        case(f"{label} edit", reference, port, images=[picture(1, (300, 200))], bot_task="image",
             image_size="304x208", steps=2, seed=9)
        case(f"{label} think t2i", reference, port, images=[], bot_task="think_recaption",
             image_size="256x256", steps=2, seed=11)
        case(f"{label} auto edit", reference, port, images=[picture(5, (320, 192))], bot_task="image",
             image_size="auto", steps=2, seed=13)
        if not distilled:
            case(f"{label} fusion", reference, port, images=[picture(2, (256, 256)), picture(3, (200, 320))],
                 bot_task="image", image_size="256x256", steps=2, seed=10)
            case(f"{label} think edit", reference, port, images=[picture(4, (256, 256))],
                 bot_task="think_recaption", image_size="256x256", steps=2, seed=12)
            case(f"{label} think auto t2i", reference, port, images=[], bot_task="think_recaption",
                 image_size="auto", steps=2, seed=14, ends=True)
    print("model proof passed")


if __name__ == "__main__":
    main()
