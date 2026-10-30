"""Qwen3.6: the package's model code (`cozy_runtime.vlm`, `cozy_runtime.moe`) against Transformers', on
CPU, with small random-weight models.

    python scripts/qwen3.6-model-proof.py <checkpoint files dir>

The files dir holds a Qwen3.6 checkpoint's `config.json`, `generation_config.json`, tokenizer, processor
configs and chat template (`Qwen/Qwen3.6-35B-A3B@995ad96e`). For the dense and the MoE layout, one
shrunken config builds Transformers' model (fused experts) and the package's (one block per expert, the
multi-token-prediction head attached) from the same float32 weights. Each case asks with an image and a
video: the processor's tensors must equal Transformers' own chat-template path, the logits match within
float32 noise, and the greedy answers equal Transformers' `generate`, with and without the draft head; a
sampled answer replays from its seed, drafted or not, and another seed gives another.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, GenerationConfig

from cozy_runtime import lm, moe, vlm

FILES = Path(sys.argv[1])


class Files:
    def names(self) -> list[str]:
        return [p.name for p in FILES.iterdir()]

    def read(self, name: str) -> bytes:
        return (FILES / name).read_bytes()


class Stored:
    """The checkpoint's dtypes, as Runtime's construction config reports them: all float32 here."""

    def __init__(self, mapping: dict[str, Any]) -> None:
        self._mapping = mapping

    def mapping(self) -> dict[str, Any]:
        return self._mapping

    def tensor_dtype(self, key: str, default: str = "") -> str:
        return "f32"


def shrunk(moe_layout: bool) -> dict[str, Any]:
    config = json.loads((FILES / "config.json").read_text())
    text = config["text_config"]
    text.update(
        hidden_size=64, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, max_position_embeddings=4096,
    )
    text["rope_parameters"]["mrope_section"] = [2, 1, 1]
    if moe_layout:
        text.update(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32)
    else:
        config["architectures"], config["model_type"] = ["Qwen3_5ForConditionalGeneration"], "qwen3_5"
        config["vision_config"]["model_type"], text["model_type"] = "qwen3_5", "qwen3_5_text"
        for key in ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size",
                    "router_aux_loss_coef", "output_router_logits"):
            text.pop(key, None)
        text["intermediate_size"] = 128
    config["vision_config"].update(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    return config


def reference(config: dict[str, Any]) -> torch.nn.Module:
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(AutoConfig.for_model(**config), dtype=torch.float32)
    for parameter in model.parameters():  # Transformers' init leaves the norms at zero
        parameter.data.normal_(0, 0.05)
    return model.eval()


def package(config: dict[str, Any], ref: torch.nn.Module) -> tuple[torch.nn.Module, Any]:
    """Runtime's construction, filled with the reference's weights the way the per-expert checkpoint
    stores them; the draft head random."""
    model, draft = vlm.construct(config, Stored(config))
    ours = dict(model.named_parameters())
    filled = set()
    for name, tensor in ref.state_dict().items():
        if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
            stem, _, kind = name.rpartition(".")
            role = "gate_and_up_proj" if kind == "gate_up_proj" else "down_proj"
            for e in range(tensor.shape[0]):
                key = f"{stem}.{e}.{role}.weight"
                ours[key].data.copy_(tensor[e])
                filled.add(key)
        else:
            ours[name].data.copy_(tensor)
            filled.add(name)
    torch.manual_seed(1)
    for name, parameter in ours.items():
        if name not in filled:
            assert name.startswith("mtp."), name
            parameter.data.normal_(0, 0.05)
    assert set(ours) - filled == {n for n in ours if n.startswith("mtp.")}
    return model, draft


def asker(model: torch.nn.Module, draft: Any) -> vlm.VisionLM:
    asking = vlm.VisionLM.__new__(vlm.VisionLM)
    asking.pipe = vlm.Pipeline.__new__(vlm.Pipeline)
    asking.pipe.components, asking.pipe.draft = {vlm.COMPONENT: model}, draft
    asking.processor = vlm.processor(Files())
    model.generation_config = GenerationConfig(**json.loads((FILES / "generation_config.json").read_text()))
    return asking


def question() -> vlm.Question:
    rng = np.random.default_rng(7)
    image = Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8))
    frames = [Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)) for _ in range(4)]
    return vlm.Question("What changes between the picture and the clip?", [image], [vlm.Watched(frames, [0.25, 0.75, 1.25, 1.75])])


def case(name: str, moe_layout: bool) -> None:
    started = time.perf_counter()
    config = shrunk(moe_layout)
    ref = reference(config)
    model, draft = package(config, ref)
    split = sum(isinstance(m, moe.Experts) for m in model.modules())
    assert split == (5 if moe_layout else 0), split  # 4 layers and the MTP layer
    asking = asker(model, draft)
    q = question()
    text = asking._chat(q, "Answer briefly.", False)
    expected = asking.processor.apply_chat_template(
        [{"role": "system", "content": "Answer briefly."},
         {"role": "user", "content": [{"type": "image"}, {"type": "video"}, {"type": "text", "text": q.text}]}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    assert text == expected and text.endswith("<think>\n\n</think>\n\n"), text
    inputs = asking._inputs(
        text=[text], images=q.images, videos=[q.videos[0].frames], do_sample_frames=False,
        video_metadata=[{"total_num_frames": 4, "fps": 1000.0, "frames_indices": [250, 750, 1250, 1750]}],
    )
    with torch.inference_mode():
        mine = model(**inputs).logits
        theirs = ref(**inputs).logits
    worst = float((mine - theirs).abs().max())
    assert worst < 1e-4, f"{name}: logits differ by {worst}"
    with torch.inference_mode():
        out = ref.generate(**inputs, max_new_tokens=12, do_sample=False, eos_token_id=[248046, 248044], pad_token_id=248044)
    want = asking.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    def answer(**decoding: Any) -> list[str]:
        sampling = {"temperature": 0.0, "top_p": 1.0, "seed": 0} | decoding
        return vlm.VisionLM.ask.__wrapped__(asking, [q], system="Answer briefly.", thinking=False, stop=[],
                                            max_new_tokens=12, presence_penalty=0.0, **sampling)

    drafted = answer()
    sampled = answer(temperature=1.0, top_p=0.9, seed=11)
    assert answer(temperature=1.0, top_p=0.9, seed=11) == sampled != answer(temperature=1.0, top_p=0.9, seed=12)
    lm.release(model)  # the next call builds a fresh engine, undrafted
    asking.pipe.draft = None
    plain = answer()
    assert drafted == plain == [want], (drafted, plain, want)
    assert answer(temperature=1.0, top_p=0.9, seed=11) == sampled, "a seed replays drafted or not"
    print(f"{name}: logits max |Δ| {worst:.2e}, answer {want!r} ({len(want)} chars), {time.perf_counter() - started:.1f} s")


def census() -> None:
    """The full-size 35B-A3B tree, built on the meta device, names exactly the per-expert checkpoint."""
    config = json.loads((FILES / "config.json").read_text())
    with torch.device("meta"):
        model, _ = vlm.construct(config, Stored(config))
    names = set(model.state_dict())
    index = json.loads((FILES / "model.safetensors.index.json").read_text())["weight_map"]
    stored = set()
    for key in index:
        if key.endswith(("experts.gate_up_proj", "experts.down_proj")):
            stem, _, kind = key.rpartition(".")
            role = "gate_and_up_proj" if kind == "gate_up_proj" else "down_proj"
            stored.update(f"{stem}.{e}.{role}.weight" for e in range(config["text_config"]["num_experts"]))
        else:
            stored.add(key)
    assert names == stored, (sorted(names - stored)[:5], sorted(stored - names)[:5])
    print(f"census: {len(names)} tensors, the per-expert checkpoint's exactly")


torch.set_num_threads(2)
case("dense", moe_layout=False)
case("moe", moe_layout=True)
census()
