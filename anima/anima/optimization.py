"""Anima's fused CUDA execution plan, installed on the filled Cosmos transformer.

Only Python callables change: parameters, buffers, state-dict names and storage stay as the
fill left them. One closed plan for the reviewed Diffusers sources, not a patching
framework: any drift runs eager and says why. The kernel is the Runtime's public fused
RMSNorm + split-half RoPE (`cozy_runtime.author.kernels`), compiled into its linux x86
cp312-abi3 wheel; it maps plane stand-ins for its launch itself.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from types import MethodType
from typing import Any

import torch
from cozy_runtime.author import device
from cozy_runtime.author.kernels import int8_rows, native, quantize_once, rms_rope_split_half
from diffusers.models.activations import GELU
from diffusers.models.attention import FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_cosmos import (
    CosmosAdaLayerNormZero,
    CosmosAttnProcessor2_0,
    CosmosRotaryPosEmbed,
    CosmosTransformer3DModel,
    CosmosTransformerBlock,
)
from diffusers.modular_pipelines.anima.before_denoise import AnimaPrepareLatentsStep

# The extension takes DLPack arrays and a CUDA stream through nanobind and never links
# libtorch, so an explicit Torch/CUDA cohort is admitted rather than every future release.
TORCH_RELEASES = ("2.13.0", "2.14.0")
CUDA_RELEASE = "13.0"
#: Diffusers 0.40.0's sources the plan replaces or mirrors.
_SOURCE_DIGESTS: dict[str, tuple[Callable[..., object], str]] = {
    "RMSNorm.forward": (
        RMSNorm.forward,
        "fa91ca74b4badaaf916ec4892e2b6ad90b4b904c7b570dba8ed99fd0ec69439b",
    ),
    "CosmosAttnProcessor2_0.__call__": (
        CosmosAttnProcessor2_0.__call__,
        "5ec61c5c754278570e927dc2bfd0f8edb71afebfdd7b90e58e542142290b56d2",
    ),
    "CosmosRotaryPosEmbed.forward": (
        CosmosRotaryPosEmbed.forward,
        "ee752d88e85f73ccde5dac50c186bd8dd9382f2d1aea8c57db2571e86b8be0e1",
    ),
    "CosmosTransformerBlock.forward": (
        CosmosTransformerBlock.forward,
        "c208749aa823185422d31daa08a40f374a86c13772a65179f5409f63a4703b0d",
    ),
    "CosmosAdaLayerNormZero.forward": (
        CosmosAdaLayerNormZero.forward,
        "c518f36caac8a01a4779325e7161e4050a51e9b4227274ef47c46e20bb946dd7",
    ),
    "FeedForward.forward": (
        FeedForward.forward,
        "dae2c9f53987c25ad38cacbeafb54bd72d4b780ab9772413b559a0a94c3c7058",
    ),
    "GELU.forward": (
        GELU.forward,
        "a23bf6faca366d6576b4c079870c27d9b1470013d8d1882d3f21f31f9fd527e6",
    ),
    "AnimaPrepareLatentsStep.__call__": (
        AnimaPrepareLatentsStep.__call__,
        "5f758c69ad2f82a37d707e3ca736b88011b41f04a958dbc657ec397b88c2bef5",
    ),
}
#: Anima Base 1.0's transformer, the one the norm and RoPE shapes below are reviewed for.
_TRANSFORMER_CONFIG = {
    "_class_name": "CosmosTransformer3DModel",
    "_diffusers_version": "0.39.0.dev0",
    "adaln_lora_dim": 256,
    "attention_head_dim": 128,
    "concat_padding_mask": True,
    "controlnet_block_every_n": None,
    "crossattn_proj_in_channels": 1024,
    "encoder_hidden_states_channels": 1024,
    "extra_pos_embed_type": None,
    "img_context_dim_in": None,
    "img_context_dim_out": 2048,
    "img_context_num_tokens": 256,
    "in_channels": 16,
    "max_size": [128, 240, 240],
    "mlp_ratio": 4.0,
    "num_attention_heads": 16,
    "num_layers": 28,
    "out_channels": 16,
    "patch_size": [1, 2, 2],
    "rope_scale": [1.0, 4.0, 4.0],
    "text_embed_dim": 1024,
    "use_crossattn_projection": False,
}


class PlanRefusal(RuntimeError):
    """The plan was installed but its exact implementation did not match."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def apply(pipe: Any) -> str:
    """Install the plan on a filled pipeline; returns why it runs eager instead, or ""."""
    refusal = _capability() or _plan_refusal(pipe)
    if refusal:
        return refusal
    transformer = pipe.components["transformer"]
    before = _tensor_census(transformer)
    _install_transformer_plan(transformer)
    if _tensor_census(transformer) != before:
        raise PlanRefusal(
            "anima_optimization_tensor_schema_changed",
            "the post-fill plan changed the tensor schema or parameter/buffer storage identity",
        )
    return ""


def installed(transformer: Any) -> bool:
    """Whether `apply` installed the plan on this transformer."""
    return isinstance(transformer.transformer_blocks[0].attn1.processor, _FusedSelfAttention)


class FusedLatents(AnimaPrepareLatentsStep):
    """Diffusers' latent preparation with the padding mask at latent size: the transformer
    resizes it to latent size every step, and resizing zeros gives the same zeros."""

    @torch.no_grad()
    def __call__(self, components: Any, state: Any) -> Any:
        block_state = self.get_block_state(state)
        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width
        self.check_inputs(components, block_state)
        block_state.latents = self.prepare_latents(
            batch_size=block_state.batch_size * block_state.num_images_per_prompt,
            num_channels_latents=components.num_channels_latents,
            height=block_state.height,
            width=block_state.width,
            vae_scale_factor=components.vae_scale_factor,
            dtype=torch.float32,
            device=components._execution_device,
            generator=block_state.generator,
            latents=block_state.latents,
        )
        block_state.padding_mask = block_state.latents.new_zeros(
            1, 1, block_state.latents.shape[-2], block_state.latents.shape[-1],
            dtype=block_state.dtype,
        )
        self.set_block_state(state, block_state)
        return components, state


def _capability() -> str:
    """Why this process cannot run the fused kernels, or "" when it can."""
    release = torch.__version__.split("+", 1)[0]
    if release not in TORCH_RELEASES:
        return f"torch {release} is outside the fused plan's releases {', '.join(TORCH_RELEASES)}"
    if torch.version.cuda != CUDA_RELEASE:
        return f"torch.version.cuda {torch.version.cuda!r} != the fused plan's {CUDA_RELEASE!r}"
    if (placed := device()).type != "cuda":
        return f"this process runs on {placed}, not a CUDA GPU"
    return native()


def _plan_refusal(pipe: Any) -> str:
    """Why this construction is not the reviewed Anima graph, or "" when it is."""
    for name, (source, expected) in _SOURCE_DIGESTS.items():
        actual = hashlib.sha256(inspect.getsource(source).encode()).hexdigest()
        if actual != expected:
            return f"{name} has source sha256:{actual}, not the reviewed sha256:{expected}"
    transformer = pipe.components["transformer"]
    if type(transformer) is not CosmosTransformer3DModel:
        return f"the transformer is {type(transformer).__qualname__}, not CosmosTransformer3DModel"
    if json.loads(json.dumps(dict(transformer.config))) != _TRANSFORMER_CONFIG:
        return "the filled Cosmos transformer config is not the Anima Base 1.0 config"
    return ""


def _tensor_census(module: Any) -> tuple[tuple[object, ...], tuple[object, ...]]:
    schema = tuple(
        (name, tuple(value.shape), str(value.dtype), type(value).__qualname__)
        for name, value in module.state_dict(keep_vars=True).items()
    )
    storage = tuple(
        (name, id(value), int(value.data_ptr()))
        for name, value in (
            *module.named_parameters(remove_duplicate=False),
            *module.named_buffers(remove_duplicate=False),
        )
    )
    return schema, storage


class _RopeMatrix:
    """One call-scoped conversion shared by all 28 self-attention blocks."""

    def __init__(self) -> None:
        self.value: Any | None = None

    def clear(self) -> None:
        self.value = None

    def get(self, rotary: Any) -> Any:
        if self.value is not None:
            return self.value
        if hasattr(rotary, "shape"):
            if rotary.ndim != 6 or tuple(rotary.shape[-3:]) != (64, 2, 2):
                raise PlanRefusal(
                    "anima_optimization_rope_layout",
                    f"direct Cosmos rotary matrix has unsupported shape {tuple(rotary.shape)}",
                )
            return rotary
        if not isinstance(rotary, tuple) or len(rotary) != 2:
            raise PlanRefusal(
                "anima_optimization_rope_layout",
                "Diffusers Cosmos did not provide its exact (cos, sin) rotary pair",
            )
        cos, sin = rotary
        if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[-1] != 128:
            raise PlanRefusal(
                "anima_optimization_rope_layout",
                f"Diffusers Cosmos rotary tensors have unsupported shapes {tuple(cos.shape)} "
                f"and {tuple(sin.shape)}",
            )
        half = cos.shape[-1] // 2
        # CosmosRotaryPosEmbed repeats the same half in source. The source digest is the
        # proof; a device->host equality read in every call would erase the kernel's win.
        cos, sin = cos[..., :half], sin[..., :half]
        self.value = torch.stack((cos, -sin, sin, cos), dim=-1).reshape(
            1, 1, cos.shape[0], half, 2, 2
        )
        return self.value


def _direct_rope_forward(rope: Any, hidden_states: Any, fps: int | None = None) -> Any:
    """The fused kernel's rotation matrix directly, without duplicate cosine/sine tensors."""
    _, _, frames, height, width = hidden_states.shape
    sizes = [
        frames // rope.patch_size[0],
        height // rope.patch_size[1],
        width // rope.patch_size[2],
    ]
    device = hidden_states.device
    seq = torch.arange(max(rope.max_size), device=device, dtype=torch.float32)

    def frequencies(dim: int, ntk: float) -> Any:
        exponent = torch.arange(0, dim, 2, device=device, dtype=torch.float32)[: dim // 2] / dim
        return 1.0 / ((10000.0 * ntk) ** exponent)

    freq_h = frequencies(rope.dim_h, rope.h_ntk_factor)
    freq_w = frequencies(rope.dim_w, rope.w_ntk_factor)
    freq_t = frequencies(rope.dim_t, rope.t_ntk_factor)
    emb_h = torch.outer(seq[: sizes[1]], freq_h)[None, :, None, :].repeat(sizes[0], 1, sizes[2], 1)
    emb_w = torch.outer(seq[: sizes[2]], freq_w)[None, None, :, :].repeat(sizes[0], sizes[1], 1, 1)
    time = seq[: sizes[0]] if fps is None else seq[: sizes[0]] / fps * rope.base_fps
    emb_t = torch.outer(time, freq_t)[:, None, None, :].repeat(1, sizes[1], sizes[2], 1)
    phase = torch.cat((emb_t, emb_h, emb_w), dim=-1).flatten(0, 2).float()
    cos, sin = torch.cos(phase), torch.sin(phase)
    return torch.stack((cos, -sin, sin, cos), dim=-1).reshape(
        1, 1, phase.shape[0], phase.shape[1], 2, 2
    )


class _FusedSelfAttention:
    #: A Runtime attention site: the Runtime sets the kernel per site (`dispatch_attention_fn`).
    _attention_backend: Any = None
    _parallel_config: Any = None

    def __init__(self, kernel: Callable[..., tuple[Any, Any]], rope: _RopeMatrix) -> None:
        self.kernel = kernel
        self.rope = rope

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        image_rotary_emb: Any | None = None,
    ) -> Any:
        if encoder_hidden_states is not None or attention_mask is not None:
            raise PlanRefusal(
                "anima_optimization_attention_shape",
                "the fused Cosmos self-attention plan received cross-attention inputs",
            )
        if image_rotary_emb is None:
            raise PlanRefusal(
                "anima_optimization_rope_absent",
                "the exact Anima Cosmos self-attention call omitted rotary embeddings",
            )
        with quantize_once(hidden_states):
            query, key, value = (
                projection(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
                for projection in (attn.to_q, attn.to_k, attn.to_v)
            )
        # The kernel reads the scales by pointer, without calling the norms that own them.
        query, key = self.kernel(
            query,
            key,
            self.rope.get(image_rotary_emb),
            attn.norm_q.weight,
            attn.norm_k.weight,
            attn.norm_q.eps,
        )
        # Diffusers repeats K and V by the ratio of head widths, 1 in every Cosmos model;
        # `repeat_interleave(1)` still copies both whole (huaisang, rtx-4070-super: 0.35 s
        # of a 30-step render, most of Anima's gap to ComfyUI).
        if key.shape[3] != query.shape[3]:
            key = key.repeat_interleave(query.shape[3] // key.shape[3], dim=3)
        if value.shape[3] != query.shape[3]:
            value = value.repeat_interleave(query.shape[3] // value.shape[3], dim=3)
        output = dispatch_attention_fn(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        output = output.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](output))


def _native_rms_forward(module: Any, hidden_states: Any) -> Any:
    output = torch.nn.functional.rms_norm(hidden_states, module.dim, module.weight, module.eps)
    return output if module.bias is None else output + module.bias


def _hooked(*modules: Any) -> bool:
    return any(m._forward_hooks or m._forward_pre_hooks for m in modules)


def _normed(norm: Any, hidden: Any, timestep: Any, temb: Any, consumers: tuple[Any, ...]) -> Any:
    """`CosmosAdaLayerNormZero.forward`'s rows and gate. When every consumer is an int8 leaf the
    rows come quantized from the norm's one pass (`int8_rows`), else as Diffusers makes them."""
    table = norm.linear_2(norm.linear_1(norm.activation(timestep)))
    if temb is not None:
        table = table + temb
    shift, scale, gate = table.chunk(3, dim=-1)
    if table.ndim == 2:
        shift, scale, gate = (t.unsqueeze(1) for t in (shift, scale, gate))
    rows = int8_rows(hidden, consumers, norm="layer", eps=norm.norm.eps, scale=scale, shift=shift)
    if rows is None:
        rows = norm.norm(hidden) * (1 + scale) + shift
    return rows, gate


def _feed_forward(ff: Any, rows: Any) -> Any:
    """`FeedForward.forward` for Cosmos' GELU MLP, its GELU folded into the down projection's
    int8 quantization when that is an int8 leaf."""
    up = ff.net[0].proj(rows)
    activation = "gelu" if ff.net[0].approximate == "none" else "gelu_tanh"
    down = int8_rows(up, (ff.net[2],), activation=activation)
    return ff.net[2](ff.net[1](ff.net[0].gelu(up)) if down is None else down)


def _block_forward(
    block: Any,
    hidden_states: Any,
    encoder_hidden_states: Any,
    embedded_timestep: Any,
    temb: Any | None = None,
    image_rotary_emb: Any | None = None,
    extra_pos_emb: Any | None = None,
    attention_mask: Any | None = None,
    controlnet_residual: Any | None = None,
    latents: Any | None = None,
    block_idx: int | None = None,
) -> Any:
    """`CosmosTransformerBlock.forward` with each gated residual as one `addcmul`."""
    if not isinstance(hidden_states, torch.Tensor):
        raise PlanRefusal("anima_optimization_tensor_type", "Cosmos received a non-torch tensor")
    if block.before_proj is not None:
        hidden_states = block.before_proj(hidden_states) + latents
    if extra_pos_emb is not None:
        hidden_states = hidden_states + extra_pos_emb
    attn1, attn2, ff = block.attn1, block.attn2, block.ff
    # Rows quantized in the norm's pass reach the projections through modules nothing hooks.
    fused = not _hooked(attn1, attn2, ff, *ff.net)
    consumers: tuple[Any, ...] = (attn1.to_q, attn1.to_k, attn1.to_v) if fused else ()
    normalized, gate = _normed(block.norm1, hidden_states, embedded_timestep, temb, consumers)
    output = attn1(normalized, image_rotary_emb=image_rotary_emb)
    hidden_states = torch.addcmul(hidden_states, gate, output)
    consumers = (attn2.to_q,) if fused else ()
    normalized, gate = _normed(block.norm2, hidden_states, embedded_timestep, temb, consumers)
    with quantize_once(encoder_hidden_states):
        output = attn2(
            normalized, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )
    hidden_states = torch.addcmul(hidden_states, gate, output)
    consumers = (ff.net[0].proj,) if fused else ()
    normalized, gate = _normed(block.norm3, hidden_states, embedded_timestep, temb, consumers)
    output = _feed_forward(ff, normalized) if fused else ff(normalized)
    hidden_states = torch.addcmul(hidden_states, gate, output)
    if controlnet_residual is not None:
        if block.after_proj is not None:
            raise PlanRefusal(
                "anima_optimization_controlnet_shape",
                "Cosmos supplied a control residual together with after_proj",
            )
        hidden_states += controlnet_residual
    if block.after_proj is not None:
        return hidden_states, block.after_proj(hidden_states)
    return hidden_states


def _install_transformer_plan(transformer: Any) -> None:
    rope = _RopeMatrix()
    blocks = tuple(transformer.transformer_blocks)
    if len(blocks) != 28 or any(type(block) is not CosmosTransformerBlock for block in blocks):
        raise PlanRefusal(
            "anima_optimization_block_shape",
            "the exact Anima plan requires 28 unmodified Diffusers CosmosTransformerBlock rows",
        )
    norms = tuple(module for module in transformer.modules() if type(module) is RMSNorm)
    shapes = sorted((tuple(norm.dim), norm.eps) for norm in norms)
    if shapes != sorted([((128,), 1e-5)] * 112 + [((2048,), 1e-6)]):
        raise PlanRefusal(
            "anima_optimization_norm_shape",
            "the exact Anima plan requires 112 head-width RMSNorms at eps=1e-5 and one "
            "model-width RMSNorm at eps=1e-6",
        )
    if any(norm.weight is None or norm.bias is not None for norm in norms):
        raise PlanRefusal(
            "anima_optimization_norm_shape",
            "Anima RMSNorm affine/bias/epsilon does not match the fused plan",
        )
    for norm in norms:
        norm.forward = MethodType(_native_rms_forward, norm)
    for block in blocks:
        if (
            block.before_proj is not None
            or block.after_proj is not None
            or type(block.attn1.processor) is not CosmosAttnProcessor2_0
            or type(block.attn2.processor) is not CosmosAttnProcessor2_0
        ):
            raise PlanRefusal(
                "anima_optimization_block_shape",
                "Anima Cosmos block projections or attention processors do not match the plan",
            )
        norms = (block.norm1, block.norm2, block.norm3)
        if (
            any(type(n) is not CosmosAdaLayerNormZero for n in norms)
            or any(n.norm.weight is not None or n.norm.bias is not None for n in norms)
            or len(block.ff.net) != 3
            or type(block.ff.net[0]) is not GELU
            or type(block.ff.net[1]) is not torch.nn.Dropout
        ):
            raise PlanRefusal(
                "anima_optimization_block_shape",
                "Anima Cosmos block norms or feed-forward do not match the plan",
            )
        block.attn1.set_processor(_FusedSelfAttention(rms_rope_split_half, rope))
        block.forward = MethodType(_block_forward, block)
    transformer.rope.forward = MethodType(_direct_rope_forward, transformer.rope)
    original = transformer.forward

    def forward(_self: Any, *args: object, **kwargs: object) -> Any:
        rope.clear()
        try:
            return original(*args, **kwargs)
        finally:
            rope.clear()

    transformer.forward = MethodType(forward, transformer)
