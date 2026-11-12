"""Anima's execution plan, installed on the filled transformer and VAE.

Only Python callables change: parameters, buffers, state-dict names and storage stay as the
fill left them. One closed plan for the reviewed Diffusers sources, not a patching
framework: any drift runs eager and says why. The transformer's kernel is the Runtime's public
fused RMSNorm + split-half RoPE (`cozy_runtime.author.kernels`), compiled into its linux x86
cp312-abi3 wheel; it maps plane stand-ins for its launch itself. The VAE decodes its one frame
on the Runtime's VAE helpers (`fusion.ChannelNorm`, `fusion.LoneFrame`, `tiles.rows`), as ComfyUI's
WanVAE does.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Protocol, cast

import torch
import torch.nn.functional as F
from cozy_runtime import tiles
from cozy_runtime.accel import fusion
from cozy_runtime.author import device
from cozy_runtime.accel.kernels import PreQuantized, int8_rows, native, quantize_once, rms_rope_split_half
from diffusers.models.activations import GELU
from diffusers.models.attention import FeedForward
from diffusers.models.attention_dispatch import AttentionBackendName, dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
    QwenImageCausalConv3d,
    QwenImageDecoder3d,
    QwenImageResidualBlock,
    QwenImageRMS_norm,
    QwenImageUpBlock,
    QwenImageUpsample,
)
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_cosmos import (
    CosmosAdaLayerNormZero,
    CosmosAttnProcessor2_0,
    CosmosRotaryPosEmbed,
    CosmosTransformer3DModel,
    CosmosTransformerBlock,
)
from diffusers.modular_pipelines.anima.before_denoise import AnimaPrepareLatentsStep
from diffusers.modular_pipelines.anima.modular_pipeline import AnimaModularPipeline
from diffusers.modular_pipelines.modular_pipeline import BlockState, PipelineState

if TYPE_CHECKING:
    from diffusers.models._modeling_parallel import ParallelConfig

    from .adapters import Adapters
    from .model import AnimaPipeline

#: A rotary table: the fused kernel's rotation matrices, or Diffusers' (cos, sin) pair.
Rotary = torch.Tensor | tuple[torch.Tensor, torch.Tensor]

# The extension takes DLPack arrays and a CUDA stream through nanobind and never links libtorch,
# so any torch this package installs on serves it. It is built for CUDA 13.0: a torch built for
# CUDA 13.0 or later proves the driver runs it.
CUDA_MIN = (13, 0)
#: Diffusers 0.40.0's sources the plan replaces or mirrors.
#: A reviewed Diffusers function, checked by its source text.
type Reviewed = Callable[..., object]  # type: ignore[explicit-any]  # any signature: its source is what is read
_SOURCE_DIGESTS: dict[str, tuple[Reviewed, str]] = {
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
#: Diffusers 0.40.0's source the frame decode replaces.
_VAE_DIGESTS: dict[str, tuple[Reviewed, str]] = {
    "AutoencoderKLQwenImage._decode": (
        AutoencoderKLQwenImage._decode,
        "5e714669ba0348fba8b2437d4fd4f32299b976fa64a3bd54b79affc41ae2deb0",
    ),
}
#: The VAE's dtype on CUDA. Anima stores Qwen-Image's VAE in bf16, which fp16 holds exactly: against an
#: F32 decode, fp16 measures 62.9 dB and bf16 53.3 (dtype audit M1).
VAE_DTYPE = torch.float16
#: The VAE's RMS norms in one kernel each, with the SiLU each residual block and the decoder's head
#: apply after them; Diffusers computes them in F32 copies of every activation.
VAE_NORM = fusion.ChannelNorm(
    QwenImageRMS_norm,
    silu=((QwenImageResidualBlock, ("norm1", "norm2")), (QwenImageDecoder3d, ("norm_out",))),
    dtype="fp16",
)
#: A lone frame's 3x3x3 causal convolutions on their last temporal tap, the only one it meets
#: (Diffusers pads two zero frames and convolves all 27 taps), and its nearest upsampling in its own
#: dtype, not F32.
VAE_FRAME = fusion.LoneFrame(QwenImageCausalConv3d, upsample=QwenImageUpsample)
#: Anima Base 1.0's transformer, the one the norm and RoPE shapes below are reviewed for: its
#: constructor arguments. The Runtime's config drops the saving library's `_`-keys (#1786), and
#: comparing them refused this plan on every construction since.
_TRANSFORMER_CONFIG = {
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


def apply(pipe: AnimaPipeline) -> tuple[str, str]:
    """Install the plan on a filled pipeline: why the transformer's fused kernel and why the VAE
    run as Diffusers wrote them instead, each "" where its part installed. The block forward,
    with the adapters' seams (se-127), installs wherever the reviewed Diffusers graph is the one
    filled; the fused kernel only where this process runs it."""
    vae = pipe.components["vae"]
    decode = _digest_refusal(_VAE_DIGESTS) or (
        "" if type(vae) is AutoencoderKLQwenImage else f"the VAE is {type(vae).__qualname__}"
    )
    if not decode:
        _unchanged(vae, _install_vae_plan)
    exact = _plan_refusal(pipe)
    fused = exact or _capability()
    if not exact:
        _unchanged(pipe.components["transformer"], partial(_install_transformer_plan, kernel=not fused))
    return fused, decode


def _unchanged[M: torch.nn.Module](module: M, install: Callable[[M], None]) -> None:
    before = _tensor_census(module)
    install(module)
    if _tensor_census(module) != before:
        raise PlanRefusal(
            "anima_optimization_tensor_schema_changed",
            "the post-fill plan changed the tensor schema or parameter/buffer storage identity",
        )


def adaptable(transformer: CosmosTransformer3DModel) -> bool:
    """Whether `apply` installed the block forward that takes adapters on this transformer."""
    return isinstance(getattr(transformer, "_cozy_scope", None), _Scope)


def installed(transformer: CosmosTransformer3DModel) -> bool:
    """Whether `apply` installed the fused kernel on this transformer."""
    block = cast(CosmosTransformerBlock, transformer.transformer_blocks[0])  # its blocks
    return isinstance(block.attn1.processor, _FusedSelfAttention)


class _LatentState(Protocol):
    """The latent step's `BlockState`: the fields it reads and writes."""

    height: int | None
    width: int | None
    latents: torch.Tensor | None
    batch_size: int
    num_images_per_prompt: int
    generator: torch.Generator | list[torch.Generator] | None
    padding_mask: torch.Tensor
    dtype: torch.dtype


class FusedLatents(AnimaPrepareLatentsStep):
    """Diffusers' latent preparation with the padding mask at latent size: the transformer
    resizes it to latent size every step, and resizing zeros gives the same zeros."""

    @torch.no_grad()
    def __call__(
        self, components: AnimaModularPipeline, state: PipelineState
    ) -> tuple[AnimaModularPipeline, PipelineState]:
        block_state = cast(_LatentState, self.get_block_state(state))  # Diffusers annotates its BlockState as dict
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
        self.set_block_state(state, cast(BlockState, block_state))
        return components, state


def _capability() -> str:
    """Why this process cannot run the fused kernels, or "" when it can."""
    cuda = tuple(int(part) for part in (torch.version.cuda or "0.0").split(".")[:2])
    if cuda < CUDA_MIN:
        return f"torch.version.cuda {torch.version.cuda!r} is older than the fused plan's {'.'.join(map(str, CUDA_MIN))}"
    if (placed := device()).type != "cuda":
        return f"this process runs on {placed}, not a CUDA GPU"
    return native()


def _digest_refusal(digests: dict[str, tuple[Reviewed, str]]) -> str:
    for name, (source, expected) in digests.items():
        actual = hashlib.sha256(inspect.getsource(source).encode()).hexdigest()
        if actual != expected:
            return f"{name} has source sha256:{actual}, not the reviewed sha256:{expected}"
    return ""


def _plan_refusal(pipe: AnimaPipeline) -> str:
    """Why this construction is not the reviewed Anima graph, or "" when it is."""
    if refusal := _digest_refusal(_SOURCE_DIGESTS):
        return refusal
    transformer = pipe.components["transformer"]
    if type(transformer) is not CosmosTransformer3DModel:
        return f"the transformer is {type(transformer).__qualname__}, not CosmosTransformer3DModel"
    arguments = {k: v for k, v in dict(transformer.config).items() if not k.startswith("_")}
    if json.loads(json.dumps(arguments)) != _TRANSFORMER_CONFIG:
        return "the filled Cosmos transformer config is not the Anima Base 1.0 config"
    return ""


def _tensor_census(module: torch.nn.Module) -> tuple[tuple[object, ...], tuple[object, ...]]:
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
        self.value: torch.Tensor | None = None

    def clear(self) -> None:
        self.value = None

    def get(self, rotary: Rotary) -> torch.Tensor:
        if self.value is not None:
            return self.value
        if hasattr(rotary, "shape"):
            matrix = cast(torch.Tensor, rotary)  # the direct rotation matrices
            if matrix.ndim != 6 or tuple(matrix.shape[-3:]) != (64, 2, 2):
                raise PlanRefusal(
                    "anima_optimization_rope_layout",
                    f"direct Cosmos rotary matrix has unsupported shape {tuple(matrix.shape)}",
                )
            return matrix
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


def _direct_rope_forward(rope: CosmosRotaryPosEmbed, hidden_states: torch.Tensor, fps: int | None = None) -> torch.Tensor:
    """The fused kernel's rotation matrix directly, without duplicate cosine/sine tensors."""
    _, _, frames, height, width = hidden_states.shape
    sizes = [
        frames // rope.patch_size[0],
        height // rope.patch_size[1],
        width // rope.patch_size[2],
    ]
    device = hidden_states.device
    seq = torch.arange(max(rope.max_size), device=device, dtype=torch.float32)

    def frequencies(dim: int, ntk: float) -> torch.Tensor:
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
    _attention_backend: AttentionBackendName | None = None
    _parallel_config: ParallelConfig | None = None

    def __init__(
        self,
        kernel: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float],
            tuple[torch.Tensor, torch.Tensor],
        ],
        rope: _RopeMatrix,
    ) -> None:
        self.kernel = kernel
        self.rope = rope

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: Rotary | None = None,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """`inputs`: the query's, key's and value's own rows, where an adapter shifted them apart."""
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
        if inputs is None:
            with quantize_once(hidden_states):
                query, key, value = (
                    projection(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
                    for projection in (attn.to_q, attn.to_k, attn.to_v)
                )
        else:
            query, key, value = (
                projection(rows).unflatten(2, (attn.heads, -1)).transpose(1, 2)
                for projection, rows in zip((attn.to_q, attn.to_k, attn.to_v), inputs, strict=True)
            )
        # The kernel reads the scales by pointer, without calling the norms that own them.
        norm_q, norm_k = cast(RMSNorm, attn.norm_q), cast(RMSNorm, attn.norm_k)  # the head-width RMSNorms the plan checked
        query, key = self.kernel(
            query,
            key,
            self.rope.get(image_rotary_emb),
            norm_q.weight,  # type: ignore[arg-type]  # affine, as the plan checked
            norm_k.weight,  # type: ignore[arg-type]  # affine
            norm_q.eps,
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
        out: torch.Tensor = attn.to_out[1](attn.to_out[0](output))
        return out


def _native_rms_forward(module: RMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
    output = torch.nn.functional.rms_norm(hidden_states, module.dim, module.weight, module.eps)
    return output if module.bias is None else output + module.bias


def _hooked(*modules: torch.nn.Module) -> bool:
    return any(m._forward_hooks or m._forward_pre_hooks for m in modules)


def _normed(
    norm: CosmosAdaLayerNormZero,
    hidden: torch.Tensor,
    timestep: torch.Tensor,
    temb: torch.Tensor | None,
    consumers: tuple[torch.nn.Module, ...],
) -> tuple[torch.Tensor | PreQuantized, torch.Tensor]:
    """`CosmosAdaLayerNormZero.forward`'s rows and gate. When every consumer is an int8 leaf the
    rows come quantized from the norm's one pass (`int8_rows`), else as Diffusers makes them."""
    table = norm.linear_2(norm.linear_1(norm.activation(timestep)))
    if temb is not None:
        table = table + temb
    shift, scale, gate = table.chunk(3, dim=-1)
    if table.ndim == 2:
        shift, scale, gate = (t.unsqueeze(1) for t in (shift, scale, gate))
    rows: torch.Tensor | PreQuantized | None = int8_rows(hidden, consumers, norm="layer", eps=norm.norm.eps, scale=scale, shift=shift)
    if rows is None and table.ndim == 2 and table.shape[0] == 1:
        # One modulation row for the batch (both CFG halves share the timestep): it is the
        # LayerNorm's own affine, one pass and one rounding instead of three.
        width = norm.norm.normalized_shape
        weight, bias = (1 + scale).reshape(width), shift.reshape(width)
        rows = F.layer_norm(hidden, width, weight, bias, norm.norm.eps)
    elif rows is None:
        rows = norm.norm(hidden) * (1 + scale) + shift
    return rows, gate


def _feed_forward(ff: FeedForward, rows: torch.Tensor | PreQuantized) -> torch.Tensor:
    """`FeedForward.forward` for Cosmos' GELU MLP, its GELU folded into the down projection's
    int8 quantization when that is an int8 leaf."""
    gelu = cast(GELU, ff.net[0])  # Cosmos' GELU MLP, as the plan checked it
    up = gelu.proj(rows)
    activation = "gelu" if gelu.approximate == "none" else "gelu_tanh"
    down = int8_rows(up, (ff.net[2],), activation=activation)
    out: torch.Tensor = ff.net[2](ff.net[1](gelu.gelu(up)) if down is None else down)
    return out


class _Scope:
    """One DiT call's adapters (se-127), set by the plan's transformer forward for its blocks."""

    def __init__(self) -> None:
        self.adapters: Adapters | None = None


def _attention(
    attn: Attention,
    rows: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rotary: Rotary | None,
    mask: torch.Tensor | None,
    lora: torch.nn.Module | None = None,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`CosmosAttnProcessor2_0.__call__` on the query's, key's and value's own rows, with an adapter's LoRA on
    the four projections (`lora.q_proj`, …, `lora.output_proj`): the output, and the normalized query an
    IP-Adapter attends from. Cosmos' heads are all full, so Diffusers' K/V repeat is the identity."""
    projections = (attn.to_q, attn.to_k, attn.to_v)
    sites = ("q_proj", "k_proj", "v_proj")
    query, key, value = (
        (projection(x) if lora is None else projection(x) + scale * lora.get_submodule(site)(x))
        .unflatten(2, (attn.heads, -1))
        .transpose(1, 2)
        for projection, site, x in zip(projections, sites, rows, strict=True)
    )
    query, key = attn.norm_q(query), attn.norm_k(key)
    if rotary is not None:
        query = apply_rotary_emb(query, rotary, use_real=True, use_real_unbind_dim=-2)
        key = apply_rotary_emb(key, rotary, use_real=True, use_real_unbind_dim=-2)
    output = dispatch_attention_fn(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )
    output = output.flatten(2, 3).type_as(query)
    projected = attn.to_out[0](output)
    if lora is not None:
        projected = projected + scale * lora.get_submodule("output_proj")(output)
    out: torch.Tensor = attn.to_out[1](projected)
    return out, query


def _adapted(
    block: CosmosTransformerBlock,
    adapters: Adapters,
    index: int,
    normalized: torch.Tensor,
    rotary: Rotary | None,
) -> torch.Tensor:
    """The block's self-attention with each projection's input shifted by the control maps' LLLite."""
    rows = cast(
        "tuple[torch.Tensor, torch.Tensor, torch.Tensor]",
        tuple(adapters.shift(index, f"self_attn_{p}_proj", normalized) for p in "qkv"),
    )
    attn1 = block.attn1
    if isinstance(attn1.processor, _FusedSelfAttention):
        out: torch.Tensor = attn1(rows[0], image_rotary_emb=rotary, inputs=rows)
        return out
    return _attention(attn1, rows, rotary, None)[0]


def _crossed(
    block: CosmosTransformerBlock,
    adapters: Adapters,
    index: int,
    normalized: torch.Tensor,
    text: torch.Tensor,
    mask: torch.Tensor | None,
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The block's cross-attention under the references' LoRA, and their IP term for the conditional rows."""
    references = adapters.references
    query = adapters.shift(index, "cross_attn_q_proj", normalized)
    lora = None if references is None else references.cross(index)
    scale = 1.0 if references is None else references.adapter.scale
    out, normed = _attention(block.attn2, (query, text, text), None, mask, lora, scale)
    if references is None or not any(adapters.rows):
        return out, None
    if all(adapters.rows):
        return out, references.term(index, normed, timestep)
    chosen = torch.tensor([i for i, c in enumerate(adapters.rows) if c], device=normed.device)
    if timestep.shape[0] == normed.shape[0]:
        timestep = timestep.index_select(0, chosen)
    return out, references.term(index, normed.index_select(0, chosen), timestep)


def _block_forward(
    block: CosmosTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    embedded_timestep: torch.Tensor,
    temb: torch.Tensor | None = None,
    image_rotary_emb: Rotary | None = None,
    extra_pos_emb: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    controlnet_residual: torch.Tensor | None = None,
    latents: torch.Tensor | None = None,
    block_idx: int | None = None,
    *,
    index: int,
    scope: _Scope,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """`CosmosTransformerBlock.forward` with each gated residual as one `addcmul`, block `index` of
    the DiT, and the adapters of `scope`'s call: LLLite shifts on its linears' inputs, the
    references' LoRA on its cross-attention and their IP term after it, on the conditional rows."""
    if not isinstance(hidden_states, torch.Tensor):
        raise PlanRefusal("anima_optimization_tensor_type", "Cosmos received a non-torch tensor")
    if block.before_proj is not None:
        hidden_states = block.before_proj(hidden_states) + latents
    if extra_pos_emb is not None:
        hidden_states = hidden_states + extra_pos_emb
    adapters = scope.adapters
    attn1, attn2, ff = block.attn1, block.attn2, block.ff
    # Rows quantized in the norm's pass reach the projections through modules nothing hooks or shifts.
    fused = adapters is None and not _hooked(attn1, attn2, ff, *ff.net)
    consumers: tuple[torch.nn.Module, ...] = (attn1.to_q, attn1.to_k, attn1.to_v) if fused else ()
    normalized, gate = _normed(block.norm1, hidden_states, embedded_timestep, temb, consumers)
    if adapters is None:
        output = attn1(normalized, image_rotary_emb=image_rotary_emb)
    else:
        output = _adapted(block, adapters, index, cast(torch.Tensor, normalized), image_rotary_emb)
    hidden_states = torch.addcmul(hidden_states, gate, output)
    consumers = (attn2.to_q,) if fused else ()
    normalized, gate = _normed(block.norm2, hidden_states, embedded_timestep, temb, consumers)
    term = None
    if adapters is None:
        with quantize_once(encoder_hidden_states):
            output = attn2(
                normalized, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
            )
    else:
        rows = cast(torch.Tensor, normalized)
        output, term = _crossed(
            block, adapters, index, rows, encoder_hidden_states, attention_mask, embedded_timestep
        )
    hidden_states = torch.addcmul(hidden_states, gate, output)
    consumers = (cast(GELU, ff.net[0]).proj,) if fused else ()
    normalized, gate = _normed(block.norm3, hidden_states, embedded_timestep, temb, consumers)
    if adapters is not None:
        normalized = adapters.shift(index, "mlp_layer1", cast(torch.Tensor, normalized))
    output = _feed_forward(ff, normalized) if fused else ff(normalized)
    hidden_states = torch.addcmul(hidden_states, gate, output)
    if term is not None:
        # The IP-Adapter's term is trained as a residual on the whole block's output.
        assert adapters is not None
        if all(adapters.rows):
            hidden_states = hidden_states + term.to(hidden_states.dtype)
        else:
            chosen = torch.tensor([i for i, c in enumerate(adapters.rows) if c], device=hidden_states.device)
            hidden_states = hidden_states.index_add(0, chosen, term.to(hidden_states.dtype))
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


def _install_transformer_plan(transformer: CosmosTransformer3DModel, *, kernel: bool) -> None:
    """The plan's forwards; with `kernel`, the fused self-attention kernel and its rotary table."""
    rope = _RopeMatrix()
    scope = _Scope()
    blocks = cast("tuple[CosmosTransformerBlock, ...]", tuple(transformer.transformer_blocks))  # checked below
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
        norm.forward = MethodType(_native_rms_forward, norm)  # type: ignore[method-assign]  # the plan's forward
    for index, block in enumerate(blocks):
        if (
            block.before_proj is not None
            or block.after_proj is not None
            # Diffusers types a processor as its base class; any is set.
            or type(cast(object, block.attn1.processor)) is not CosmosAttnProcessor2_0
            or type(cast(object, block.attn2.processor)) is not CosmosAttnProcessor2_0
        ):
            raise PlanRefusal(
                "anima_optimization_block_shape",
                "Anima Cosmos block projections or attention processors do not match the plan",
            )
        ada = (block.norm1, block.norm2, block.norm3)
        if (
            any(type(n) is not CosmosAdaLayerNormZero for n in ada)
            or any(n.norm.weight is not None or n.norm.bias is not None for n in ada)  # type: ignore[unreachable]  # torch types an absent affine as a tensor
            or len(block.ff.net) != 3
            or type(block.ff.net[0]) is not GELU
            or type(block.ff.net[1]) is not torch.nn.Dropout
        ):
            raise PlanRefusal(
                "anima_optimization_block_shape",
                "Anima Cosmos block norms or feed-forward do not match the plan",
            )
        if kernel:
            block.attn1.set_processor(_FusedSelfAttention(rms_rope_split_half, rope))  # type: ignore[arg-type]  # the plan's own processor
        block.forward = MethodType(partial(_block_forward, index=index, scope=scope), block)  # type: ignore[method-assign]  # the plan's forward
    if kernel:
        transformer.rope.forward = MethodType(_direct_rope_forward, transformer.rope)  # type: ignore[method-assign]  # the plan's
    transformer._cozy_scope = scope  # type: ignore[assignment]  # a plain attribute: nn.Module keeps non-tensors as such
    original = transformer.forward

    def forward(
        _self: CosmosTransformer3DModel, *args: object, adapters: Adapters | None = None, **kwargs: object
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        """Diffusers' forward; `adapters` reach the blocks. A call with them is never a recorded graph: the
        Runtime keys a graph on plain arguments only and runs such a call as written."""
        rope.clear()
        scope.adapters = adapters
        try:
            out: tuple[torch.Tensor] | Transformer2DModelOutput = original(*args, **kwargs)  # type: ignore[arg-type]  # its own arguments, passed on
            return out
        finally:
            rope.clear()
            scope.adapters = None

    transformer.forward = MethodType(forward, transformer)  # type: ignore[method-assign]  # the plan's forward


def _install_vae_plan(vae: AutoencoderKLQwenImage) -> None:
    vae._decode = MethodType(_decode_frame, vae)  # type: ignore[method-assign]  # the plan's decode


def _decode_frame(
    vae: AutoencoderKLQwenImage, z: torch.Tensor, return_dict: bool = True
) -> DecoderOutput | tuple[torch.Tensor]:
    """`AutoencoderKLQwenImage._decode` for one latent frame (`_decoded`)."""
    if z.shape[2] != 1 or vae.use_tiling:
        decoded: DecoderOutput | tuple[torch.Tensor] = type(vae)._decode(vae, z, return_dict=return_dict)
        return decoded
    out = _decoded(vae, z).clamp(-1.0, 1.0)
    return DecoderOutput(sample=out) if return_dict else (out,)


def _decoded(vae: AutoencoderKLQwenImage, z: torch.Tensor) -> torch.Tensor:
    """The decode of one frame, the whole canvas: Diffusers' without its feature cache (each
    convolution's input kept for a next frame an image never has), every residual and resample
    block over the canvas in strips of rows (`tiles.rows`: exact, a strip's working set)."""
    decoder = vae.decoder
    x = decoder.conv_in(vae.post_quant_conv(z))
    mid = decoder.mid_block
    x = tiles.rows(mid.resnets[0], x, halo=2)
    for attention, resnet in zip(mid.attentions, mid.resnets[1:], strict=True):
        x = tiles.rows(resnet, attention(x), halo=2)
    for block in decoder.up_blocks:
        up = cast(QwenImageUpBlock, block)  # its up blocks
        for resnet in up.resnets:
            x = tiles.rows(resnet, x, halo=2)
        if up.upsamplers is not None:
            x = tiles.rows(up.upsamplers[0], x, halo=1, scale=2)
    return tiles.rows(
        lambda s: decoder.conv_out(decoder.nonlinearity(decoder.norm_out(s))), x, halo=1
    )
