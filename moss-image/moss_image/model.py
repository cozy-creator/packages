"""MOSS-Image: a 5B single-stream DiT over Qwen3.5-9B's last hidden state, decoded by FLUX.2's VAE.

The DiT, its timestep grid and the sampling loop follow upstream's `pipeline_moss_image.py`
(OpenMOSS-Team/MOSS-Image-SFT `f972bd4b`, Apache-2.0) module for module, with its state-dict names;
Diffusers has no MOSS-Image. The text encoder is Transformers' Qwen3.5 text model and the VAE Diffusers'
`AutoencoderKLFlux2`. Runtime constructs and fills the modules and owns their residency; the denoise is
one `denoise_loop`.

On a GPU a DiT block runs as ComfyUI's MOSS-Image does, with the Runtime's public kernels: each
modulated RMSNorm comes out of the int8 leaves' quantization pass (`int8_rows`, the norm weight folded
into `1 + scale`), the q/k norms and split-half RoPE are one kernel (`rms_rope_split_half`), SiTU is
this package's (`glue`) and each gated residual one in-place `addcmul`. The int8 lane's DiT blocks
attend at 8 bits, as Comfy-Org's int8 file declares (`comfy_attention.config`); the bf16 lane stays
exact. On the CPU every block runs upstream's eager composition.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import torch
from cozy_runtime import rng
from cozy_runtime.attention import Attention
from cozy_runtime.author import (
    Config,
    Loader,
    MemoDistribution,
    Model,
    ModelAssets,
    Telemetry,
    denoise_loop,
    hold_dtype,
    uses_components,
)
from cozy_runtime.author.kernels import Launch, Plan, int8_rows, native, quantize_once, rms_rope_split_half
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import register_to_config
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, Qwen3_5Config
from transformers.initialization import no_init_weights
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from . import glue

#: The DiT's activations and the dtype upstream casts each linear's input to (its weights' in bf16).
DTYPE = torch.bfloat16
EPS = 1e-6
MAX_TOKENS = 4096
#: Pixels per DiT token: the VAE's 8x, then 2x2 patches.
PATCH = 16
#: Upstream's default guidance: the negative branch skips this block (zero-based).
UNCOND_SKIP = frozenset({22})


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = EPS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def situ(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def rotate(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Split-half RoPE of `x` [B, N, heads, D] by the `[1, N, D]` sin|cos table."""
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def attention(
    site: Attention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rope: torch.Tensor
) -> torch.Tensor:
    shape = (*q.shape[:2], site.heads, site.head_dim)
    return site(rotate(q.reshape(shape), rope), rotate(k.reshape(shape), rope), v.reshape(shape)).flatten(2)


def modulated(x: torch.Tensor, norm: RMSNorm, scale: torch.Tensor, shift: torch.Tensor,
              consumers: tuple[Any, ...]) -> Any:
    """`norm(x) * (1 + scale) + shift` for `consumers`: quantized for them in the norm's one pass when
    they are int8 leaves (the norm weight folded into the scale, as ComfyUI's `rms_adaln`), else as
    upstream computes it."""
    folded = (norm.weight.float() * (1 + scale.float()) - 1).to(x.dtype)
    rows = int8_rows(x, consumers, norm="rms", eps=norm.eps, scale=folded, shift=shift)
    if rows is None:
        rows = (norm(x) * (1 + scale[:, None]) + shift[:, None]).to(DTYPE)
    return rows


class TextBlock(torch.nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int) -> None:
        super().__init__()
        self.heads, self.attn = heads, Attention(heads, dim // heads, DTYPE)
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = torch.nn.Linear(dim, 3 * dim)
        self.proj = torch.nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = torch.nn.Linear(dim, ffn_dim), torch.nn.Linear(dim, ffn_dim)
        self.ffn_down = torch.nn.Linear(ffn_dim, dim)

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(attention(self.attn, q, k, v, rope))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextProjection(torch.nn.Module):
    def __init__(self, text_dim: int, dim: int, ffn_dim: int, heads: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(text_dim, dim)
        self.blocks = torch.nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        x = self.proj(x.to(DTYPE))
        for block in self.blocks:
            x = block(x, rope)
        return x


class Block(torch.nn.Module):
    """One DiT block over one sample's stream `x` [1, N, dim] (bf16), updated in place. `rope` is the
    `[1, N, 1, D/2, 2, 2]` rotation matrices on the fused path, the `[1, N, D]` sin|cos table eager."""

    def __init__(self, dim: int, ffn_dim: int, heads: int) -> None:
        super().__init__()
        self.heads, self.attn = heads, Attention(heads, dim // heads, DTYPE)
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = torch.nn.ModuleDict(
            {name: torch.nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")}
        )
        self.proj["ffn_up_gate"] = torch.nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = torch.nn.Linear(ffn_dim, dim, bias=False)
        #: The package's SiTU kernel once it is built here (`glue.install`); eager until then.
        self.situ_launch: Launch | None = None

    def forward(self, x: torch.Tensor, rope: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        proj = self.proj
        q, k, v = proj["q"], proj["k"], proj["v"]
        rows = modulated(x, self.norm1, scale_a, shift_a, (q, k, v))
        with quantize_once(rows):
            q, k, v = (p(rows).unflatten(-1, (self.heads, -1)) for p in (q, k, v))
        if rope.dim() == 6:
            q, k = rms_rope_split_half(q, k, rope, self.q_norm.weight, self.k_norm.weight, self.q_norm.eps)
        else:
            q, k = rotate(self.q_norm(q), rope), rotate(self.k_norm(k), rope)
        x.addcmul_(proj["o"](self.attn(q, k, v).flatten(2)), gate_a[:, None])
        hidden = proj["ffn_up_gate"](modulated(x, self.norm2, scale_f, shift_f, (proj["ffn_up_gate"],)))
        activated = None if self.situ_launch is None else glue.situ(self.situ_launch, hidden)
        if activated is None:
            activated = situ(*hidden.chunk(2, -1))
        return x.addcmul_(proj["ffn_down"](activated), gate_f[:, None])


class MossTransformer(torch.nn.Module):
    """Upstream's `MOSSImageTransformer` for one sample: image tokens then text tokens in one stream,
    one shared modulation for every block. Its RoPE bands are derived constants, computed in f32 on the
    CPU as training built them; the checkpoint's `rope.bands` is not read. The denoise composes its
    parts: the text stream and RoPE once a request, the modulation, blocks and head each step."""

    def __init__(self, dim: int = 3072, ffn_dim: int = 8192, num_heads: int = 24, num_layers: int = 42,
                 text_dim: int = 4096, freq_dim: int = 256, latent_channels: int = 128,
                 rope_base: float = 10000.0) -> None:
        super().__init__()
        self.heads, self.freq_dim, self.latent_channels = num_heads, freq_dim, latent_channels
        self.rope_base, self.num_bands = rope_base, (dim // num_heads) // 8
        self.video_embedder = torch.nn.Linear(latent_channels, dim)
        self.text_embedder = TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(freq_dim, dim), torch.nn.SiLU(), torch.nn.Linear(dim, dim)
        )
        self.shared_block_modulation = torch.nn.Linear(dim, 6 * dim)
        self.layers = torch.nn.ModuleList([Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = torch.nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = torch.nn.Linear(dim, 2 * dim)

    def phase(self, positions: torch.Tensor) -> torch.Tensor:
        """RoPE angles [N, D/2] of (r, f, h, w) `positions` [N, 4]: axis-major, 16 bands an axis."""
        bands = 1 / (self.rope_base ** (torch.arange(self.num_bands, dtype=torch.float32, device="cpu") / self.num_bands))
        return (positions.unsqueeze(-1) * bands.to(positions.device)).flatten(-2)

    def positions(self, height: int, width: int, text_length: int, device: torch.device) -> torch.Tensor:
        """(r, f, h, w) per token: the image centred on its grid with f = the text length, then the text
        counting up on f."""
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        return torch.cat((image, self.text_positions(text_length, device)), 0)

    @staticmethod
    def text_positions(text_length: int, device: torch.device) -> torch.Tensor:
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        return text

    @staticmethod
    def table(phase: torch.Tensor) -> torch.Tensor:
        """The eager sin|cos table [1, N, D] in bf16, as upstream builds it."""
        return torch.cat((phase.sin(), phase.cos()), -1).unsqueeze(0).to(DTYPE)

    @staticmethod
    def rotations(phase: torch.Tensor) -> torch.Tensor:
        """The fused kernel's f32 rotation matrices [1, N, 1, D/2, 2, 2]."""
        cos, sin = phase.cos(), phase.sin()
        return torch.stack((cos, -sin, sin, cos), -1).unflatten(-1, (2, 2)).unsqueeze(0).unsqueeze(2)

    def text(self, context: torch.Tensor) -> torch.Tensor:
        """The step-invariant text stream [1, S, dim] of `context` [1, S, text_dim]."""
        phase = self.phase(self.text_positions(context.shape[1], context.device))
        return self.text_embedder(context, self.table(phase))  # type: ignore[no-any-return]

    def modulation(self, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The blocks' shared modulation [1, 6, dim] and the head's [1, 2, dim] at `timestep` [1] (0-1000).
        Training computes the sinusoid in f64 and feeds f32 into its MLP."""
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=timestep.device, dtype=torch.float64) / half)
        phase = timestep.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(DTYPE))
        return self.shared_block_modulation(time).view(1, 6, -1), self.video_head_modulation(time).view(1, 2, -1)

    def velocity(self, x: torch.Tensor, head: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """The f32 velocity [1, C, H, W] from the stream's image tokens."""
        x = self.final_norm_video(x[:, : height * width].to(DTYPE))
        x = self.final_linear_video((x * (1 + head[:, 1, None]) + head[:, 0, None]).to(DTYPE))
        return x.transpose(1, 2).reshape(1, -1, height, width).float()


class MossScheduler(FlowMatchEulerDiscreteScheduler):
    """Upstream's `MOSSImageScheduler`: Euler over sigmas shifted by `shift`, then snapped to the
    nearest of the 1000 training sigmas."""

    @register_to_config  # type: ignore[untyped-decorator]
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, training_shift: float = 1.0) -> None:
        super().__init__(num_train_timesteps=num_train_timesteps, shift=1.0)
        self.register_to_config(shift=shift, training_shift=training_shift)

    def set_timesteps(self, num_inference_steps: int = 50, device: Any = None) -> None:  # type: ignore[override]
        total = self.config.num_train_timesteps

        def grid(count: int, shift: float) -> torch.Tensor:
            lower = shift / (total + shift - 1)
            values = torch.linspace(1, lower, count, device="cpu")
            return shift * values / (1 + (shift - 1) * values)

        sigmas = grid(num_inference_steps, self.config.shift)
        training = grid(total, self.config.training_shift).flip(0)
        upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
        lower = (upper - 1).clamp(min=0)
        nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
        sigmas = torch.unique_consecutive(training[nearest])
        # Already shifted and snapped: no second shift.
        self._shift = 1.0
        super().set_timesteps(device=device, sigmas=sigmas.tolist())


class TextEncoder(torch.nn.Module):
    """Qwen3.5-9B's text model (`language_model.*`). The vision tower is never built, so its bytes are
    never read."""

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.language_model = Qwen3_5TextModel._from_config(config.text_config, dtype=DTYPE)


class MossGraph:
    def __init__(self, config: Config) -> None:
        with no_init_weights():
            self.components: dict[str, torch.nn.Module] = {
                "transformer": MossTransformer(**config.section("transformer").mapping()).to(DTYPE).eval(),
                "text_encoder": TextEncoder(Qwen3_5Config(**config.section("text_encoder").mapping())).eval(),
                "vae": AutoencoderKLFlux2.from_config(config.section("vae").mapping()).to(torch.float32).eval(),
            }
        self.scheduler_config = config.section("scheduler").mapping()


def build_tokenizer(assets: ModelAssets) -> PreTrainedTokenizerFast:
    """Qwen3.5's tokenizer and chat template from the checkpoint's `tokenizer/` files, parsed in memory."""
    config = json.loads(assets.read("tokenizer/tokenizer_config.json"))
    for key in ("tokenizer_file", "name_or_path", "added_tokens_decoder", "tokenizer_class", "processor_class",
                "backend", "is_local", "local_files_only", "chat_template", "model_specific_special_tokens"):
        config.pop(key, None)
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(assets.read("tokenizer/tokenizer.json").decode()),
        chat_template=assets.read("tokenizer/chat_template.jinja").decode(),
        **config,
    )


#: Code the memoized encode reads beyond its own body.
_CODE = (build_tokenizer, TextEncoder, MemoDistribution("transformers"), MemoDistribution("tokenizers"))


class MossImage(Model[MossGraph], encoded_leaves="accept", attention_bits={"transformer.layers.": 8}):
    """One MOSS-Image checkpoint (Base or SFT)."""

    graph: MossGraph
    tokenizer: PreTrainedTokenizerFast

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(MossGraph, factory=MossGraph)
        self.tokenizer = build_tokenizer(loader.assets)

    def execution_plan(self, plan: Plan) -> str:
        return glue.install(plan, list(self.graph.components["transformer"].layers))

    def hold_vae(self, tel: Telemetry) -> None:
        """Decode in bf16 where the GPU computes it natively, as ComfyUI stages FLUX.2's VAE; the
        checkpoint stores F32, which the CPU and other GPUs keep. The Runtime casts the stored bytes
        when it next fills them; called before any VAE scope opens."""
        vae = self.graph.components["vae"]
        device = torch.device(str(self.device))
        if device.type != "cuda" or not torch.cuda.is_bf16_supported(including_emulation=False):
            return
        if next(vae.parameters()).dtype != DTYPE:
            hold_dtype(vae, DTYPE)
            tel.log("vae representation", dtype="bfloat16")

    @uses_components("text_encoder", memoize=True, memo_dependencies=_CODE)
    def encode(self, prompt: str) -> torch.Tensor:
        """The prompt's last hidden state [S, 4096] (bf16, on the CPU): the user turn of the chat
        template with thinking open, at most 4096 tokens, one unpadded forward. One prompt a call, so
        a request's fixed negative is a memo hit."""
        encoder = self.graph.components["text_encoder"]
        device = torch.device(str(self.device))
        with torch.inference_mode():
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
            out = encoder.language_model(
                input_ids=inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device),
                use_cache=False,
            )
            return out.last_hidden_state[0].cpu()  # type: ignore[no-any-return]

    @uses_components("transformer")
    def denoise(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor | None,
        *,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        on_step: Callable[[int], None],
    ) -> torch.Tensor:
        """Upstream's sampler as one `denoise_loop`: f32 noise from `seed` on the CPU, CFG against
        `negative` with block 22 skipped in its branch, Euler in f32. Returns the latents
        [1, 128, H/16, W/16] on the CPU.

        Each prompt's text stream and RoPE are built once, before the loop; with CFG both branches
        pass through each block in turn, so a streamed block comes in once per step. Their text
        lengths differ, so the branches are two streams, not one batch (ComfyUI runs two forwards)."""
        transformer = self.graph.components["transformer"]
        device = torch.device(str(self.device))
        rows, columns = height // PATCH, width // PATCH
        fused = device.type == "cuda" and not native()
        scheduler = MossScheduler.from_config(self.graph.scheduler_config)
        scheduler.set_timesteps(steps, device=device)
        with torch.inference_mode(False), torch.no_grad():
            latents = rng.noise((1, transformer.latent_channels, rows, columns), seed, device=device)
            branches: list[torch.Tensor] = []
            for text in (positive, *([] if negative is None else [negative])):
                context = text[None].to(device)
                phase = transformer.phase(transformer.positions(rows, columns, context.shape[1], device))
                branches += [transformer.text(context), transformer.rotations(phase) if fused else transformer.table(phase)]
            times = scheduler.timesteps.to(device=device, dtype=torch.float32)
            deltas = (scheduler.sigmas[1:] - scheduler.sigmas[:-1]).to(device=device, dtype=torch.float32)
        blocks = transformer.layers

        def step(i: torch.Tensor, latents: torch.Tensor, times: torch.Tensor, deltas: torch.Tensor,
                 *branches: torch.Tensor) -> None:
            modulation, head = transformer.modulation(times.index_select(0, i))
            video = transformer.video_embedder(latents.flatten(2).transpose(1, 2).to(DTYPE).contiguous())
            streams = [torch.cat((video, text), 1) for text in branches[0::2]]
            ropes = branches[1::2]
            for index, block in enumerate(blocks):
                streams = [x if branch and index in UNCOND_SKIP else block(x, rope, modulation)
                           for branch, (x, rope) in enumerate(zip(streams, ropes, strict=True))]
            velocity = [transformer.velocity(x, head, rows, columns) for x in streams]
            prediction = velocity[0]
            if len(velocity) > 1:
                prediction = velocity[1] + guidance * (velocity[0] - velocity[1])
            latents.add_(deltas.index_select(0, i) * prediction)

        denoise_loop(step, range(len(times)), state=(latents,), inputs=(times, deltas, *branches), on_step=on_step)
        return latents.cpu()

    @uses_components("vae")
    def decode(self, latents: torch.Tensor) -> Image.Image:
        """The latents' BatchNorm statistics restored in f32, 2x2 patches unfolded, then the small
        decoder in the VAE's dtype."""
        vae = self.graph.components["vae"]
        with torch.inference_mode():
            z = latents.to(str(self.device), torch.float32)
            bn, eps = vae.bn, vae.config.batch_norm_eps
            std = (bn.running_var.float() + eps).sqrt().view(1, -1, 1, 1)
            z = z * std.to(z.device) + bn.running_mean.float().to(z.device).view(1, -1, 1, 1)
            b, c, h, w = z.shape
            z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
            image = vae.decode(z.to(vae.dtype), return_dict=False)[0].clamp(-1, 1)
            if not torch.isfinite(image).all():
                raise ValueError("MOSS-Image's decoder produced non-finite pixels")
            (picture,) = VaeImageProcessor(vae_scale_factor=PATCH).postprocess(image.float(), output_type="pil")
        return picture  # type: ignore[no-any-return]
