"""MOSS-Image: a 5B single-stream DiT over Qwen3.5-9B's last hidden state, decoded by FLUX.2's VAE.

The DiT, its timestep grid and the sampling loop follow upstream's `pipeline_moss_image.py`
(OpenMOSS-Team/MOSS-Image-SFT `f972bd4b`, Apache-2.0) module for module, with its state-dict names and
operation order; Diffusers has no MOSS-Image. The text encoder is Transformers' Qwen3.5 text model
and the VAE Diffusers' `AutoencoderKLFlux2`. Runtime constructs and fills the modules and owns their
residency: the text encoder's and VAE's results are handed to the transformer's scope.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import torch
from cozy_runtime import rng
from cozy_runtime.attention import Attention
from cozy_runtime.author import Config, Loader, MemoDistribution, Model, ModelAssets, uses_components
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import register_to_config
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, Qwen3_5Config
from transformers.initialization import no_init_weights
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

#: The DiT's activations and the dtype upstream casts each linear's input to (its weights' in bf16).
DTYPE = torch.bfloat16
MAX_TOKENS = 4096
#: Upstream's default guidance: the negative branch skips this block (zero-based), still two forwards.
UNCOND_SKIP = frozenset({22})


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def situ(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def rotate(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, rope: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = (self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]).to(DTYPE)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](attention(self.attn, q, k, self.proj["v"](y), rope)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(DTYPE)).chunk(2, -1)
        return x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f[:, None]


class MossTransformer(torch.nn.Module):
    """Upstream's `MOSSImageTransformer` for one sample: image tokens then text tokens in one stream,
    one shared modulation for every block. Its RoPE bands are derived constants, computed in f32 on the
    CPU as training built them, at each call; the checkpoint's `rope.bands` is not read."""

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

    def rope(self, height: int, width: int, text_length: int, device: torch.device) -> torch.Tensor:
        """(r, f, h, w) per token: the image centred on its grid with f = the text length, then the text
        counting up on f."""
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        bands = 1 / (self.rope_base ** (torch.arange(self.num_bands, dtype=torch.float32, device="cpu") / self.num_bands))
        phase = torch.cat((image, text), 0).unsqueeze(-1) * bands.to(device)
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)

    def forward(self, latents: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor,
                skip: frozenset[int] = frozenset()) -> torch.Tensor:
        """The velocity for `latents` [1, C, H, W] at `timestep` (0-1000) given `context` [1, S, text_dim]."""
        _, _, height, width = latents.shape
        times = torch.as_tensor(timestep, device=latents.device, dtype=torch.float32).reshape(-1)
        # Training computes the sinusoid in f64 and feeds f32 into its MLP.
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(DTYPE)).float()
        mod = self.shared_block_modulation(time.to(DTYPE)).view(1, 6, -1)
        head = self.video_head_modulation(time.to(DTYPE)).view(1, 2, -1)
        video = self.video_embedder(latents.flatten(2).transpose(1, 2).to(DTYPE).contiguous())
        rope = self.rope(height, width, context.shape[1], latents.device).to(video.dtype)
        text = self.text_embedder(context, rope[:, height * width :])
        x = torch.cat((video, text), 1)
        for index, layer in enumerate(self.layers):
            if index not in skip:
                x = layer(x, rope, mod)
        x = self.final_norm_video(x[:, : height * width].to(DTYPE))
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x.to(DTYPE))
        return x.transpose(1, 2).reshape(1, -1, height, width)


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


def _section(mapping: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = mapping.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"MOSS-Image {name} configuration must be a mapping")
    return {key: item for key, item in value.items() if not key.startswith("_")}


class MossGraph:
    def __init__(self, config: Config) -> None:
        mapping = config.mapping()
        with no_init_weights():
            self.components: dict[str, torch.nn.Module] = {
                "transformer": MossTransformer(**_section(mapping, "transformer")).to(DTYPE).eval(),
                "text_encoder": TextEncoder(Qwen3_5Config(**_section(mapping, "text_encoder"))).eval(),
                "vae": AutoencoderKLFlux2.from_config(_section(mapping, "vae")).to(torch.float32).eval(),
            }
        self.scheduler_config = _section(mapping, "scheduler")


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


class MossImage(Model[MossGraph], encoded_leaves="accept"):
    """One MOSS-Image checkpoint (Base or SFT)."""

    graph: MossGraph
    tokenizer: PreTrainedTokenizerFast

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(MossGraph, factory=MossGraph)
        self.tokenizer = build_tokenizer(loader.assets)

    @uses_components("text_encoder", memoize=True, memo_dependencies=_CODE)
    def encode(self, prompts: list[str]) -> list[torch.Tensor]:
        """Each prompt's last hidden state [S, 4096] (bf16, on the CPU): the user turn of the chat
        template with thinking open, at most 4096 tokens, one unpadded forward each."""
        encoder = self.graph.components["text_encoder"]
        device = torch.device(str(self.device))
        rows = []
        with torch.inference_mode():
            for prompt in prompts:
                text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                    enable_thinking=True,
                )
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
                out = encoder.language_model(
                    input_ids=inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device),
                    use_cache=False,
                )
                rows.append(out.last_hidden_state[0].cpu())
        return rows

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
        cancel: Callable[[], None],
    ) -> torch.Tensor:
        """Upstream's sampler: f32 noise from `seed` on the CPU, CFG against `negative` with block 22
        skipped in its branch, Euler in f32. Returns the latents [1, 128, H/16, W/16] on the CPU."""
        transformer = self.graph.components["transformer"]
        device = torch.device(str(self.device))
        scheduler = MossScheduler.from_config(self.graph.scheduler_config)
        latents = rng.noise((1, transformer.latent_channels, height // 16, width // 16), seed, device=device)
        context = positive[None].to(device)
        uncond = None if negative is None else negative[None].to(device)
        scheduler.set_timesteps(steps, device=device)
        with torch.inference_mode():
            for index, timestep in enumerate(scheduler.timesteps):
                cancel()
                prediction = transformer(latents, timestep, context).float()
                if uncond is not None:
                    unconditional = transformer(latents, timestep, uncond, UNCOND_SKIP).float()
                    prediction = unconditional + guidance * (prediction - unconditional)
                latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0]
                on_step(index)
        return latents.cpu()

    @uses_components("vae")
    def decode(self, latents: torch.Tensor) -> Image.Image:
        """The latents' BatchNorm statistics restored, 2x2 patches unfolded, then the small decoder."""
        vae = self.graph.components["vae"]
        with torch.inference_mode():
            z = latents.to(str(self.device), vae.dtype)
            bn, eps = vae.bn, vae.config.batch_norm_eps
            z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
            b, c, h, w = z.shape
            z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
            image = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
            if not torch.isfinite(image).all():
                raise ValueError("MOSS-Image's decoder produced non-finite pixels")
            (picture,) = VaeImageProcessor(vae_scale_factor=16).postprocess(image, output_type="pil")
        return picture  # type: ignore[no-any-return]
