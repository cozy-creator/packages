"""SDXL's numerics outside Diffusers' pipeline: CLIP conditioning, the guided UNet call, VAE decode."""

from __future__ import annotations

from typing import Any

import torch
from cozy_runtime.accel import hold_dtype
from cozy_runtime.author import Cut, Telemetry, split
from transformers.masking_utils import create_causal_mask


def _clip_text(model: Any, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CLIP's tensor orchestration, with an explicit penultimate result.

    Transformers 5.16 installs Python output-collection hooks when hidden states are
    requested. Keep its causal-mask, layer, normalization and EOS-pooling operations,
    calling the original numerical modules.
    The real parity gate compares this path with that pinned library forward.
    """
    ids = ids.view(-1, ids.shape[-1])
    hidden = model.embeddings(input_ids=ids)
    # This helper always encodes a dense sequence, without a padding mask or KV cache, so
    # SDPA keeps its implicit causal path (an equivalent explicit mask rounds differently).
    mask = (
        None
        if model.config._attn_implementation == "sdpa"
        else create_causal_mask(
            config=model.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=None,
        )
    )
    if not model.encoder.layers:
        raise ValueError("SDXL conditioning requires a nonempty CLIP encoder")
    penultimate = hidden
    for layer in model.encoder.layers:
        penultimate = hidden
        hidden = layer(hidden, mask, is_causal=True)
    hidden = model.final_layer_norm(hidden)
    positions = ids.to(dtype=torch.int, device=hidden.device)
    if model.eos_token_id != 2:
        positions = (positions == model.eos_token_id).int()
    pooled = hidden[
        torch.arange(hidden.shape[0], device=hidden.device),
        positions.argmax(dim=-1),
    ]
    return penultimate, pooled


def encode_tokens(
    first_encoder: Any,
    second_encoder: Any,
    ids: torch.Tensor,
    ids_2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two-tower tensor computation, after inputs and weights have been placed."""
    first, _ = _clip_text(first_encoder, ids)
    second, pooled = _clip_text(second_encoder.text_model, ids_2)
    prompt = torch.cat([first, second], dim=-1)
    return prompt, second_encoder.text_projection(pooled)


#: Latents a decode window shares with each neighbour, cross-faded (256 px): the blend of Diffusers'
#: 1024 px tiles.
OVERLAP = 32
#: A decode's peak beside its output, in outputs (`split(holds=)`): a whole 2048² decode held about
#: 9.7 GB in bf16.
HOLDS = 290


def decode_latents(vae: Any, latents: torch.Tensor, scale: float) -> torch.Tensor:
    """Decode with the VAE's already selected representation, never changing its weights: whole
    where it fits, else in the fewest windows that do (`split`), neighbours cross-faded. Each
    window's GroupNorms read its own statistics: 34 dB against a whole decode at 2048²."""
    factor = 2 ** (len(vae.config.block_out_channels) - 1)
    quant = vae.post_quant_conv

    def decoded(z: torch.Tensor) -> torch.Tensor:
        return vae.decoder(quant(z) if quant is not None else z)  # type: ignore[no-any-return]

    cuts = (Cut(-2, factor, OVERLAP), Cut(-1, factor, OVERLAP))
    return split(decoded, latents / scale, *cuts, holds=HOLDS)  # type: ignore[no-any-return]


def decode_dtype(vae: Any) -> torch.dtype:
    """The dtype every SDXL decode runs the VAE in. The SDXL VAE overflows fp16 (force_upcast):
    bf16 where the GPU computes it natively (fp32's range; 48 dB PSNR to fp32 on the 1024px fox,
    run 1642), else fp32."""
    if not getattr(vae.config, "force_upcast", False):
        return next(vae.parameters()).dtype
    return (
        torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float32
    )


def hold_decode_dtype(vae: Any, tel: Telemetry) -> None:
    """Hold the VAE in `decode_dtype` for good, before any VAE scope opens: the Runtime casts its
    stored bytes when it next fills them, so nothing loads here and no request converts weights."""
    dtype = decode_dtype(vae)
    if next(vae.parameters()).dtype == dtype:
        return
    hold_dtype(vae, dtype)
    tel.log("vae representation", dtype=str(dtype).removeprefix("torch."))


def guided(
    unet: Any, latents: torch.Tensor, timestep: torch.Tensor, prompt: torch.Tensor, pooled: torch.Tensor,
    time_ids: torch.Tensor, guidance: float,
) -> torch.Tensor:
    """The UNet's noise prediction, guided against the negative half of `prompt` above 1."""
    batch = torch.cat([latents] * 2) if guidance > 1.0 else latents
    noise = unet(
        batch, timestep, encoder_hidden_states=prompt,
        added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
    ).sample
    if guidance > 1.0:
        uncond, cond = noise.chunk(2)
        noise = uncond + guidance * (cond - uncond)
    return noise  # type: ignore[no-any-return]
