"""SDXL's numerics outside Diffusers' pipeline: CLIP conditioning, the Euler step, VAE decode."""

from __future__ import annotations

from typing import Any

import torch
from cozy_runtime import tiles
from cozy_runtime.author import Telemetry, hold_dtype
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


#: The largest canvas decoded whole where it fits, about 9.7 GB in bf16. HiDiffusion's larger canvases
#: (up to 16 MP) decode in tiles: a whole 4096² decode asked for 8 GiB more with 14 GiB allocated on a
#: 24 GB card. Tiles renormalize per tile (the VAE's GroupNorm), 34 dB against a whole 2048² decode.
WHOLE_PIXELS = 2048 * 2048
#: Tile sides (px), largest first: smaller where they do not fit.
TILES = (1024, 512, 256)


def decode_latents(vae: Any, latents: torch.Tensor, scale: float) -> torch.Tensor:
    """Decode with the VAE's already selected representation, never changing its weights: whole
    up to `WHOLE_PIXELS` where it fits, else in the largest `TILES` that fit."""
    z = latents / scale
    factor = 2 ** (len(vae.config.block_out_channels) - 1)
    if z.shape[-2] * z.shape[-1] * factor * factor <= WHOLE_PIXELS:
        try:
            return vae.decode(z).sample
        except torch.OutOfMemoryError as error:
            error.__traceback__ = None  # the whole decode's activations go with its frames
    *larger, smallest = TILES
    for side in larger:
        try:
            return _tiled(vae, z, factor, side)
        except torch.OutOfMemoryError as error:
            error.__traceback__ = None  # this attempt's tiles go with its frames
    return _tiled(vae, z, factor, smallest)


def _tiled(vae: Any, z: torch.Tensor, factor: int, side: int) -> torch.Tensor:
    """Diffusers' tiled decode (`AutoencoderKL.tiled_decode`'s quarter-side blend), its bits, in
    `side` px tiles (`cozy_runtime.tiles`)."""

    def decoded(tile: torch.Tensor) -> torch.Tensor:
        quant = vae.post_quant_conv
        return vae.decoder(quant(tile) if quant is not None else tile)  # type: ignore[no-any-return]

    return tiles.tiled(decoded, z, scale=factor, side=side, stride=side * 3 // 4)


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


class _Euler:
    """Diffusers' epsilon Euler step in its operation order, over one device schedule `table`.

    `table` is float32 [5, steps]: timesteps, 1/sqrt(sigma^2+1), 1/sigma, sigma and
    sigma_next - sigma, indexed by the global step.
    """

    def __init__(self, scheduler: Any, device: Any) -> None:
        if scheduler.config.prediction_type != "epsilon":
            raise ValueError(
                f"SDXL's Euler step predicts epsilon, not {scheduler.config.prediction_type}"
            )
        # A CPU denominator takes reciprocal in double, then multiplies in float opmath.
        # Moving the original scalar to the GPU changes half rounding, so prepare the
        # same reciprocals before transfer.
        sigmas = scheduler.sigmas[:-1]
        scales = torch.stack([(sigma**2 + 1) ** 0.5 for sigma in sigmas])
        sigma_hat = torch.stack([sigma * 1 for sigma in sigmas])
        rows = (
            scheduler.timesteps,
            scales.double().reciprocal().float(),
            sigma_hat.double().reciprocal().float(),
            sigma_hat,
            scheduler.sigmas[1:] - sigma_hat,
        )
        if any(row.dtype != torch.float32 for row in rows):
            raise ValueError("SDXL's Euler schedules are float32")
        self.table = torch.stack([row.to(device=device) for row in rows])

    @staticmethod
    def step(
        unet: Any,
        index: torch.Tensor,
        state: torch.Tensor,
        prompt: torch.Tensor,
        pooled: torch.Tensor,
        time_ids: torch.Tensor,
        table: torch.Tensor,
        guidance: float,
    ) -> None:
        timestep, inverse_scale, inverse_sigma, sigma, delta = table.index_select(1, index)
        inverse_scale, inverse_sigma = inverse_scale.reshape(()), inverse_sigma.reshape(())
        sigma, delta = sigma.reshape(()), delta.reshape(())
        batch = torch.cat([state] * 2) if guidance > 1.0 else state
        noise = unet(
            (batch.float() * inverse_scale).to(batch.dtype),
            timestep,
            encoder_hidden_states=prompt,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
        ).sample
        if guidance > 1.0:
            uncond, cond = noise.chunk(2)
            noise = uncond + guidance * (cond - uncond)
        sample = state.to(torch.float32)
        original = sample - (sigma * noise.float()).to(noise.dtype)
        derivative = (sample - original) * inverse_sigma
        state.copy_((sample + derivative * delta).to(noise.dtype))
