"""SDXL's numerics outside Diffusers' pipeline: CLIP conditioning, the Euler step, VAE decode."""

from __future__ import annotations

from typing import Any

import torch
from cozy_runtime import rng
from cozy_runtime.author import Cut, Telemetry, hold_dtype, split
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


def euler_schedule(scheduler: Any, device: Any) -> torch.Tensor:
    """Diffusers' Euler (epsilon or v) as one float32 device table [7, steps], indexed by the
    global step: timesteps, 1/sqrt(sigma^2+1), 1/sigma, sigma, sigma_next - sigma,
    -sigma/sqrt(sigma^2+1), 1/(sigma^2+1)."""
    if scheduler.config.prediction_type not in ("epsilon", "v_prediction"):
        raise ValueError(f"SDXL's Euler step predicts epsilon or v, not {scheduler.config.prediction_type}")
    # A CPU denominator takes reciprocal in double, then multiplies in float opmath.
    # Moving the original scalar to the GPU changes half rounding, so prepare the
    # same reciprocals before transfer.
    sigmas = scheduler.sigmas[:-1]
    squares = torch.stack([sigma**2 + 1 for sigma in sigmas])
    scales = torch.stack([square**0.5 for square in squares])
    sigma_hat = torch.stack([sigma * 1 for sigma in sigmas])
    rows = (
        scheduler.timesteps,
        scales.double().reciprocal().float(),
        sigma_hat.double().reciprocal().float(),
        sigma_hat,
        scheduler.sigmas[1:] - sigma_hat,
        torch.stack([-sigma / scale for sigma, scale in zip(sigmas, scales, strict=True)]),
        squares.double().reciprocal().float(),
    )
    if any(row.dtype != torch.float32 for row in rows):
        raise ValueError("SDXL's Euler schedules are float32")
    return torch.stack([row.to(device=device) for row in rows])


def euler_step(
    unet: Any, index: torch.Tensor, state: torch.Tensor, prompt: torch.Tensor, pooled: torch.Tensor,
    time_ids: torch.Tensor, guidance: float, v: bool, table: torch.Tensor,
) -> None:
    """Diffusers' Euler step in its operation order, `v` its v prediction."""
    rows = table.index_select(1, index)
    timestep = rows[0]
    inverse_scale, inverse_sigma, sigma, delta, v_scale, v_inverse = (row.reshape(()) for row in rows[1:])
    noise = _guided(unet, (state.float() * inverse_scale).to(state.dtype), timestep, prompt, pooled,
                    time_ids, guidance)
    sample = state.to(torch.float32)
    if v:
        original = (v_scale * noise.float()).to(noise.dtype) + sample * v_inverse
    else:
        original = sample - (sigma * noise.float()).to(noise.dtype)
    derivative = (sample - original) * inverse_sigma
    state.copy_((sample + derivative * delta).to(noise.dtype))


def ddim_schedule(
    scheduler: Any, device: Any, generator: torch.Generator, shape: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diffusers' epsilon DDIM at eta 1 (HiDiffusion's recipe) as one float32 device table
    [6, steps], indexed by the global step: timesteps, sqrt(1 - a_t), 1/sqrt(a_t), sqrt(a_prev),
    sqrt(1 - a_prev - s^2) and the step's noise scale s; and each step's noise [steps, *shape],
    drawn on the CPU from `generator` in the order DDIM draws it."""
    config = scheduler.config
    if config.prediction_type != "epsilon" or config.thresholding or config.clip_sample:
        raise ValueError("SDXL's DDIM step predicts epsilon, unclipped and unthresholded")
    rows = []
    for t in scheduler.timesteps.tolist():
        prev = t - config.num_train_timesteps // scheduler.num_inference_steps
        alpha = scheduler.alphas_cumprod[t]
        alpha_prev = scheduler.alphas_cumprod[prev] if prev >= 0 else scheduler.final_alpha_cumprod
        deviation = 1.0 * scheduler._get_variance(t, prev) ** 0.5
        rows.append((
            torch.tensor(float(t)), (1 - alpha) ** 0.5, (alpha**0.5).double().reciprocal().float(),
            alpha_prev**0.5, (1 - alpha_prev - deviation**2) ** 0.5, deviation,
        ))
    table = torch.stack([torch.stack(row) for row in zip(*rows, strict=True)])
    noise = torch.stack([rng.noise(shape, generator, dtype=torch.float16) for _ in rows])
    return table.to(device=device), noise.to(device=device)


def ddim_step(
    unet: Any, index: torch.Tensor, state: torch.Tensor, prompt: torch.Tensor, pooled: torch.Tensor,
    time_ids: torch.Tensor, guidance: float, table: torch.Tensor, noise: torch.Tensor,
) -> None:
    """Diffusers' DDIM step in its operation order."""
    rows = table.index_select(1, index)
    timestep = rows[0]
    root_beta, inverse_root_alpha, root_alpha_prev, direction, deviation = (row.reshape(()) for row in rows[1:])
    eps = _guided(unet, state, timestep, prompt, pooled, time_ids, guidance)
    dtype = eps.dtype
    original = ((state - (root_beta * eps.float()).to(dtype)).float() * inverse_root_alpha).to(dtype)
    sample = (root_alpha_prev * original.float()).to(dtype) + (direction * eps.float()).to(dtype)
    state.copy_(sample + (deviation * noise.index_select(0, index)[0].float()).to(dtype))


def _guided(
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
