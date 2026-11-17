"""se-157 proof only, never merged: the upstream reference beside the package, on the same weights.

BFL's `flux3/pipeline.py` + `flux3/sampling.py` (bfc8889) step for step, with Diffusers' unmodified
`Flux3Transformer2DModel.forward` as the DiT (BFL's own DiT needs BFL-format weights), the full Qwen3-VL
through Diffusers' `_get_qwen3_vl_prompt_embeds` (BFL's `TextEncoder` copied), and the VAE in float32 as BFL
loads it. Pixels are rounded (BFL truncates), so the comparison sees the pipeline, not 0.5 LSB.
"""

from __future__ import annotations

from typing import Annotated, cast

import msgspec
import numpy as np
import torch
from cozy_runtime import vlm
from cozy_runtime.author import Config, Context, Loader, Model, Telemetry, uses_components
from cozy_runtime.fields import FileBound
from cozy_runtime.image import ImageResult
from cozy_runtime.media import ImageFile
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_flux3 import Flux3Transformer2DModel
from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
from PIL import Image
from transformers import AutoModelForImageTextToText, Qwen3VLConfig
from transformers import initialization as transformer_init

from .klein import MAX_TOKENS, PAD_MULTIPLE, TEXT_LAYERS, construct, prepare_reference

ALPHA = 16.0


class RefGraph:
    def __init__(self, config: Config) -> None:
        with transformer_init.no_init_weights():
            transformer = construct(Flux3Transformer2DModel, config.section("transformer").mapping())
            text_encoder = AutoModelForImageTextToText.from_config(  # type: ignore[no-untyped-call]
                Qwen3VLConfig(**config.section("text_encoder").mapping()), dtype=torch.bfloat16
            )
            vae = construct(AutoencoderKLFlux2, config.section("vae").mapping())
        # BFL's checkpoint stores layers 0-31 only (353 tensors: no vision tower, layers 32-35, final norm or head);
        # BFL's `from_pretrained` leaves those random and never reads them for hidden state 32 (pre-norm).
        language = text_encoder.model.language_model
        language.layers = torch.nn.ModuleList(list(language.layers)[:32])
        language.norm = torch.nn.Identity()
        text_encoder.lm_head = torch.nn.Identity()
        del text_encoder.model.visual
        self.components: dict[str, torch.nn.Module] = {
            "transformer": transformer.to(dtype=torch.bfloat16),
            "text_encoder": text_encoder.eval(),
            "vae": vae.to(dtype=torch.float32),
        }


class RefKlein(Model[RefGraph]):
    graph: RefGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(RefGraph)
        # BFL `TextEncoder`: the folder's processor, right-padded (the Runtime parses it: no from_pretrained here).
        self.processor = vlm.processor(loader.assets, "text_encoder/")
        self.processor.tokenizer.padding_side = "right"

    @uses_components("text_encoder")
    def encode(self, prompt: str) -> torch.Tensor:
        with torch.inference_mode():
            embeds = Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(
                self.graph.components["text_encoder"], self.processor, prompt,  # type: ignore[arg-type]
                device=torch.device(str(self.device)), max_sequence_length=MAX_TOKENS, pad_multiple=PAD_MULTIPLE,
                hidden_states_layers=TEXT_LAYERS,
            )
        return embeds[0].cpu()

    @uses_components("vae")
    def encode_references(self, references: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """BFL `Pipeline.encode_references` / `AutoEncoder.encode`, f32."""
        vae = cast(AutoencoderKLFlux2, self.graph.components["vae"])
        device = torch.device(str(self.device))
        tokens, ids = [], []
        with torch.inference_mode():
            for index, reference in enumerate(references):
                x = (torch.from_numpy(np.asarray(reference)).permute(2, 0, 1).float() / 127.5 - 1.0).to(device)
                mean = vae.encode(x[None]).latent_dist.mode()
                c = mean.shape[1]
                z = mean.reshape(1, c, mean.shape[2] // 2, 2, mean.shape[3] // 2, 2).permute(0, 1, 3, 5, 2, 4)
                z = z.reshape(1, c * 4, mean.shape[2] // 2, mean.shape[3] // 2)
                bn = vae.bn
                z = (z - bn.running_mean.view(1, -1, 1, 1)) / torch.sqrt(bn.running_var.view(1, -1, 1, 1) + vae.config["batch_norm_eps"])  # type: ignore[union-attr, operator]
                h, w = z.shape[2:]
                tokens.append(z[0].permute(1, 2, 0).reshape(h * w, c * 4).to(torch.bfloat16).cpu())
                ids.append(_ids(h, w, -(1 + index)))
        return torch.cat(tokens), torch.cat(ids)

    @uses_components("transformer")
    def denoise(self, text: torch.Tensor, negative: torch.Tensor | None, refs: tuple[torch.Tensor, torch.Tensor] | None,
                width: int, height: int, steps: int, guidance: float | None, seed: int) -> torch.Tensor:
        """BFL `Pipeline.generate` + `denoise` / `denoise_flow_map` (bf16 autocast), Diffusers' forward per sample."""
        tf = cast(Flux3Transformer2DModel, self.graph.components["transformer"])
        device = torch.device(str(self.device))
        h, w = height // 16, width // 16
        noise = torch.randn((h * w, 128), generator=torch.Generator().manual_seed(seed))
        x = noise.to(torch.bfloat16).float().to(device)[None]
        img_ids = _ids(h, w, 0).to(device)
        t = torch.linspace(1.0, 0.0, steps + 1)
        schedule = ALPHA * t / (1.0 + (ALPHA - 1.0) * t)
        flow_map = bool(tf.config["flow_map"])
        texts = [text] if negative is None else [negative, text]
        texts = [e[None].to(device=device, dtype=torch.bfloat16) for e in texts]
        txt_ids = [_txt_ids(e.shape[1]).to(device) for e in texts]
        cond = None if refs is None else (refs[0][None].to(device), refs[1].to(device))
        with torch.inference_mode():
            for step in range(steps):
                now, dt = schedule[step].item(), (schedule[step + 1] - schedule[step]).item()
                timestep = torch.full((1,), now, dtype=torch.bfloat16, device=device)
                target = schedule[step + 1].reshape(1).to(device=device, dtype=torch.bfloat16) if flow_map else None
                outs = []
                for e, ids in zip(texts, txt_ids, strict=True):
                    outs.append(tf(
                        hidden_states=x.to(torch.bfloat16), encoder_hidden_states=e, timestep=timestep, img_ids=img_ids,
                        txt_ids=ids, image_cond_hidden_states=None if cond is None else cond[0],
                        image_cond_ids=None if cond is None else cond[1], target_timestep=target, return_dict=False,
                    )[0])
                if guidance is None:
                    velocity = outs[0]
                else:
                    unconditional, conditional = outs
                    velocity = unconditional + guidance * (conditional - unconditional)
                x = x + dt * velocity
        return x[0].cpu()

    @uses_components("vae")
    def decode(self, tokens: torch.Tensor, width: int, height: int) -> Image.Image:
        """BFL `AutoEncoder.decode` in f32, then `to_pil` with rounding."""
        vae = cast(AutoencoderKLFlux2, self.graph.components["vae"])
        device = torch.device(str(self.device))
        h, w = height // 16, width // 16
        with torch.inference_mode():
            z = tokens.float().to(device).reshape(h, w, 128).permute(2, 0, 1)[None]
            bn = vae.bn
            z = z * torch.sqrt(bn.running_var.view(1, -1, 1, 1) + vae.config["batch_norm_eps"]) + bn.running_mean.view(1, -1, 1, 1)  # type: ignore[union-attr, operator]
            z = z.reshape(1, 32, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(1, 32, 2 * h, 2 * w)
            image = vae.decode(z, return_dict=False)[0][0]
            pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(pixels)


def _ids(h: int, w: int, layer: int) -> torch.Tensor:
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    zeros = torch.zeros_like(rows)
    ids = torch.stack([zeros, rows, cols, zeros + layer], dim=-1).reshape(h * w, 4).float()
    ids[:, 1:3] *= 0.5
    return ids


def _txt_ids(length: int) -> torch.Tensor:
    positions = torch.arange(length)
    return torch.stack([torch.zeros_like(positions)] * 3 + [positions], dim=-1).float()


class RefInput(msgspec.Struct, kw_only=True):
    prompt: str
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)] | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    seed: int = 7


def reference(ctx: Context, payload: RefInput, model: RefKlein, tel: Telemetry) -> ImageResult:
    """BFL `generate` with `pup=False`: the prompt as given, BFL's reference preparation and default size."""
    flow_map = bool(model.graph.components["transformer"].config["flow_map"])  # type: ignore[index]
    steps = payload.steps or (16 if flow_map else 50)
    guidance = None if flow_map else (4.0 if payload.guidance is None else payload.guidance)
    images = [payload.image.read()] if payload.image is not None else []
    crop = None
    if images:
        prepared, box = prepare_reference(images[0])
        references = [prepared]
        crop = box if not (payload.width or payload.height) else None
    else:
        references = []
    width, height = references[0].size if references else (1024, 1024)
    width, height = payload.width or width, payload.height or height
    with tel.stage("encoding prompt"):
        text = model.encode(payload.prompt)
        negative = model.encode("") if guidance is not None else None
    refs = model.encode_references(references) if references else None
    with tel.stage("generating image"):
        tokens = model.denoise(text, negative, refs, width, height, steps, guidance, payload.seed)
    image = model.decode(tokens, width, height)
    if crop is not None:
        image = image.crop(crop)
    tel.log("reference", steps=steps, guidance=guidance, width=width, height=height, text_tokens=int(text.shape[0]))
    return ImageResult(ImageFile.from_image(image, format="png"), image.width, image.height, payload.seed, payload.prompt)

