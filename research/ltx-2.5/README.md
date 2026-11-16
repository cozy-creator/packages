# LTX-2.5 implementation research

Source-only investigation on 2026-11-15. Owner: `ltx25_upstream_research`. This note supports the Cozy package and video-generation tool implementation; it does not assert inference, converted-weight, VRAM, or output qualification. No weights were downloaded and no GPU was rented by this researcher.

## Pinned upstream inputs

| Source | Verified revision |
| --- | --- |
| [`Lightricks/LTX-2.5` API](https://huggingface.co/api/models/Lightricks/LTX-2.5) | `49b093a39c19169a8a81721dcc541e07e945232f`, last modified 2026-10-11 |
| [`Lightricks/LTX-2.5-Diffusers` API](https://huggingface.co/api/models/Lightricks/LTX-2.5-Diffusers) | `4791a1100e0cf97f7ed5c55c12dd71dce13f9dd3`, last modified 2026-10-22 |
| [Official LTX inference source](https://github.com/Lightricks/LTX-2/tree/f0f2f8607189c78a72420b2b63cb1975b5eca179) | `f0f2f8607189c78a72420b2b63cb1975b5eca179` |
| [Diffusers source inspected](https://github.com/huggingface/diffusers/tree/43e30028af5120d8b0868cb6975b9fad39cc4fab) | `43e30028af5120d8b0868cb6975b9fad39cc4fab` |

Anonymous HF model metadata, filenames, and model cards are readable. Requests for actual pinned `config.json` files returned HTTP 401, and no `HF_TOKEN` or default cached token was present in the researcher environment. Consequently, constructor values below are verified against official converter/reference source, **not** the gated checkpoint's actual config bytes. The implementation must retain and inspect the actual downloaded configuration before making checkpoint-specific parity claims.

## Implementation decision

The smallest officially documented generation path is the **distilled transformer + convolutional video VAE + Gemma 4 encoder/connectors + audio VAE/vocoder**, doing one generation call. A stronger default is two stages: half-resolution generation, x2 spatial latent upsampling, then three refinement steps at target resolution. Both generate synchronized audio and video. The separate diffusion decoder and DFR are quality extensions and have substantially more activation/backend complexity. [Official Diffusers model card](https://huggingface.co/Lightricks/LTX-2.5-Diffusers/blob/4791a1100e0cf97f7ed5c55c12dd71dce13f9dd3/README.md), [reference distilled pipeline](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-pipelines/src/ltx_pipelines/distilled.py#L168-L250).

Do not silently port an LTX-2.3 Gemma 3 loader: 2.5 uses a customized **Gemma 4 12B** encoder. Do not load the Comfy int8-convrot exports through ordinary PyTorch inference; upstream explicitly limits them to ComfyUI. [Official model pack](https://huggingface.co/Lightricks/LTX-2.5/blob/49b093a39c19169a8a81721dcc541e07e945232f/README.md).

## Components and checkpoint choices

The original pack has these exact component filenames; the [official API tree](https://huggingface.co/api/models/Lightricks/LTX-2.5/tree/49b093a39c19169a8a81721dcc541e07e945232f?recursive=true) identifies their custody:

| Component | Original pack path | Diffusers pack equivalent |
| --- | --- | --- |
| Distilled DiT | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | `transformer/` (default) |
| Dev/full DiT | `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | `transformer_full/` |
| Gemma 4 + projections | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | `text_encoder/`, `tokenizer/`, `connectors/` |
| Conv video VAE | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | `vae/` |
| Diffusion video VAE | `vae/ltx-2.5-video-vae-bf16.safetensors` | encoder stays `vae/`; decoder `diffusion_decoder/` |
| Audio VAE + vocoder | `vae/ltx-2.5-audio-vae-bf16.safetensors` | `audio_vae/`, `vocoder/` |
| Spatial x2 upsampler | `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `latent_upsampler/` |
| Temporal x2 upsampler | `latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` | `temporal_latent_upsampler/` |
| Optional duration head | `model_patches/ltx-2.5-duration-head-bf16.safetensors` | `duration_head/` |
| Distillation LoRA for dev workflows | `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | root LoRA file |

There are also dev/distilled `*-comfy-int8-convrot.safetensors` exports and a distilled `*-nvfp4.safetensors` export. NVFP4 reference execution requires Blackwell plus `ltx-kernels` and `nvfp4-prequant`; it is not a general GPU fallback. Reference BF16 checkpoints can instead use `fp8-cast` and CPU/disk offload. [Reference optimization policies](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-pipelines/docs/optimization.md#fp8-quantization-lower-memory-footprint).

Distilled inference needs no distillation LoRA. The full/dev transformer is the trainable/guided path; official two-stage dev workflows apply a distilled LoRA in refinement. DFR specifically requires the **distilled** checkpoint plus the separate [`LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler`](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler), whose strength is fixed at `0.5`. [Reference pipeline inventory](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-pipelines/docs/pipelines.md#12-dfrpipeline).

## Dependency release facts

The HF card says support is unreleased and requests Diffusers main. This is stale relative to the live registry: **Diffusers `0.43.0` was published 2026-11-12**, and its release tag includes the 2.5 converter, Gemma 4 pipeline integration, diffusion decoder, `ff_bias`, and generated-keyframe support. A pinned released dependency can therefore provide these classes. This is source availability, not a tested Cozy environment. [PyPI release metadata](https://pypi.org/pypi/diffusers/0.43.0/json), [v0.43.0 converter](https://github.com/huggingface/diffusers/blob/v0.43.0/scripts/convert_ltx2_to_diffusers.py), [v0.43.0 pipeline](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/ltx2/pipeline_ltx2.py).

Official LTX source packages are version `1.6.2`; PyPI `ltx-core` and `ltx-pipelines` still expose `1.0.0` (2026-03-18), so an unqualified PyPI install of those names does not reproduce current 2.5 reference source. Current reference requirements are Python `>=3.10`, `torch~=2.7`, `transformers>=5.8.0,<5.15`, `torchvision`, `torchaudio`, `safetensors`, `accelerate`, `einops`, and others. The upper Transformers bound documents a 5.15 Gemma 4 config failure and identifies `5.14.1` as newest verified there. Do not assume newest Transformers `5.23.0` has been qualified with this reference. [Core requirements](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-core/pyproject.toml#L1-L32), [pipeline requirements](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-pipelines/pyproject.toml#L1-L8), [core PyPI](https://pypi.org/pypi/ltx-core/json), [pipelines PyPI](https://pypi.org/pypi/ltx-pipelines/json).

The optimized native DiffVAE extra is unusually strict at this source revision: `natten==0.21.7+torch2130cu132` with `torch==2.13.0` on Linux; it is optional. Diffusers uses a different kernel path through `kernels` and `LTX2VideoVaeNeighborhoodNattenProcessor`. Its default uncompiled FlexAttention decoder can materialize a huge score matrix. Convolutional decode avoids this extra dependency for the initial implementation. [Reference extra pins](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-core/pyproject.toml#L34-L64), [Diffusers diffusion decoder](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/models/autoencoders/ltx2_diffusion_decoder.py).

## Shapes and validated input grids

Single-stage dimensions must be multiples of **32**. A two-stage target must be multiples of **64**, because its half-resolution first stage still needs the 32-pixel grid. Reference DFR with two spatial upscalings requires **128**. The causal frame grid is **`8k+1`**, including 121 and 241. The reference floors off-grid counts. Avoid silently reporting the originally requested count after this snapping. [Resolution and frame helpers](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py#L591-L620).

Video VAE compression is `(time,height,width)=(8,32,32)`, with 128 latent channels. Thus pixel video `[B,3,F,H,W]` becomes `[B,128,(F-1)/8+1,H/32,W/32]` and DiT input is `[B,N_video,128]` with patch size 1. Audio latent space is `[B,8,L_audio,16]`, packed as `[B,L_audio,128]`; `L_audio=round((F/fps)*25)`, from 16k internal sample rate / 160 mel hop / 4 temporal compression. Do not substitute `(F-1)/fps` for the audio duration used by this implementation. [Latent preparation](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L766-L849), [audio duration calculation](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L1306-L1342).

24 fps is the documented example/default. DFR can double playback rate and frame count per temporal round: from 121 frames/24fps to 241/48 and 481/96, approximately preserving duration. Reference DFR uses 60fps model conditioning whenever playback is above 30fps. This is a DFR policy, not evidence that arbitrary conditioning fps works equally well. 1920x1088 and 3840x2176 meet the native grids; ordinary 1080/2160 heights do not. [DFR grids and fps](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-pipelines/docs/pipelines.md#running-dfr), [conditioning fps source](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/utils.py#L54-L61).

## Constructors and component extraction

### Diffusers assembly

The relevant public constructor is:

```python
LTX2Pipeline(
    scheduler, vae, audio_vae, text_encoder, tokenizer,
    connectors, transformer, vocoder,
    processor=None, prompt_enhancer=None, duration_head=None,
)
```

Types are `FlowMatchEulerDiscreteScheduler`, `AutoencoderKLLTX2Video`, `AutoencoderKLLTX2Audio`, `Gemma4UnifiedForConditionalGeneration`, tokenizer, `LTX2TextConnectors`, `LTX2VideoTransformer3DModel`, and `LTX2VocoderWithBWE`. `LTX2ImageToVideoPipeline` takes the same components. Use `from diffusers import LTX2TextConnectors`; the older `diffusers.pipelines.ltx2.connectors` import is now a deprecated alias. [T2V constructor](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L235-L269), [I2V constructor](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/pipeline_ltx2_image2video.py#L238-L272), [connector class](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/models/condition_embedders/condition_embedder_ltx2.py#L335-L397).

The converter's `get_ltx2_transformer_config("2.5")` defines the actual architecture, rather than the generic class defaults:

```python
dict(
    in_channels=128, out_channels=128, patch_size=1, patch_size_t=1,
    num_attention_heads=32, attention_head_dim=128, cross_attention_dim=4096,
    vae_scale_factors=(8, 32, 32), pos_embed_max_pos=20,
    base_height=2048, base_width=2048, gated_attn=True, cross_attn_mod=True,
    audio_in_channels=128, audio_out_channels=128,
    audio_patch_size=1, audio_patch_size_t=1,
    audio_num_attention_heads=32, audio_attention_head_dim=64,
    audio_cross_attention_dim=2048, audio_scale_factor=4,
    audio_pos_embed_max_pos=20, audio_sampling_rate=16000, audio_hop_length=160,
    audio_gated_attn=True, audio_cross_attn_mod=True,
    num_layers=48, activation_fn="gelu-approximate",
    qk_norm="rms_norm_across_heads", norm_elementwise_affine=False,
    norm_eps=1e-6, caption_channels=3840,
    attention_bias=True, attention_out_bias=True, rope_theta=10000.0,
    rope_double_precision=True, causal_offset=1,
    timestep_scale_multiplier=1000, cross_attn_timestep_scale_multiplier=1000,
    rope_type="split", use_prompt_embeddings=False, perturbed_attn=True,
    ff_bias=False, use_keyframes_abs_pos_embedding=True,
)
```

`audio_ff_bias` and `use_prompt_adaln_single` retain their **True** defaults for the released checkpoint described by the converter. Do not infer all 2.5 checkpoints disable prompt AdaLN from generic KV-cache comments. The connector's input width/layer count is derived from the actual Gemma `text_config`: `caption_channels=hidden_size`, `text_proj_in_factor=num_hidden_layers+1`. [Official converter, 2.5 transformer/connector configs](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/scripts/convert_ltx2_to_diffusers.py#L399-L570).

### Text encoder extraction and projections

1. Tokenize the stripped raw caption with left padding, max length **1024**, truncation, a leading BOS, and no appended EOS on the native reference encode path. Gemma 4's raw tokenizer does not automatically add BOS there; its wrapper does. This is plain caption encoding, not a chat-template prompt. [Native tokenizer](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-core/src/ltx_core/text_encoders/gemma/tokenizer.py#L11-L63), [asset max length](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-core/src/ltx_core/text_encoders/gemma/gemma_assets.py#L162-L175).
2. Native reference calls `conditional_model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)` to skip unused LM-head logits. Collect **all** hidden states including the embedding layer. Stack at `dim=-1` to `[B,T,D,L+1]`, then flatten the final two dimensions preserving that order. Diffusers currently calls the full conditional model; an adapted Cozy encode component can use the inner model if it preserves these outputs. [Native encode](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py#L49-L71), [Diffusers encode](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L303-L365).
3. 22B feature extraction applies per-token RMS normalization over `D`, independently for every layer, zeros padded tokens, rescales by `sqrt(output_dim/D)`, and uses independent learned Linear projections of `D*(L+1)` to **4096 video** and **2048 audio**, both with bias. Use the packaged projections, not Gemma's final hidden state or a newly initialized readout. [FeatureExtractorV2](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-core/src/ltx_core/text_encoders/gemma/feature_extractor.py#L48-L129), [dimension construction](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/encoder_configurator.py#L176-L214).
4. `LTX2TextConnectors.forward(text_encoder_hidden_states, attention_mask, padding_side="left", scale_factor=8)` accepts packed 3D or 4D hidden states and returns `(video_embeddings, audio_embeddings, binary_mask)`. 2.5 uses eight connector layers per modality, 128 learnable registers, 32 heads, video head dim128/audio64, gated attention, split RoPE. Native `EmbeddingsProcessor` explicitly moves valid left-padded features to the right-padded connector layout. [Diffusers connector forward](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/models/condition_embedders/condition_embedder_ltx2.py#L399-L482), [native processor](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_processor.py#L70-L117).

The original single-file encoder embeds `config.json` metadata, tokenizer bytes, and JSON/Jinja sidecars as U8 tensors. `GemmaAssets.from_single_file` reads these; it is not sufficient to extract only floating tensors during conversion. `config_dict` selects `gemma4_unified` through Transformers `CONFIG_MAPPING`. Some original TE layouts flatten LM weights to `model.layers.*`; official state-dict operations remap them to the conditional model's nested language-model tree and mirror tied token embeddings to the LM head. [Packed assets reader](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-core/src/ltx_core/text_encoders/gemma/gemma_assets.py#L105-L142), [Gemma key remapping](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/encoder_configurator.py#L220-L280).

Prompt enhancement is optional and uses a **separate instruct checkpoint**, such as `google/gemma-4-E2B-it`. The fine-tuned 2.5 encoder is not itself trained for enhancement. Explicit `num_frames` also makes the duration head unnecessary for a minimal package. [Prompt-encoder enhancement guard](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-pipelines/src/ltx_pipelines/utils/blocks/prompt_encoder.py#L148-L195).

## Denoising and sampling

### Distilled schedules

Native reference includes terminal zero:

```python
stage_1 = [1., .99375, .9875, .98125, .975, .909375, .725, .421875, 0.]
stage_2 = [.909375, .725, .421875, 0.]
```

These are **eight and three model evaluations**. Some prose says four stage-2 steps by counting sigma points; the executable code is authoritative. Diffusers `sigmas=` constants omit terminal zero because its scheduler appends it. Use the explicit schedules, not merely `num_inference_steps=8`. Distilled overrides are `guidance_scale=audio_guidance_scale=1`, `stg_scale=audio_stg_scale=0`, `modality_scale=audio_modality_scale=1`. [Native schedules](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py#L15-L25), [Diffusers schedules](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/utils.py#L25-L40).

The 2.5 native reference has an important newer behavior: model version `>=2.5` uses **Euler ancestral**, `eta=1`, `s_noise=1`. It seeds injected stage-1 noise at `seed+10000` and stage-2 at `seed+20000`; within each step it draws video noise before audio noise and skips noise draw on terminal zero. Its step computes in FP32 and then stores the next state in BF16 by default. [Native ancestral selection/seeds](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-pipelines/src/ltx_pipelines/distilled.py#L82-L146), [joint ancestral loop](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py#L563-L721).

By contrast, the inspected Diffusers standard T2V/I2V pipeline uses ordinary `FlowMatchEulerDiscreteScheduler.step` and retains FP32 latents, casting only model inputs to the BF16 prompt/model dtype. Matching the sigmas alone therefore does **not** establish native-reference numerical parity. For an immediate supported implementation, choose one path explicitly and record which sampler/dtype path produced the demo; test a native ancestral adaptation separately if exact reference behavior is required. [Diffusers latent dtype and scheduler calls](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L1290-L1342), [Diffusers model forward](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L1396-L1439), [Diffusers scheduler step](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L1588-L1596).

### Transformer extraction signature

```python
video_velocity, audio_velocity = transformer(
    hidden_states=video_tokens.to(model_dtype),       # [B,Nv,128]
    audio_hidden_states=audio_tokens.to(model_dtype), # [B,Na,128]
    encoder_hidden_states=video_context,             # [B,T,4096]
    audio_encoder_hidden_states=audio_context,       # [B,T,2048]
    timestep=t, sigma=t,
    encoder_attention_mask=context_mask,
    audio_encoder_attention_mask=context_mask,
    num_frames=latent_frames, height=latent_h, width=latent_w,
    fps=conditioning_fps, audio_num_frames=audio_length,
    video_coords=video_coords, audio_coords=audio_coords,
    isolate_modalities=False, spatio_temporal_guidance_blocks=None,
    perturbation_mask=None, use_cross_timestep=True,
    return_dict=False,
)
```

Use the scheduler-provided `t` directly: it is already on the 1000-scaled timestep grid. If a custom loop starts from unit sigma, it must perform that scaling itself. Prediction is velocity, with `x0 = sample - velocity*sigma`; guidance in Diffusers is applied to x0 and converted back to velocity before scheduler stepping. Audio and video need separate scheduler state objects. [Forward call and conversion](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L851-L867), [transformer signature](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/models/transformers/transformer_ltx2.py#L1354-L1420).

Full/dev uses 30 steps by current 2.5 inherited generation defaults, video CFG3/audio CFG7, STG1 on block28, modality3, rescale0.7. The distilled HF scheduler disables dynamic shifting (`use_dynamic_shifting=False`, `shift_terminal=None`); a dev swap must re-enable `use_dynamic_shifting=True, shift_terminal=0.1`, not just substitute weights. [Reference generation defaults](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py#L40-L133), [Diffusers call defaults](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L931-L976).

## Image conditioning

The first-frame I2V pipeline encodes `[B,3,1,H,W]` through the conv video VAE and uses the posterior **mode** (`argmax`), normalizes the latent, repeats its shape to the full temporal canvas, then keeps only the first latent plane clean with a conditioning mask. The other planes start from noise. Apply that mask to timestep/noise updates as the official pipeline does. Reference image handling includes H264 re-compression at **CRF18** for 2.5, instead of CRF33 in 2.3. [Image latent preparation](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/pipeline_ltx2_image2video.py#L829-L864), [image CRF rules](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/utils.py#L326-L380).

The native split pipeline accepts `images=[ImageConditioningInput(...)]`; CLI is `--image PATH FRAME_IDX STRENGTH`, with frame0 for first-frame conditioning. It supports multiple image/keyframe conditions. The minimum public first-frame UI can use the dedicated Diffusers I2V pipeline, while arbitrary keyframes require its condition/native path. [Native conditioning calls](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-pipelines/src/ltx_pipelines/distilled.py#L346-L365).

## Two stages and decode boundaries

`LTX2LatentUpsamplePipeline(vae=vae, latent_upsampler=upsampler)` accepts `latents=...`, `latents_normalized=False`, `output_type="latent"`; it works on **unnormalized** video latents. The stage-1 `output_type="latent"` returns the correct denormalized 5D form. Then stage2 takes `latents=upsampled, audio_latents=stage1_audio`, `noise_scale=.909375`, `sigmas=[.909375,.725,.421875]`; omit height/width so the upsampled 5D latent controls them. Reuse the generator across both calls. A 544x960 stage1 produces 1088x1920 stage2. [Upsample input contract](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/pipeline_ltx2_latent_upsample.py#L266-L283), [upsample execution](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/pipeline_ltx2_latent_upsample.py#L366-L397).

Direct decode contracts:

```python
pixels = vae.decode(denormalized_video_5d.to(vae.dtype), None, return_dict=False)[0]
mel = audio_vae.decode(denormalized_audio_4d.to(audio_vae.dtype), return_dict=False)[0]
pcm = vocoder(mel)
```

Video latent statistics and audio statistics are checkpoint state/buffers; carry them through conversion. The pipeline returns denormalized video and audio for `output_type="latent"`. **Do not denormalize these again** before either decode or upsampling. Normal generation returns `(video,audio)` when `return_dict=False`. Audio export rate comes from `vocoder.config.output_sampling_rate`, **48000** for the shipped BWE vocoder. The native reference requires FP32 vocoder arithmetic to avoid spectral degradation even when weights are stored BF16. [Pipeline finish/decode](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L1614-L1672), [native vocoder precision](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-pipelines/src/ltx_pipelines/utils/blocks/audio_decoder.py#L55-L73), [BWE config](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/scripts/convert_ltx2_to_diffusers.py#L1138-L1179).

Diffusion decode is separate:

```python
decoder.set_attn_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
decoder.enable_tiling()
decode_pipe = LTX2VideoDiffusionDecodePipeline(diffusion_decoder=decoder, scheduler=scheduler)
pixels = decode_pipe(latents, generator=generator, denormalize=False,
                     output_type="np", return_dict=False)[0]
```

It draws its own noise, so seed/generator is necessary for reproducibility. Tiled and untiled decode are not numerically identical. The basic pipeline's latent output skips audio decode/vocoder, so finish audio explicitly too. [Diffusion decode pipeline](https://github.com/huggingface/diffusers/blob/43e30028af5120d8b0868cb6975b9fad39cc4fab/src/diffusers/pipelines/ltx2/pipeline_ltx2_diffusion_decode.py#L48-L120).

## Resource and qualification boundaries

The inspected current official docs do **not** give a reliable numeric minimum GPU VRAM guarantee for this exact BF16 model and output shape. They recommend staged residency, quantization/offload, and VAE tiling. A 22B BF16 transformer alone implies approximately 44GB decimal weight bytes before runtime overhead; this is arithmetic, not a measured GPU requirement. A single resident BF16 transformer is therefore unsuitable for a 24GB GPU without a supported streaming/quantization path. Record actual peak memory for the selected path and requested clip rather than turning parameter arithmetic into a rental sizing promise. [Reference residency/optimization](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-pipelines/docs/optimization.md).

Use `torch.inference_mode()` around native pipeline calls; `eval()` alone leaves autograd enabled, and retained text-encoder outputs can hold tens of GiB. For conv VAE the native auto-tiling recipe uses spatial long-side768/overlap64 and temporal80/overlap24; diffusion decode uses a separate measured-budget tile calculation. [Inference-mode guidance](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-pipelines/docs/installation.md#calling-pipelines-from-python), [tiling rules](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-pipelines/docs/optimization.md#diffusion-vae-decoder).

Initial real qualification should cover text-to-video plus audible output, first-frame I2V, exact frame/fps/dimensions, deterministic rerun for the chosen sampler/decoder, both latent normalization boundaries, and the intended longer clip rather than only a tiny smoke. Native-reference and Diffusers output should be treated as distinct baselines until sampler, state dtype, tokenizer/BOS, image re-compression, and decoder behavior are matched.

## License metadata

The code is Apache2; model weights use **LTX-2.x Community License**, not Apache2. Current source license is dated 2026-08-11 and amended 2026-10-29. It includes the entity revenue threshold of $10M, redistribution/pass-through conditions, and Attachment A item20 concerning competing/replacement products/services and separate commercial licensing. Preserve the model's full license and notices in package distribution and surface that license accurately. This note records source terms without deciding whether a particular Cozy use is covered. [Authoritative model license](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/LICENSE-2_x), [code license](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/LICENSE).

## Remaining source/runtime checks

- Read actual gated HF configuration/tokenizer/index bytes at the pinned revisions, using the already authorized user's credentials; verify Gemma D/layer count rather than reusing Gemma3 defaults.
- Confirm the selected installed Diffusers/Transformers combination imports and builds the actual Gemma4 config. Released-class availability is verified; runtime compatibility is not.
- Preserve the actual checkpoint statistics, safetensor keys, config, tokenizer assets, licenses, and source revision in conversion receipts.
- Decide whether to ship the documented Diffusers Euler path first or implement/match current reference Euler ancestral. Report the chosen path in demo evidence.
- Measure actual supported memory/offload/backend behavior on the selected GPU; no numeric VRAM guarantee was found or established here.

## Follow-up: exact ingestion and config cleanup (2026-11-15)

Read-only implementation research against the installed CLI's exact Creator source `b7594d88` (`cozy 0.26.0-dev+gb7594d88`). This supersedes the earlier gated-config limitation: the parent supplied an already authorized HF credential location, and small index documents now read successfully. No ingestion, weight download, publication, rental action, or runtime code change was performed by this researcher.

### Exact BF16 index rosters

At Diffusers pack revision `4791a1100e0cf97f7ed5c55c12dd71dce13f9dd3`, authenticated index reads establish:

| Index | Only the referenced shards | Tensor bytes |
| --- | --- | --- |
| `transformer/diffusion_pytorch_model.safetensors.index.json` | `diffusion_pytorch_model-00001-of-00004.safetensors` through `00004-of-00004` | 37,975,718,400 |
| `connectors/diffusion_pytorch_model.safetensors.index.json` | `diffusion_pytorch_model-00001-of-00002.safetensors` and `00002-of-00002` | 6,344,455,168 |
| `text_encoder/model.safetensors.index.json` | `model-00001-of-00005.safetensors` through `00005-of-00005` | 23,919,460,448 |

The eight-shard transformer alternative and the monolithic connectors file must remain unselected when these indexes are selected. Actual local metadata at `.task/metadata/ltx25/text_encoder/config.json` confirms hidden size3840, 48 layers, 49 hidden-state outputs, projection input188160, Transformers5.14.1, and `gemma_version=gemma4-12b-ltx-v1`.

### Why `--add` cannot repair structured config

`cozy model upload --add text_encoder/config.json=clean.json` adds/replaces an **asset**, and does not change `Component.config`. Creator `b7594d88`, `crates/tensorfs/src/import/compose.rs:253–281`, `with_assets`, copies the header's component declarations and inserts into `component.assets`; class/config/tensors are untouched. Runtime consumes structured config before the model factory, so an asset of the same filename cannot remove its rejection of `ltx_source_checkpoint.base_checkpoint`.

The CLI also exposes only one source string. `crates/tensord/src/published.rs:1654–1663` constructs `Request { sources: &[source.to_string()], source_variant: None, ... }`. A source's `:components` suffix cuts its **converted result**, after import, so it cannot prevent the whole source's duplicate-selection failure. `--base` composes existing headers; it does not widen that one-source selection interface.

### Smallest public-API rerun

Use **one `tensorfs.convert_to_cozytensors` call with an ordered source list**. Create a task-local metadata-only directory from `.task/metadata/ltx25`, with:

- The same component directory names, tokenizer JSON/config/template, scheduler config, and component configs.
- A trimmed `model_index.json` naming only `transformer`, `connectors`, `text_encoder`, `audio_vae`, `vae`, `vocoder`, `latent_upsampler`, `tokenizer`, and `scheduler` as appropriate to the package. Leave unused optional duration/enhancer/diffusion-decode components out.
- The top-level `ltx_source_checkpoint` object removed from the **text-encoder config only**. Preserve its exact original JSON as e.g. `text_encoder/upstream-training-provenance.json`, a separate asset. Preserve the source-config digest and HF source revision in the conversion receipt too. No Runtime validation weakening is needed.

Put this local metadata directory **first**, then exact pinned index/weight sources:

```python
import tensorfs

upstream = "hf://Lightricks/LTX-2.5-Diffusers@4791a1100e0cf97f7ed5c55c12dd71dce13f9dd3/"
sources = [
    str(sanitized_metadata_directory),
    upstream + "transformer/diffusion_pytorch_model.safetensors.index.json",
    upstream + "connectors/diffusion_pytorch_model.safetensors.index.json",
    upstream + "text_encoder/model.safetensors.index.json",
    upstream + "audio_vae/diffusion_pytorch_model.safetensors",
    upstream + "vae/diffusion_pytorch_model.safetensors",
    upstream + "vocoder/diffusion_pytorch_model.safetensors",
    upstream + "latent_upsampler/diffusion_pytorch_model.safetensors",
]
checkpoint = tensorfs.convert_to_cozytensors(
    sources, token=hf_token, cache_dir=task_cache, progress=progress,
)
```

This is source-derived and not executed here. The public signature is `convert_to_cozytensors(source: str | Sequence[str], *, converter=None, source_variant=None, cache_dir=None, token=None, progress=None) -> Checkpoint` (`crates/tensorfs-py/python/tensorfs/api.pyi:42–44`). An ordinary token string is passed without a `Bearer ` prefix; the native provider adds it. Do not print credentials.

The mechanism is explicit in `b7594d88` source:

- `crates/tensorfs/src/import/remote.rs:852–870`: selecting an exact index expands its `weight_map` to the full named checkpoint, includes that folder's config/sidecars, and excludes other loose weights. Selecting a referenced shard also expands to its entire indexed checkpoint.
- `crates/tensorfs/src/import/mod.rs:878–908`: each source is selected independently, then members are merged by relative path. Duplicate nonweight files keep the **first** occurrence and produce a warning. Therefore the local sanitized text-encoder config wins over the original HF config without changing importer/model code.
- `crates/tensorfs/src/import/remote.rs:615–647`: a local directory lists paths relative to that directory, matching the HF component paths.
- `crates/tensorfs/src/import/mod.rs:768–803`: multiple groups are accepted when one supplied `model_index.json` explicitly declares their distinct component folders. Two independent weight groups inside one component folder are refused. The selected exact-index roster supplies only one group per folder.

After conversion, assert the structured `checkpoint.components["text_encoder"].config` has no `ltx_source_checkpoint`, all expected components/assets are present, the exact index rosters were used, and actual tensor counts/bytes match the receipt. Then publish through the parent's existing authorized run-output/CLI flow, or use the public named Writer below with the Hub credential separately from the HF credential.

### Public Writer fallback for a checkpoint already ingested

`Writer(base=checkpoint)` retains base tensors/assets. Redeclaring a component changes its class/config while retaining tensors, so an existing good exact-selection checkpoint can be corrected without tensor decoding:

```python
component = checkpoint.components["text_encoder"]
clean = dict(component.config)
training_provenance = clean.pop("ltx_source_checkpoint", None)
writer = tensorfs.Writer(base=checkpoint, cache_dir=task_cache)
out = writer.component("text_encoder", cls=component.cls, config=clean,
                       config_file=component.config_file)
if training_provenance is not None:
    out.asset("upstream-training-provenance.json", provenance_json_bytes)
clean_checkpoint = writer.commit()
```

`provenance_json_bytes` should serialize the saved object before removing it. To publish directly instead, the same constructor accepts `model_id=destination, lane="bf16", endpoint=hub_endpoint, token=hub_token`; `commit()` publishes a new private checkpoint. A `dest=...` argument instead writes a complete portable file, including weights; avoid that when the intended operation is only header/config correction and publication. Signature: `Writer(dest=None, *, base=None, resume=None, model_id=None, lane=None, endpoint=None, token=None, cache_dir=None)`. `component(name, *, cls=None, config=None, config_file=None) -> ComponentWriter`. `asset(name, data: Buffer)`. `commit() -> Checkpoint`.

Evidence: `crates/tensorfs-py/python/tensorfs/_ext.pyi:376–404` explicitly documents that redeclaration changes class/config while tensors stay; `crates/tensorfs/src/session/writer.rs:252–277` implements the declaration. Its commit (`:322–325`) fetches carried missing base objects before sealing, so this is not a guarantee of zero transfer for a lazy remote base; it is cheap when conversion already holds those objects locally. For a new ingestion, first-source sanitized metadata is the simpler path.

### Worker-local alias to controller-authorized upload

The controller's `cozy model upload /work/task/model.cozytensors ... --on fujimaru` cannot open a path that exists only on the worker. `crates/cli/src/send.rs:454–473` resolves filesystem paths on the **controller**, snapshots them as held attachments, and immediately fails if absent there. `--on` chooses execution placement rather than path interpretation.

The smaller supported handoff uses a public local-model ref in the worker's actual TensorFS store:

```python
# On the worker, after conversion in the same store:
retained = tensorfs.Writer(
    base=checkpoint,
    model_id="local/ltx25-distilled-bf16-import",
    lane="bf16",
    cache_dir="/var/lib/tensorfs",
).commit()
```

Then, on the controller using its normal Hub authorization:

```sh
cozy model upload local/ltx25-distilled-bf16-import paul/ltx-2.5 --lane bf16 --on fujimaru
```

`Writer` forces `model_id="local/..."` to a local endpoint (`session/writer.rs:114–127`); the endpoint records its commit with `Store.update_ref(name)` (`fetch/endpoint.rs:219–237`). TensorD resolves local models through the same `Store.get_ref(model)` (`tensord/src/storage.rs:148–166`). No private store editing, portable checkpoint file, or Hub credential transfer is required.

**Fujimaru root verified read-only:** its daemon PID312 held `/var/lib/tensorfs/owner.lock` when inspected via `/proc/*/fd` over authorized SSH. `/var/lib/tensorfs` is also the documented rental default, with configurable override `COZY_TENSORFS_ROOT` / `machine.store` (`api/src/config.rs:30`, `tensord/src/machine/grant.rs:58–82`). Use that exact root for both conversion and alias Writer. An unrelated task cache produces a local ref that this machine cannot resolve.

`ctx.adopt_model` is not a direct-import bridge: it takes a Runtime `Output` from a declared derivation, not a raw TensorFS `Checkpoint`. Local alias plus regular model upload avoids that additional custody conversion.

### RC auto-update and stable qualification

There is no per-`cozy run` cohort/version override in the inspected CLI. A dev home's ordered rental follows the open RC by default. The supported machine-specific stable hold is an explicit update naming software, for example, after current machine work finishes:

```sh
cozy machine update fujimaru --runtime 0.46.0 --tensorfs 0.25.1 --tensord 0.25.0
```

This source-derived command was **not executed by this researcher**. `calcifer/src/rc.rs:210–236` records named software updates as durable manual holds; `follow_rc` exits when held (`:261`), and a held machine does not automatically move to a package's newer Runtime floor. A flagless `cozy machine update fujimaru` releases the hold. The RC CLI explicitly permits work on a released machine (`calcifer/src/update.rs:92–99`), so no shared `dev` configuration change is needed.

The actual worker was observed on Runtime `0.47.0.dev2274+g44c19b44`, TensorFS `0.25.1+dev.gec4c0ef8`, cozy `0.26.0-dev+gec4c0ef8`. Source ref `44c19b44a01f9148cb31a3df383e9c8b97b34313` is available in the Runtime repository. Comparing it with `v0.46.0` gives **no differences** in `author/_loader.py`, `_config.py`, `_context.py`, or `__init__.py`; `_model.py` only adds `Residency.keep` to a protocol. In particular `Serving` and `loader.construct(factory: Callable[[Config], T]) -> T` are unchanged. Author API adaptation for that RC is unnecessary on this evidence. RC GPU execution remains a different qualification gate from released0.46 GPU execution.
