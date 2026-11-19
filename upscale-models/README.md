# Upscale models

Dedicated upscale networks, the classic feed-forward kind ComfyUI loads through spandrel: a picture enlarged and cleaned
of blur, noise and JPEG blocks, faithfully, never redrawn. The architecture is spandrel's; the
Runtime builds it from the checkpoint and owns its weights.

`upscale_image` (the Upscale tool on a picture box) takes seedvr2's fields: `image`, the box's own version (`<self.N>`),
and `resolution` (`1080p`, `1440p`, `2160p`, the default): the picture's short side, held to that 16:9 canvas's pixels
(a 9:16 picture at 2160p is 2160x3840). The network runs one ×4 pass, and another while the picture is still at least 2×
short of that size; Lanczos takes it the rest of the way, up or down. Transparency is enlarged by Lanczos beside the colour. The result is a PNG, the box's
next version. A pass works in even tiles of at most 512 px that overlap by 32, so a tile, not the picture, sets the memory it
needs.

## Networks

| slot `model` | for | network | licence |
|---|---|---|---|
| `the-database/illustrationjanai-v3detail-dat2` (default) | drawings, official art, illustrations | [IllustrationJaNai V3detail](https://github.com/the-database/MangaJaNai/releases/tag/3.0.0), 4x DAT2 | CC-BY-NC-4.0 (the repository's LICENSE) |
| `phhofm/4xbhi-dat2-real` | photos | [4xBHI_dat2_real](https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_real), 4x DAT2 | CC-BY-4.0 |

Pick the other in the box's Models picker, or `model.model=phhofm/4xbhi-dat2-real@0.1.0`. Any DAT checkpoint serves the
slot: the network is built from the shapes its checkpoint stores, as spandrel's DAT loader reads them. Other spandrel
architectures (ESRGAN, HAT, SPAN, FDAT, …) wait on the Runtime listing a checkpoint's tensors (tracker xs-215).

```
cozy run upscale-models/upscale_image --file image=small.jpg resolution=2160p
```

Both are stored as their authors ship them, F32 (JaNai's file is named `bf16` but holds F32 tensors), and run at that precision.
On an RTX PRO 4000 (se-151): a pass on a 512 px picture takes about 7 s, on a 1024 px one about 35 s; a 225 px JPEG made
2160p (two passes) about 28 s.
