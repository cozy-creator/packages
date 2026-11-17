"""Upstream BiRefNet, with construction and weight residency owned by Runtime."""

from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms

from cozy_runtime.author import Config, Loader, Model, Serving, uses_components

from .upstream.BiRefNet_config import BiRefNetConfig
from .upstream.birefnet import BiRefNet


class BiRefNetGraph:
    def __init__(self, config: Config) -> None:
        if config.mapping().get("bb_pretrained") is not False:
            raise ValueError("BiRefNet's checkpoint is the complete model (bb_pretrained: false)")
        model: torch.nn.Module = BiRefNet(config=BiRefNetConfig(bb_pretrained=False)).eval().half()
        self.components = {"model": model}


def build_birefnet(config: Config) -> BiRefNetGraph:
    return BiRefNetGraph(config)


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    """`cv2.blur(x, (r, r))` of `[C, H, W]`: each pixel the mean of the r x r window at offset -(r // 2), the border
    mirrored without repeating its edge (OpenCV's default BORDER_REFLECT_101)."""
    for dim in (1, 2):
        n = x.shape[dim]
        at = torch.arange(-(r // 2), n + r - 1 - r // 2, device=x.device)
        if n > 1:
            at = at.remainder(2 * (n - 1))
            at = torch.where(at >= n, 2 * (n - 1) - at, at)
        else:
            at = torch.zeros_like(at)
        sums = torch.nn.functional.pad(x.index_select(dim, at).double().cumsum(dim), (1, 0) if dim == 2 else (0, 0, 1, 0))
        x = ((sums.narrow(dim, r, n) - sums.narrow(dim, 0, n)) / r).float()
    return x


def _estimate(image: torch.Tensor, f: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One pass of the official handler's `FB_blur_fusion_foreground_estimator` (Photoroom's fast foreground
    estimation): the foreground and the blurred background."""
    blurred_alpha = _box(alpha, r)
    blurred_f = _box(f * alpha, r) / (blurred_alpha + 1e-5)
    blurred_b = _box(b * (1 - alpha), r) / ((1 - blurred_alpha) + 1e-5)
    f = blurred_f + alpha * (image - alpha * blurred_f - (1 - alpha) * blurred_b)
    return f.clamp(0, 1), blurred_b


def refine_foreground(image: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """The official handler's `refine_foreground` (BiRefNet's `handler.py`, r = 90 then 6): the subject's own colour
    at its soft edge, without the old background's, as `[3, H, W]` in 0..1 from `image` `[3, H, W]` and `alpha`
    `[1, H, W]`, both 0..1."""
    f, blurred_b = _estimate(image, image, image, alpha, 90)
    return _estimate(image, f, blurred_b, alpha, 6)[0]


class BackgroundRemovalModel(Model[BiRefNetGraph]):
    serving = Serving(encoded_leaves=False)

    graph: BiRefNetGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(build_birefnet)

    @uses_components("model")
    def cutout(self, image: Image.Image) -> tuple[Image.Image, Image.Image]:
        """The subject's refined colours and its soft mask, at the image's size: the official handler's
        (`handler.py`): its input transform, the final sigmoid, the mask resized as Pillow's default (bicubic), then
        `refine_foreground` at its truncation to 8 bits."""
        model = self.graph.components["model"]
        device = torch.device(str(self.device))
        prepare = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        with torch.inference_mode():
            pixels = prepare(image).unsqueeze(0).to(device=device, dtype=torch.float16)
            predictions: list[torch.Tensor] = model(pixels)
            probability = predictions[-1].sigmoid().float().cpu()[0].squeeze(0)
            mask = transforms.ToPILImage()(probability).resize(image.size, Image.Resampling.BICUBIC)
            rgb = transforms.functional.pil_to_tensor(image).to(device=device, dtype=torch.float32) / 255
            alpha = transforms.functional.pil_to_tensor(mask).to(device=device, dtype=torch.float32) / 255
            foreground = (refine_foreground(rgb, alpha) * 255).to(torch.uint8).cpu()
        return transforms.functional.to_pil_image(foreground), mask
