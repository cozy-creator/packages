from collections.abc import Sequence

from PIL import Image
from torch import Tensor

from . import functional as functional

class Resize:
    def __init__(self, size: int | Sequence[int]) -> None: ...

class ToTensor:
    def __call__(self, pic: Image.Image) -> Tensor: ...

class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None: ...

class ToPILImage:
    def __init__(self, mode: str | None = None) -> None: ...
    def __call__(self, pic: Tensor) -> Image.Image: ...

class Compose:
    """As the packages compose them: a picture in, a tensor out."""

    def __init__(self, transforms: Sequence[Resize | ToTensor | Normalize]) -> None: ...
    def __call__(self, img: Image.Image) -> Tensor: ...
