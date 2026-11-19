# What upscale-models calls of spandrel 0.4.2's DAT (architectures/DAT/__arch/DAT.py), which ships no py.typed.
from torch import nn

class DAT(nn.Module):
    def __init__(
        self,
        img_size: int = ...,
        in_chans: int = ...,
        embed_dim: int = ...,
        split_size: list[int] = ...,
        depth: list[int] = ...,
        num_heads: list[int] = ...,
        expansion_factor: float = ...,
        qkv_bias: bool = ...,
        qk_scale: float | None = ...,
        drop_rate: float = ...,
        attn_drop_rate: float = ...,
        drop_path_rate: float = ...,
        act_layer: type[nn.Module] = ...,
        norm_layer: type[nn.Module] = ...,
        use_chk: bool = ...,
        upscale: int = ...,
        img_range: float = ...,
        resi_connection: str = ...,
        upsampler: str = ...,
    ) -> None: ...
