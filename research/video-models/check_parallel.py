"""CPU parity for PAN2/LTX Ulysses, before full-model NCCL qualification.

Run in an environment containing both packages' Diffusers/Runtime dependencies:
  python research/video-models/check_parallel.py --world 2 --out /path/to/new-evidence

The prompt/joint attention sequences are uneven; video/audio output lengths divide
2/4/8 because Gloo cannot all-gather unequal output sizes. NCCL must additionally
qualify production frame counts. --original reproduces LTX's upstream rotary-plan
failure. These are real transformer forwards and collectives, not fake workers.
"""

import argparse
import copy
import json
import pathlib
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
from diffusers.models.transformers.transformer_pan2 import PAN2Transformer3DModel
from cozy_runtime.internal.parallel.cp import CpComms, _install_on_component
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ltx-2.5"))
from ltx_2_5.parallel import install


def ltx():
    return LTX2VideoTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_layers=2,
        num_attention_heads=8,
        attention_head_dim=8,
        cross_attention_dim=64,
        caption_channels=32,
        audio_in_channels=2,
        audio_out_channels=2,
        audio_num_attention_heads=8,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=32,
        audio_scale_factor=4,
        rope_type="split",
        rope_double_precision=False,
        use_prompt_embeddings=False,
        cross_attn_mod=False,
        audio_cross_attn_mod=False,
        gated_attn=False,
        audio_gated_attn=False,
    )


def pan():
    return PAN2Transformer3DModel(
        patch_size=(1, 2, 2),
        in_channels=9,
        out_channels=4,
        num_attention_heads=8,
        attention_head_dim=16,
        num_layers=2,
        num_refiner_layers=1,
        mlp_ratio=2,
        context_mlp_ratio=2,
        refiner_mlp_ratio=2,
        text_embed_dim=32,
        rope_axes_dim=(4, 6, 6),
    )


def worker(rank, world, address, out, original):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=address, rank=rank, world_size=world)
    records = []
    try:
        for kind in ["ltx", "pan"]:
            torch.manual_seed(7)
            model = (ltx() if kind == "ltx" else pan()).eval()
            model.set_attention_backend("native")
            for firstframe in [False, True]:
                torch.manual_seed(31)
                if kind == "ltx":
                    kwargs = dict(
                        hidden_states=torch.randn(1, 24, 4),
                        audio_hidden_states=torch.randn(1, 16, 2),
                        encoder_hidden_states=torch.randn(1, 11, 64),
                        audio_encoder_hidden_states=torch.randn(1, 13, 32),
                        timestep=torch.tensor([700.0]),
                        audio_timestep=torch.tensor([700.0]),
                        num_frames=3,
                        height=2,
                        width=4,
                        audio_num_frames=16,
                        return_dict=False,
                    )
                    kwargs["encoder_attention_mask"] = torch.tensor([[1.0] * 9 + [0.0] * 2])
                    kwargs["audio_encoder_attention_mask"] = torch.tensor([[1.0] * 10 + [0.0] * 3])
                    if firstframe:
                        kwargs["timestep"] = torch.tensor([[0.0] * 8 + [700.0] * 16])
                else:
                    kwargs = dict(
                        hidden_states=torch.randn(1, 9, 6, 4, 4),
                        encoder_hidden_states=torch.randn(1, 13, 32),
                        timestep=torch.tensor([700.0]),
                        return_dict=False,
                    )
                    kwargs["hidden_states"][:, 4:] = 0
                    if firstframe:
                        kwargs["hidden_states"][:, 4:8, 0] = torch.randn(1, 4, 4, 4)
                        kwargs["hidden_states"][:, 8, 0] = 1
                with torch.inference_mode():
                    reference = model(**kwargs)
                sharded = copy.deepcopy(model)
                if kind == "ltx" and not original:
                    install(sharded)
                _install_on_component(
                    sharded,
                    degree=world,
                    comms=CpComms(dist.group.WORLD, rank, torch.device("cpu")),
                )
                with torch.inference_mode():
                    actual = sharded(**kwargs)
                errors = []
                for a, b in zip(actual, reference, strict=True):
                    assert a.shape == b.shape, (kind, a.shape, b.shape)
                    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
                    errors.append(float((a - b).abs().max()))
                records.append(
                    {
                        "model": kind,
                        "image_conditioned_timesteps": firstframe,
                        "max_abs": errors,
                        "world": world,
                        "rank": rank,
                    }
                )
                del sharded
            del model
        pathlib.Path(out, f"rank-{rank}.json").write_text(json.dumps(records, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--world", type=int, default=2)
    p.add_argument("--original", action="store_true")
    p.add_argument("--out", type=pathlib.Path, required=True)
    a = p.parse_args()
    a.out.mkdir(exist_ok=False, parents=True)
    rendezvous = (a.out / "gloo").resolve()
    mp.spawn(
        worker,
        args=(a.world, "file://" + str(rendezvous), str(a.out), a.original),
        nprocs=a.world,
        join=True,
    )
    print("PASS", a.world, "ranks")
