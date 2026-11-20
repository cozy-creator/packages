"""Convert the pinned MIT Practical-RIFE 4.26 archive to a native TensorFS artifact."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import tensorfs
import torch

from anime_interpolation.net import IFNet

ARCHIVE_SHA256 = "c2452dd2b244947d4be580156bbead60d6b72af5736860f7d6b3f99648c9c4cc"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    with args.archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != ARCHIVE_SHA256:
        parser.error(f"Expected official RIFE 4.26 archive SHA-256 {ARCHIVE_SHA256}, got {digest}")
    with ZipFile(args.archive) as archive:
        contents = io.BytesIO(archive.read("train_log/flownet.pkl"))
    weights = torch.load(contents, map_location="cpu", weights_only=True)
    model = IFNet()
    state = {key.removeprefix("module."): value for key, value in weights.items()}
    # The upstream archive includes the training-only teacher. Every inference tensor is required.
    model.load_state_dict({key: state[key] for key in model.state_dict()})
    with tensorfs.Writer(args.checkpoint) as writer:
        component = writer.component("rife", config={"architecture": "Practical-RIFE 4.26"})
        for key, tensor in model.state_dict().items():
            component.put(key, tensor.contiguous())
        component.asset(
            "LICENSE",
            Path(__file__).with_name("anime_interpolation").joinpath("LICENSE.rife").read_bytes(),
        )
        writer.commit()


if __name__ == "__main__":
    main()
