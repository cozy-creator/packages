"""Model selections shared by the shots in one rendering."""

from __future__ import annotations

import re

import msgspec
from cozy_runtime.author import InvalidRequest


class RenderProvenance(msgspec.Struct, frozen=True):
    model_manifest: str
    turbo_lora_manifest: str = ""


def provenance(model_manifest: str, turbo_lora_manifest: str = "") -> RenderProvenance:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", model_manifest):
        raise InvalidRequest("H3 rendering needs its exact model manifest", code="render_model")
    if turbo_lora_manifest and not re.fullmatch(r"sha256:[0-9a-f]{64}", turbo_lora_manifest):
        raise InvalidRequest("H3 turbo needs its exact adapter manifest", code="render_model")
    return RenderProvenance(model_manifest, turbo_lora_manifest)


def compatible_base(actual: RenderProvenance, previous: RenderProvenance | None) -> None:
    if previous is not None and actual.model_manifest != previous.model_manifest:
        raise InvalidRequest(
            "shots in one rendering must use the same base model",
            code="render_provenance",
        )


def context_provenance(renderer: RenderProvenance) -> str:
    """Record model selections for native context compatibility without code hashes."""
    return msgspec.json.encode(renderer).decode("utf-8")
