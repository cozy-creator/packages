"""H3's construction config: one section per component (D4 names), stored as each
component's config; Runtime reassembles them into one `{component: config}` mapping. The
package's copy holds the supported constructor only; a DiT's table rows come from its plan."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from cozy_runtime.author import canonical_json

from .kernel import H3Topology
from .plans import Task, TimestepPlan

COMPONENTS = ("transformer", "transformer_ref", "text_encoder", "vae", "audio_vae")
DIT_COMPONENT: dict[Task, str] = {"fl2va": "transformer", "ref2va": "transformer_ref"}
_TEXT_CONDITIONER_CONFIG = {
    "source_architecture": "Qwen3VLForConditionalGeneration",
    "retained_decoder_layers": 50,
    "conditioning_hidden_state": 50,
    "output": "pre_norm",
    "language_model_head": False,
}


def validate_text_conditioner(config: Mapping[str, object]) -> None:
    if config.get("cozy_h3") != _TEXT_CONDITIONER_CONFIG:
        raise ValueError(
            "text_encoder config does not declare the exact 50-layer pre-norm conditioner"
        )


def _sections(raw: bytes, label: str) -> dict[str, dict[str, object]]:
    try:
        value = canonical_json.decode(raw)
    except ValueError as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if not isinstance(value, dict) or set(value) != set(COMPONENTS):
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} sections are {actual}, expected {sorted(COMPONENTS)}")
    if not all(isinstance(section, dict) for section in value.values()):
        raise TypeError(f"{label} has a section that is not a mapping")
    return cast(dict[str, dict[str, object]], value)


def parse_full_config(raw: bytes) -> dict[str, dict[str, object]]:
    """The official constructor sections: both DiTs one architecture, the 50-layer conditioner."""
    sections = _sections(raw, "official H3 model config")
    fl, ref = (
        {k: v for k, v in sections[c].items() if k != "_diffusers_version"}
        for c in DIT_COMPONENT.values()
    )
    if fl != ref:
        raise ValueError("FL2VA and Ref2VA do not declare one identical DiT architecture config")
    H3Topology.from_config(fl)
    validate_text_conditioner(sections["text_encoder"])
    return sections


def dual_full_config(sections: Mapping[str, Mapping[str, object]]) -> bytes:
    document = dict(sections)
    for task, component in DIT_COMPONENT.items():
        document[component] = {**sections[component], "cozy_h3": {"task": task, "modulation": "full"}}
    return canonical_json.encode(document)


def task_config(
    sections: Mapping[str, Mapping[str, object]], plan: TimestepPlan
) -> dict[str, dict[str, object]]:
    component = DIT_COMPONENT[plan.task]
    config = dict(sections[component])
    config["cozy_h3"] = {
        "task": plan.task,
        "modulation": "adaln-pruned",
        "table_keys": plan.table_keys,
    }
    return {component: config}


def dual_adaln_pruned_config(
    sections: Mapping[str, Mapping[str, object]], fl2va: TimestepPlan, ref2va: TimestepPlan
) -> bytes:
    if fl2va.task != "fl2va" or ref2va.task != "ref2va":
        raise ValueError("dual AdaLN-pruned config requires each task's own table layout")
    return canonical_json.encode(
        {**sections, **task_config(sections, fl2va), **task_config(sections, ref2va)}
    )

