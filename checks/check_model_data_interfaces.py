"""Check model defaults in genuine static describe output; never import or execute a package."""
import json
from pathlib import Path
import sys

proof = Path(sys.argv[1])
interfaces = {}
for package in ("anima", "sdxl", "minimax-h3"):
    body = json.loads((proof / f"interface-{package}.json").read_text())
    assert body["format"] == "cozy.package.interface/1", package
    assert body["entrypoints"], package
    interfaces[package] = body

entries = {entry["name"]: entry for entry in interfaces["minimax-h3"]["entrypoints"]}


def default(entry, parameter):
    slots = [slot for slot in entries[entry]["models"] if slot["path"].rsplit(".", 1)[-1] == parameter]
    assert len(slots) == 1, (entry, parameter, slots)
    return slots[0]["default"]


base_slots = [
    (entry["name"], slot)
    for entry in entries.values() for slot in entry.get("models", [])
    if slot["path"].rsplit(".", 1)[-1] == "base_model" and "default" in slot
]
assert len(base_slots) >= 3, base_slots
for entry, slot in base_slots:
    assert slot["default"] == "minimaxai/minimax-h3@1.0.0-rc.8", (entry, slot)
upscaler_slots = [
    (entry["name"], slot)
    for entry in entries.values() for slot in entry.get("models", [])
    if slot["path"].rsplit(".", 1)[-1] == "latent_upscaler" and "default" in slot
]
assert upscaler_slots
for entry, slot in upscaler_slots:
    assert slot["default"] == "lbh-123-ai/h3-latent-upscaler@0.1.1", (entry, slot)
for parameter in ("model", "upsampler"):
    assert default("image", parameter) == "prunaai/p-h3-image@0.1.0", parameter

(proof / "interface-checks.json").write_text(json.dumps({
    "packages": sorted(interfaces), "h3_base_default": "minimaxai/minimax-h3@1.0.0-rc.8",
    "h3_upscaler_default": "lbh-123-ai/h3-latent-upscaler@0.1.1",
    "p_h3_image_default_unchanged": "prunaai/p-h3-image@0.1.0", "package_code_executed": False,
}, indent=2) + "\n")
print("Static interfaces and all H3/P-H3 model defaults passed")
