"""Small actual Runtime Output/native Writer asset inheritance proof; no inference."""
from array import array
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

import tensorfs
from cozy_runtime.weights_store import Derivation, Output

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package-root", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--portable", action="store_true", help="Use only with the new complete-file SDK")
args = parser.parse_args()
if args.out.exists():
    parser.error("receipt already exists")
processor = args.package_root / "minimax-h3/minimax_h3/processor"
names = ["transformer", "transformer_ref", "text_encoder", "vae", "audio_vae"]
processor_files = {f"processor/{p.name}": p.read_bytes() for p in processor.iterdir() if p.is_file()}
results = []
with tempfile.TemporaryDirectory(prefix="xs170-derived-asset-preservation-") as directory:
    area = Path(directory)
    with tensorfs.Writer(area / "source.cozytensors", cache_dir=area / "source-cache") as writer:
        for name in names:
            component = writer.component(name, config={"fixture_component": name})
            component.put("w", array("f", [1]), dtype="F32", shape=[1])
            component.asset("extra/runtime.data", name.encode())
            if name in names[:2]:
                component.asset("timestep-plan.json", b"old-plan")
            if name == "text_encoder":
                for path, raw in processor_files.items():
                    component.asset(path, raw)
    saved = writer.commit()
    saved.close()
    source_cache = area / ("fresh-source" if args.portable else "source-cache")
    source = tensorfs.open(area / "source.cozytensors", cache_dir=source_cache, local_files_only=True)
    source_assets = {name: {path: source.components[name].asset(path) for path in source.components[name].assets} for name in names}
    order = [(name, "w") for name in names]
    for label, base in [("produce-lane", None), ("retable", source)]:
        transaction = "sha256:" + hashlib.sha256(label.encode()).hexdigest()
        with Output(area / label, label, transaction, Derivation(base=base, sources=(source,), order=order), 32 << 20) as output:
            for name in names:
                output.config(name, {"fixture_component": name, "derived": label})
            if base is None:
                output.inherit(source)
                for name in names:
                    output.copy(name, "w", source)
            for name in names[:2]:
                output.component(name).asset("timestep-plan.json", b"new-plan")
            derived = output.commit()
            if args.portable:
                artifact = area / f"{label}.cozytensors"
                derived.write_file(artifact)
                derived.close()
                held = tensorfs.open(artifact, cache_dir=area / f"{label}-fresh", local_files_only=True)
            else:
                held = derived
            assert held.missing() == []
            assert list(held.components) == names
            for name in names:
                assert held.components[name].config == {"fixture_component": name, "derived": label}
                for path, raw in source_assets[name].items():
                    expected = b"new-plan" if path == "timestep-plan.json" else raw
                    assert held.components[name].asset(path) == expected, (label, name, path)
            results.append({"path": label, "all_source_component_assets_preserved": True,
                            "processor_files": len(processor_files), "component_count": len(held.components),
                            "intentional_plan_overwrite_only": True, "fresh_complete_file": args.portable,
                            "processor_asset_hashes": {path: hashlib.sha256(raw).hexdigest() for path, raw in processor_files.items()}})
            held.close()
    source.close()
receipt = {"runtime": version("cozy-runtime"), "tensorfs": version("tensorfs"),
           "scope": "actual Runtime Output/native Writer contracts, no H3 tensor calculations or GPU",
           "source_edits": False, "publication": False, "portable_transport_claimed": args.portable,
           "results": results}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
