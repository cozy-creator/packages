"""Verify installed stable candidate wheels and their source-only public interfaces."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[1]

import minimax_h3.official
import h3_tables.job

for name, expected in {"cozy-runtime": "0.44.2", "tensorfs": "0.23.1", "msgspec": "0.21.1", "minimax-h3": "0.9.5", "minimax-h3-tools": "0.4.1"}.items():
    assert importlib.metadata.version(name) == expected, name
assert Path(minimax_h3.official.__file__).is_relative_to(Path(sys.prefix))
assert Path(h3_tables.job.__file__).is_relative_to(Path(sys.prefix))

test_path = root / "checks/test_stable_h3_checkpoint_data.py"
text = test_path.read_text().replace('sys.path[:0] = [str(ROOT / "minimax-h3"), str(ROOT / "minimax-h3-tools/src")]', "")
module = types.ModuleType("stable_installed_wheel_tests")
module.__file__ = str(test_path)
sys.modules[module.__name__] = module
exec(compile(text, str(test_path), "exec"), module.__dict__)
suite = unittest.defaultTestLoader.loadTestsFromModule(module)
assert suite.countTestCases() == 12
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise SystemExit(1)


def defaults(value):
    if isinstance(value, dict):
        return {key: replacements.get(item, item) if key == "default" and isinstance(item, str) else defaults(item) for key, item in value.items()}
    if isinstance(value, list):
        return [defaults(item) for item in value]
    return value


replacements = {
    "minimaxai/minimax-h3@1.0.0-rc.7": "minimaxai/minimax-h3@1.0.0-rc.8",
    "lbh-123-ai/h3-latent-upscaler@0.1.0": "lbh-123-ai/h3-latent-upscaler@0.1.1",
}
for package, baseline in [("minimax-h3", "0.9.3"), ("minimax-h3-tools", "0.4.0")]:
    current = json.loads(subprocess.check_output([str(Path(sys.executable).with_name("cozy-runtime")), "--json", "--dir", str(root / package), "describe"], text=True))
    original = json.loads((root / f"checks/fixtures/{package}-{baseline}-interface.json").read_text())
    assert current == defaults(original), package
    (args.out / f"{package}-interface.json").write_text(json.dumps(current, indent=2) + "\n")
(args.out / "installed-wheel-receipt.json").write_text(json.dumps({"tests_passed": result.testsRun, "interfaces_preserved_except_updated_model_defaults": True,
    "runtime": importlib.metadata.version("cozy-runtime"), "tensorfs": importlib.metadata.version("tensorfs"),
    "h3": importlib.metadata.version("minimax-h3"), "tools": importlib.metadata.version("minimax-h3-tools"),
    "hub_or_database_used": False}, indent=2) + "\n")
