"""Hosted installed-wheel tests; all model inputs are local native fixtures."""

import argparse
import ctypes
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


def refuse_ip_sockets() -> None:
    class Filter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    instructions = (Filter * 7)(Filter(0x20, 0, 0, 0), Filter(0x15, 0, 4, 41), Filter(0x20, 0, 0, 16),
                                Filter(0x15, 1, 0, 2), Filter(0x15, 0, 1, 10),
                                Filter(0x06, 0, 0, 0x00050001), Filter(0x06, 0, 0, 0x7FFF0000))
    program = Program(len(instructions), instructions)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) or libc.prctl(22, 2, ctypes.byref(program), 0, 0):
        raise OSError(ctypes.get_errno(), "cannot isolate hosted tokenizer fixtures")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--baseline-control", action="store_true")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
if args.baseline_control:
    sys.path.insert(0, str(args.baseline / "sdxl"))
refuse_ip_sockets()

import anima.model
import sdxl

root = Path(__file__).resolve().parents[1]
for name, expected in {"cozy-runtime": "0.44.0", "tensorfs": "0.23.0", "anima": "0.7.1", "sdxl": "0.8.1"}.items():
    assert importlib.metadata.version(name) == expected, name
assert Path(anima.model.__file__).is_relative_to(Path(sys.prefix))
assert Path(sdxl.__file__).is_relative_to(args.baseline if args.baseline_control else Path(sys.prefix))

path = root / "checks/test_stable_tokenizer_data.py"
spec = importlib.util.spec_from_file_location("stable_tokenizer_tests", path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
if args.baseline_control:
    suite = unittest.TestSuite([module.SdxlCheckpointDataTests("test_absent_checkpoint_pair_is_not_replaced_by_package_data")])
else:
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    assert suite.countTestCases() == 7
result = unittest.TextTestRunner(verbosity=2).run(suite)
if args.baseline_control:
    assert len(result.failures) == 1 and not result.errors, "baseline must reproduce its bundled fallback, not fail for an unrelated reason"
    (args.out / "baseline-refusal-control.json").write_text(json.dumps({"expected_failure_reproduced": True, "assertion": "missing checkpoint pair must not use bundled CLIP data"}, indent=2) + "\n")
    raise SystemExit(0)
if not result.wasSuccessful():
    raise SystemExit(1)

runtime = str(Path(sys.executable).with_name("cozy-runtime"))
for package in ("anima", "sdxl"):
    candidate = json.loads(subprocess.check_output([runtime, "--json", "--dir", str(root / package), "describe"], text=True))
    baseline = json.loads(subprocess.check_output([runtime, "--json", "--dir", str(args.baseline / package), "describe"], text=True))
    assert candidate == baseline, package
    (args.out / f"{package}-interface.json").write_text(json.dumps(candidate, indent=2) + "\n")
(args.out / "installed-tokenizer-receipt.json").write_text(json.dumps({
    "tests": result.testsRun, "all_passed": True, "interfaces_unchanged": True,
    "runtime": importlib.metadata.version("cozy-runtime"), "tensorfs": importlib.metadata.version("tensorfs"),
    "source_cache_removed_before_read": True, "fresh_local_only_native_views": True,
    "ip_socket_creation_refused": True, "portable_file_support_claimed": False,
    "hub_database_or_gpu_used": False,
}, indent=2) + "\n")
