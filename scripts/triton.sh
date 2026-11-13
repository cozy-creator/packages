#!/usr/bin/env bash
# triton.sh <package> [out] [runtime wheel]: the Triton sets the package's import declares, compiled on this CPU for every
# card row, into `_triton` beside each top-level module that declares one (its home, where machines look; minimax-h3's
# object is h3, its sets are minimax_h3's), under `out` (default: the package's own directory). A release ships them
# (`task publish`); a gate or session running the package from source builds them into its checkout first, so the source
# Calcifer uploads carries the same kernels and its first request serves them as a Hub install's does.
# They compile with the Runtime the package's lock pins (an RC build: the RC's newest; or the wheel given, a URL or file), Torch $IMAGE_TORCH,
# the Hub's worker image's: that Runtime and Triton key them. Only the Runtime's prebuild entry is called, so a Runtime that
# moves it fails here rather than shipping nothing (se-117). A package whose source constructs no TritonKernels builds nothing.
set -euo pipefail
p=${1:?usage: scripts/triton.sh <package> [out] [runtime wheel]}; out=$(realpath -m "${2:-$p}"); runtime=${3:-}
grep -rqs --include='*.py' 'TritonKernels(' "$p" || exit 0
torch=${IMAGE_TORCH:-torch==2.15.0 torchvision==0.29.2}  # the current worker image's (its tag): a lock carries none (cl-715)
m=$(sed -n '/cozy\.application/,/^\[/s/^default = "\([A-Za-z0-9_.]*\):.*/\1/p' "$p/pyproject.toml")
[ -n "$m" ] || { echo "$p: no cozy.application entry point" >&2; exit 1; }
e=$(mktemp -d); trap 'rm -rf "$e"' EXIT
bind=(); ! grep -qs '= { *index = "tensorhub"' "$p/pyproject.toml" || bind=(--index "tensorhub=${LOCK_HUB:-https://tensorhub.com}/v1/index/_/simple/")
(cd "$p" && UV_PROJECT_ENVIRONMENT=$e/env uv sync -q --frozen --no-dev --python 3.12 --no-install-package cozy-runtime "${bind[@]}")
# shellcheck disable=SC2086 # the requirements, one word each
VIRTUAL_ENV=$e/env uv pip install -q $torch
# The lock's Runtime from PyPI; one it took from the open RC (by URL, xs-194) is the RC's newest, which its machines run: the
# wheel it named is gone after the next Runtime merge.
if [ -z "$runtime" ]; then
  locked=$(grep -A2 -x 'name = "cozy-runtime"' "$p/uv.lock" | sed -n 's/^version = "\(.*\)"$/\1/p' | head -1)
  if grep -A2 -x 'name = "cozy-runtime"' "$p/uv.lock" | grep -q '^source = { url = '; then runtime=$("$(dirname "$0")/rc-runtime.sh" native)
  else runtime=cozy-runtime==$locked; fi
fi
case $runtime in cozy-runtime==*) VIRTUAL_ENV=$e/env uv pip install -q --no-deps "$runtime" ;; *) VIRTUAL_ENV=$e/env uv pip install -q --no-deps "cozy-runtime @ $runtime" ;; esac
# Sets built before, for older source, go: a machine would take their manifest for this source's.
find "$out" -type d -name _triton -prune -exec rm -rf {} + 2>/dev/null || true
prebuild='import importlib, sys
from pathlib import Path
importlib.import_module(sys.argv[1])
from cozy_runtime.internal import triton_kernels, triton_prebuild
here, out = Path.cwd().resolve(), Path(sys.argv[2])
for home, module in sorted(sys.modules.copy().items()):
    file = getattr(module, "__file__", None)
    if "." in home or not file or not Path(file).resolve().is_relative_to(here):
        continue
    said = triton_prebuild.build(out / Path(file).resolve().parent.relative_to(here) / triton_kernels.SHIPPED, None, home)
    rows = said["sets"]
    left = sorted(row for row, took in rows.items() if isinstance(took, str))
    if rows:
        print("%s: %d kernels, %d KiB, %d of %d set rows" % (home, said["kernels"], said["bytes"] // 1024, len(rows) - len(left), len(rows)), file=sys.stderr)
    for row in left:
        print("  %s %s" % (row, rows[row]), file=sys.stderr)
    if rows and not said["kernels"]:
        sys.exit(home + ": its Triton sets compiled for no card row")'
(cd "$p" && CUDA_VISIBLE_DEVICES='' nice -n 19 "$e/env/bin/python" -I -c "$prebuild" "$m" "$out") > /dev/null
