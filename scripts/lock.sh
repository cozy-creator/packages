#!/usr/bin/env bash
# lock.sh <package> [uv lock args]: the package's uv.lock as `cozy package publish --root` makes it (cozy-creator
# crates/calcifer/src/hub/author.rs, `Stage`): uv, in a copy whose Torch layer is the machine's and whose `tensorhub`
# sources take the root index of $LOCK_HUB (production unless set), with the uv Calcifer pins (platform::tools::UV_VERSION).
set -euo pipefail
p=${1:?usage: scripts/lock.sh <package> [uv lock args, e.g. --upgrade-package=qwen-image-2]}; shift
s=$(mktemp -d); trap 'rm -rf "$s"' EXIT
cp "$p/pyproject.toml" "$s/"; [ ! -f "$p/uv.lock" ] || cp "$p/uv.lock" "$s/"
grep -qx '\[tool\.uv\]' "$s/pyproject.toml" || printf '\n[tool.uv]\n' >> "$s/pyproject.toml"
sed -i '/^\[tool\.uv\]$/a exclude-dependencies = ["torch", "torchvision", "torchaudio", "triton"]' "$s/pyproject.toml"
if grep -q '= { *index = "tensorhub"' "$s/pyproject.toml"; then
  printf '\n[[tool.uv.index]]\nname = "tensorhub"\nurl = "%s/v1/index/_/simple/"\nexplicit = true\n' "${LOCK_HUB:-https://tensorhub.com}" >> "$s/pyproject.toml"
fi
(cd "$s" && uvx -q uv@0.12.24 lock --python 3.12 "$@")
cp "$s/uv.lock" "$p/uv.lock"
