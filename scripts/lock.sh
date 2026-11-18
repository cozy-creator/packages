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
# A floor only the open RC meets (`cozy-runtime>=0.47.0.dev0`, which no release does yet) takes the RC's newest build, by URL
# in this copy alone: what the RC's machines run (Calcifer's lock does the same, floor::rc_runtime). Beside it, the RC's newest
# TensorFS build, which that Runtime floors on (`tensorfs>=0.27.0.dev<n>`, xs-216): by URL for Linux x86_64, the one platform
# the RC builds it for; every other platform's row is the index's. The package never names
# either wheel, which each merge replaces; after the release, the floors' own releases on PyPI meet them.
lock() { (cd "$s" && uvx -q uv@0.12.24 lock --python 3.12 "$@"); }
if ! lock "$@" 2> "$s/lock.err"; then
  grep -qE '"cozy-runtime(\[[^]]*\])?>=[0-9.]+\.dev[0-9]+' "$s/pyproject.toml" && ! grep -qE '^cozy-runtime = ' "$s/pyproject.toml" \
    || { cat "$s/lock.err" >&2; exit 1; }
  grep -qx '\[tool\.uv\.sources\]' "$s/pyproject.toml" || printf '\n[tool.uv.sources]\n' >> "$s/pyproject.toml"
  sed -i "/^\[tool\.uv\.sources\]\$/a cozy-runtime = { url = \"$("$(dirname "$0")/rc-runtime.sh" pure)\" }" "$s/pyproject.toml"
  if ! grep -qE '^tensorfs = ' "$s/pyproject.toml"; then
    sed -i "/^\[tool\.uv\.sources\]\$/a tensorfs = { url = \"$("$(dirname "$0")/rc-runtime.sh" tensorfs)\", marker = \"sys_platform == 'linux' and platform_machine == 'x86_64'\" }" "$s/pyproject.toml"
    # uv reads a source only for a dependency the project names itself.
    grep -qE '^ *"tensorfs[][ <>=!~;"]' "$s/pyproject.toml" || sed -i '0,/^dependencies = \[$/s//dependencies = [\n    "tensorfs",/' "$s/pyproject.toml"
  fi
  lock "$@"
fi
cp "$s/uv.lock" "$p/uv.lock"
