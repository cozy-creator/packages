#!/usr/bin/env bash
# rc-runtime.sh [pure|native|tensorfs]: the URL of the open RC's newest Runtime wheel (cozy-runtime's rolling prerelease rc-<x.y.z>),
# pure (what a lock takes) or native (Linux x86_64, with kernels: what a machine runs), or of the TensorFS wheel the RC's Runtime
# is locked with (cozy-creator's rc-<x.y.z>, Linux x86_64: the build its floor names, xs-216). The open RC is this computer's cozy's:
# the minor after a release (0.25.0 → 0.26.0) or a dev build's own (0.26.0-dev → 0.26.0). The prerelease keeps the RC's
# newest 20 builds, its SHA256SUMS newest first, so the first match is the newest; a package still names only its floor
# (`cozy-runtime>=0.47.0.dev0`, xs-194).
set -euo pipefail
v=$(cozy --version | awk '{print $2}'); x=${v%%-*}; IFS=. read -r major minor _ <<< "$x"
[[ $v == *-* ]] || minor=$((minor + 1))
repo=cozy-runtime
case ${1:-pure} in
  pure) pattern='^cozy_runtime-.*-py3-none-any[.]whl$' ;;
  native) pattern='^cozy_runtime-.*manylinux_2_28_x86_64[.]whl$' ;;
  # The TensorFS build the RC's Runtime is locked and type-checked with (its [tool.uv.sources], xs-216): one build for the Runtime, every
  # package lock and the type checks, which rc-dev keeps while the Runtime's lock names it.
  tensorfs)
    pyproject=$(curl -fsSL "https://raw.githubusercontent.com/cozy-creator/cozy-runtime/rc/$major.$minor.0/pyproject.toml")
    url=$(sed -n 's/^tensorfs = { url = "\([^"]*\)".*/\1/p' <<< "$pyproject")
    [ -n "$url" ] || { echo "cozy-runtime's rc/$major.$minor.0 names no TensorFS build" >&2; exit 1; }
    echo "$url"; exit 0 ;;
  *) echo "usage: scripts/rc-runtime.sh [pure|native|tensorfs]" >&2; exit 2 ;;
esac
base=https://github.com/cozy-creator/$repo/releases/download/rc-$major.$minor.0
# SHA256SUMS.old while rc-dev swaps SHA256SUMS in by rename (cl-1128): it never deletes one under a reader. Read whole, then
# matched: an awk that exits at its match leaves curl writing into a closed pipe (exit 23 under pipefail: gate 44223772507).
sums=$(curl -fsSL "$base/SHA256SUMS" || curl -fsSL "$base/SHA256SUMS.old")
file=$(awk -v p="$pattern" '$2 ~ p && !f {f = $2} END {print f}' <<< "$sums")
[ -n "$file" ] || { echo "$repo's rc-$major.$minor.0 holds no ${1:-pure} wheel" >&2; exit 1; }
echo "$base/$file"
