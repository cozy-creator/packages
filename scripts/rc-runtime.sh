#!/usr/bin/env bash
# rc-runtime.sh [pure|native]: the URL of the open RC's newest Runtime wheel (cozy-runtime's rolling prerelease rc-<x.y.z>),
# pure (what a lock takes) or native (Linux x86_64, with kernels: what a machine runs). The open RC is this computer's cozy's:
# the minor after a release (0.25.0 → 0.26.0) or a dev build's own (0.26.0-dev → 0.26.0). Each Runtime merge into the RC
# replaces its wheels, so nothing keeps this URL: a package names only its floor (`cozy-runtime>=0.47.0.dev0`, xs-194).
set -euo pipefail
v=$(cozy --version | awk '{print $2}'); x=${v%%-*}; IFS=. read -r major minor _ <<< "$x"
[[ $v == *-* ]] || minor=$((minor + 1))
base=https://github.com/cozy-creator/cozy-runtime/releases/download/rc-$major.$minor.0
case ${1:-pure} in
  pure) pattern='-py3-none-any[.]whl$' ;;
  native) pattern='manylinux_2_28_x86_64[.]whl$' ;;
  *) echo "usage: scripts/rc-runtime.sh [pure|native]" >&2; exit 2 ;;
esac
file=$(curl -fsSL "$base/SHA256SUMS" | awk -v p="$pattern" '$2 ~ p {print $2; exit}')
[ -n "$file" ] || { echo "cozy-runtime's rc-$major.$minor.0 holds no ${1:-pure} wheel" >&2; exit 1; }
echo "$base/$file"
