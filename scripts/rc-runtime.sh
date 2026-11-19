#!/usr/bin/env bash
# rc-runtime.sh open | pure [pyproject.toml] | native | tensorfs: the open RC this checkout is on, or a URL of its builds.
#   open      the RC's version (0.28.0): the newest origin/rc/<x.y.z> whose history HEAD shares past master's (the RC's branch
#             or one cut from it); nothing off one (master, a release branch) or without origin/master to tell (CI's one-ref
#             checkouts: a gate's `task publish -- --hub`, build.yaml's dry run, which lock as the promotion does). Read from
#             this repository, never this computer's cozy: a 0.27.0-dev cozy locked a branch of rc/0.28.0 onto rc-0.27.0's
#             builds, below the package's floor.
#   pure      the newest pure Runtime wheel (what a lock takes) on cozy-runtime's rolling prerelease rc-<x.y.z> that meets the
#             `cozy-runtime` requirement pyproject.toml names: uv takes a URL source without checking it against the floor.
#   native    the newest Linux x86_64 wheel, with kernels (what a machine runs).
#   tensorfs  the TensorFS wheel the RC's Runtime is locked with (cozy-creator's rc-<x.y.z>, Linux x86_64: the build its floor
#             names, xs-216).
# The prerelease keeps the RC's newest 20 builds; a package still names only its floor (`cozy-runtime>=0.47.0.dev0`, xs-194).
set -euo pipefail
git=(git -C "$(dirname "$0")")
open() {
  "${git[@]}" rev-parse -q --verify origin/master > /dev/null || return 0
  for b in $("${git[@]}" for-each-ref --format='%(refname:lstrip=3)' 'refs/remotes/origin/rc/[0-9]*' | sort -t/ -k2 -V -r); do
    at=$("${git[@]}" merge-base HEAD "origin/$b") || continue
    if ! "${git[@]}" merge-base --is-ancestor "$at" origin/master; then echo "${b#rc/}"; return 0; fi
  done
}
case ${1:-pure} in
  open) open; exit 0 ;;
  pure) pattern='^cozy_runtime-.*-py3-none-any[.]whl$' ;;
  native) pattern='^cozy_runtime-.*manylinux_2_28_x86_64[.]whl$' ;;
  tensorfs) ;;
  *) echo "usage: scripts/rc-runtime.sh open | pure [pyproject.toml] | native | tensorfs" >&2; exit 2 ;;
esac
rc=$(open)
[ -n "$rc" ] || { echo "rc-runtime.sh: $("${git[@]}" rev-parse --abbrev-ref HEAD) is on no open RC's branch (origin/rc/<x.y.z>; fetch it)" >&2; exit 1; }
if [ "$1" = tensorfs ]; then
  # The TensorFS build the RC's Runtime is locked and type-checked with (its [tool.uv.sources], xs-216): one build for the Runtime,
  # every package lock and the type checks, which rc-dev keeps while the Runtime's lock names it.
  pyproject=$(curl -fsSL "https://raw.githubusercontent.com/cozy-creator/cozy-runtime/rc/$rc/pyproject.toml")
  url=$(sed -n 's/^tensorfs = { url = "\([^"]*\)".*/\1/p' <<< "$pyproject")
  [ -n "$url" ] || { echo "cozy-runtime's rc/$rc names no TensorFS build" >&2; exit 1; }
  echo "$url"; exit 0
fi
base=https://github.com/cozy-creator/cozy-runtime/releases/download/rc-$rc
# SHA256SUMS.old while rc-dev swaps SHA256SUMS in by rename (cl-1128): it never deletes one under a reader. Read whole, then
# matched: an awk that exits at its match leaves curl writing into a closed pipe (exit 23 under pipefail: gate 44223772507).
sums=$(curl -fsSL "$base/SHA256SUMS" || curl -fsSL "$base/SHA256SUMS.old")
# The newest wheel whose version, less its local part, the requirement contains (as Calcifer's floor::rc_runtime reads it).
newest='import sys, tomllib
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version
want = SpecifierSet()
if len(sys.argv) > 1:
    for spec in tomllib.load(open(sys.argv[1], "rb"))["project"].get("dependencies", []):
        if (requirement := Requirement(spec)).name == "cozy-runtime":
            want = requirement.specifier
            break
version = lambda name: Version(name.split("-")[1])
met = [name for name in sys.stdin.read().split() if want.contains(version(name).public, prereleases=True)]
print(max(met, key=version) if met else "")'
file=$(awk -v p="$pattern" '$2 ~ p {print $2}' <<< "$sums" \
  | uvx -q uv@0.12.24 run -q --no-project --python 3.12 --with packaging==24.2 python -I -c "$newest" ${2:+"$2"})
[ -n "$file" ] || { echo "cozy-runtime's rc-$rc holds no $1 wheel${2:+ that meets the cozy-runtime requirement of $2}" >&2; exit 1; }
echo "$base/$file"
