# First-party package releases

Nothing untested goes to tensorhub.com, not even as a prerelease (Paul, 11-13). On the open RC a changed package carries
its next prerelease version (`0.13.0rc2`) and is tested from source: release gates and smoke sessions run
`cozy run ./<package>/<function> --on <machine>` from a packages commit, after `scripts/triton.sh <package>` built its
Triton sets into the checkout, so the source Calcifer uploads carries the kernels a release ships.

Its final is published once, at the promotion, after every gate passed and cozy is Latest: cozy-creator's
`release-gate/promote.sh` runs, on a checkout of master once the RC is merged into it,

```sh
task publish -- --gated <the gate's packages_ref>
```

It publishes every package whose final no Hub holds, each refused unless master's source of it is the gated source. Only
the version changes: `0.13.0rc2` becomes `0.13.0`, relocked as its publish will (`task lock`, upgrading only the roots it
takes from the Hub), committed to master and published live to each of `HUBS`, with its Triton sets compiled on this CPU
for every card row (`scripts/triton.sh`, by the Runtime its lock pins) in `_triton` beside each module that declares one.
Then, in dependency order, each package whose pyproject takes a published package from `tensorhub` is relocked to it,
patch-bumped, pushed to master and published the same way. A version any Hub holds is refused before anything is
published; a dependent takes the next patch no Hub holds.

- `task publish -- --dry-run <package>...` does all of it but the push and the Hub publish, builds each wheel into `dist/`
  and leaves the checkout as it was (CI: `build.yaml`'s `publish-dry-run` scope).
- `task publish -- --hub <url> <package>...` publishes the finals, and the in-repo packages they take, to a throwaway Hub
  only, locked against its index, pushing nothing: CI's isolated Hub (cozy-creator release-gate suite `publish`).

## Early access (`--rc`, cl-922)

Only when Paul asks for early access, and only after the RC passed its full release gate (a prerelease is tested as
fully as a release):

```sh
task publish -- --rc sdxl             # on a checkout of rc/<x.y.z>; sdxl's version is a prerelease (0.10.0rc1)
```

It publishes the prerelease live as it is. Released clients and each Hub's unversioned default skip prereleases. Its
floor is the one its source declares; a package that needs what only the RC's Runtime has declares that build and locks
it by URL (`cozy-runtime>=0.47.0.dev2260,<0.48`, `[tool.uv.sources] cozy-runtime = { url = "…/rc-<x.y.z>/….whl" }`). Its
Triton sets compile with the RC's Runtime. A lock the relock changes goes into the RC branch as a pull request, queued.

Complete the [dependency qualification checklist](DEPENDENCIES.md) first. Worker images
are OS/CUDA/PyTorch bases named by tag; the CLI updates the machine, Runtime and TensorFS
at runtime, so a package never waits on an image.

## One source, every Hub

First-party packages publish at the root namespace, with no owner (`sdxl`, `minimax-h3`): `task publish`
runs `cozy package publish --root` on each Hub, which only Hub admins may do. Model defaults name upstream
models in their authors' orgs (`minimaxai/minimax-h3@1.0.0-rc.4`); a default with no org names a model we make. First-party package dependencies name the
`tensorhub` index (`qwen-image-2 = { index = "tensorhub" }`), which is the root's (`/v1/index/_/simple/`).

Commit locks bound to production. After changing dependencies:

```sh
task lock -- qwen-image-2
```

A published release runs exactly the dependency versions in its `uv.lock`, including
first-party callees such as qwen-image-2, so each callee release needs a dependent
release; `task publish` makes it. The exception is `cozy-runtime` and `tensorfs`: every
package runs on its machine's Runtime, so the package only declares the minimum its
`cozy-runtime>=` bound admits.

Each publication rebinds the committed lock to its target: `cozy package publish --root` to the
local Hub relocks a private copy against the root's index there and refuses
(`account_index_lock_drift`) unless every locked version exists unchanged, which is
why `task publish` releases dependencies before dependents on each Hub.

Prefer lower bounds for package dependencies, such as `pydantic-core>=2.46.4`.
If a compatibility ceiling is needed, bound the major version (`>=2.46.4,<3`) or
at most the minor version (`>=2.46.4,<2.47`). Major/minor wildcards and equivalent
compatible-release ranges are allowed. Exact releases (`==2.46.4`, including
short forms such as `==2.46`) and patch ceilings are rejected. Keep the exact
tested versions in `uv.lock`; compatibility declarations must admit patch updates.

When a package starts using a new Runtime API:

1. Release and verify Runtime, including its public wheel artifacts.
2. Set the package's minimum Runtime version to the oldest version that supplies
   the APIs it actually uses, and relock.
3. Gate it from source (above); it publishes at the promotion.

Private development or custom-image packages can still use `cozy package publish`
directly. Their requirements need not fit every first-party worker image. The
worker capability check remains authoritative for new worker APIs, and an existing
private rental may need `cozy machine update <name>` before using such an API.
Ordinary dependency updates belong to the package environment and do not require
updating the worker's global Python environment.

Publish the same frozen H3 source to each Hub; each Hub's root index must resolve the
committed versions.
