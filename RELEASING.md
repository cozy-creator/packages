# First-party package releases

Publish a merged package, and everything that locks it, to both Hubs:

```sh
task publish -- qwen-image-2
```

It fast-forwards the checkout to `origin/master` and publishes a `git archive` of the package to each of `HUBS`.
Then, in dependency order, each package whose `uv.lock` holds a republished package is
relocked to it (`cozy package lock --upgrade-package <it>` against production, plus
`cozy-runtime` or `tensorfs` when its floor is above the dependent's lock),
patch-bumped, pushed to master and published the same way. A dependent whose
master version is unpublished stops it too: its merged changes are not released yet.
Bump the package's version on master first: a version any Hub already holds is refused before
anything is published, and nothing is ever yanked but what that run just published. A dependent
takes the next patch version no Hub holds. After a failed run, bump and run again.

Complete the [dependency qualification checklist](DEPENDENCIES.md) first. Worker images
are OS/CUDA/PyTorch bases named by tag; the CLI updates the machine, Runtime and TensorFS
at runtime, so a package never waits on an image.

## One source, every Hub

First-party packages publish at the root namespace, with no owner (`sdxl`, `minimax-h3`): `task publish`
runs `cozy package publish --root` on each Hub, which only Hub admins may do. Model defaults name upstream
models in their authors' orgs (`minimaxai/minimax-h3@1.0.0-rc.4`); a default with no org names a model we make. First-party package dependencies name the
`tensorhub` index (`qwen-image-2 = { index = "tensorhub" }`), which is the root's (`/v1/index/_/simple/`).

Commit locks bound to production. After changing dependencies, as a tensorhub.com admin:

```sh
cozy package lock --root
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
3. Publish with `task publish`.
4. Install the published release with the local Creator CLI and verify its input
   and callable interface. Include execution qualification appropriate to the change.

Private development or custom-image packages can still use `cozy package publish`
directly. Their requirements need not fit every first-party worker image. The
worker capability check remains authoritative for new worker APIs, and an existing
private rental may need `cozy machine update <name>` before using such an API.
Ordinary dependency updates belong to the package environment and do not require
updating the worker's global Python environment.

Publish the same frozen H3 source to each Hub; each Hub's root index must resolve the
committed versions.
