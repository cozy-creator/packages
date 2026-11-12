# First-party package releases

Publish a merged package, and everything that locks it, to both Hubs:

```sh
task publish -- qwen-image-2
```

It fast-forwards the checkout to `origin/master`, relocks each named package as its publish will (`task lock`, no
upgrade, so master holds the lock it publishes) and publishes a `git archive` of it to each of `HUBS`, with the
Triton sets its import declares compiled on this CPU for every card row into `_triton` beside the module that declares
each, by the Runtime its lock pins, so a fresh machine's first request runs its fused plans without compiling.
Then, in dependency order, each package whose pyproject takes a republished package from `tensorhub` is
relocked to it (`task lock -- <dependent> --upgrade-package=<it>`, plus
`cozy-runtime` or `tensorfs` when its floor is above the dependent's lock),
patch-bumped, pushed to master and published the same way. A dependent whose
master version is unpublished stops it too: its merged changes are not released yet.
`task publish -- --dry-run qwen-image-2` does all of it but the push and the Hub publish, builds each wheel into
`dist/` and leaves the checkout as it was; CI runs it as `build.yaml`'s `publish-dry-run` scope.
Bump the package's version on master first: a version any Hub already holds is refused before
anything is published, and nothing is ever yanked but what that run just published. A dependent
takes the next patch version no Hub holds. After a failed run, bump and run again.

## Built for the open RC's Runtime (cl-922)

A package that calls what only the open RC's Runtime has (`author.preview` before Runtime 0.46) is published from the
RC branch as a PEP 440 prerelease, live:

```sh
uv version --project sdxl 0.10.0rc1   # the next rcN, committed to rc/<x.y.z>
task publish -- --rc sdxl             # on a checkout of rc/<x.y.z>
```

Released clients and each Hub's unversioned default skip prereleases; a dev home's rental runs the RC's Runtime and takes
the newest one. Its Triton sets compile with the RC's Runtime build. At the release the package publishes its final
version (`0.10.0`) the usual way.

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
