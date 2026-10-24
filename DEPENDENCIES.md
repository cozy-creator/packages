# Package dependencies and development qualification

## Required behavior

Each package executes in its own resolved Python environment. Local execution and
private workers use the same Runtime installation and execution paths. Installing
or updating package A must not change package B, the worker control process, or
the base image. An already prepared environment is reused for subsequent requests;
dependency installation is preparation work, never part of each inference call.

Dependencies baked into a worker image are a startup optimization, not a whitelist
or a global set of version constraints. For example, if Diffusers needs
`huggingface-hub>=1.23.0` but the image contains `1.16.1`, install a compatible
version in that package's environment. Resolve the whole dependency chain, not
only the named mismatch. Do not repair this by mutating the base environment,
adding another version exception, or ignoring dependency metadata with `--no-deps`.

Use a machine-local uv cache to share identical dependency files across package
environments. Reflinks or hardlinks avoid duplicating bytes when the filesystem
supports them; different versions remain independently installable. Shared cache
files and active environments must not be modified in place. A uv cache does not
mean all packages share one Python environment.

Prefer a compatible baked PyTorch/native stack. If a package requires a different
installable version, prepare that version privately and report **degraded mode**:
the extra download, disk space and cold-start work require operator attention.
This warning must not abort an otherwise valid installation. Native extensions
must match the selected PyTorch/CUDA/Python stack; never import incompatible baked
extensions simply because their files are present. Unsatisfiable requirements,
unsupported wheel/driver combinations and exhausted resources remain real errors
and must identify the actual cause.

The worker's control protocol and package SDK compatibility are distinct from
ordinary application dependencies. Do not turn a Hugging Face library mismatch
into a mandatory worker update. Changes that require new worker capabilities need
their own release and compatibility checks.

## Authoring and updating

Declare every direct dependency in `pyproject.toml`, including appropriate extras
and platform markers. Use the oldest supported version floor and only justified
major/minor ceilings. The authored-dependency policy in [RELEASING.md](RELEASING.md)
rejects exact release pins; captured resolutions and third-party wheel metadata
may contain exact versions. Do not remove those identities from lockfiles.

Refresh the complete resolution with uv when changing dependencies. Check the
installed closure, not only direct requirements or a top-level import. Record
the exact package release, Runtime version, Python/native stack, selected wheel
versions and hashes, and the image digest it ran on. Mutable `latest`
tags and the package author's laptop are insufficient release evidence.

## Required regression evidence

Dependency/environment changes must demonstrate these behaviors before being
declared delivered. Keep executable regression checks with the environment owner
in Runtime and end-user qualification with Creator; do not duplicate a resolver
inside this package repository.

- Install a package needing a newer transitive dependency than the base provides.
  Include the reported Diffusers/Hugging Face mismatch. Import and execute using
  the private version, while verifying the base's version and files are unchanged.
- Install a second package needing a different version, then execute A, B, A.
  Verify each import's version and origin, including native dependencies, and
  verify undeclared base packages cannot leak into either environment.
- Repeat preparation and invocation without dependency resolution or downloads.
  Verify shared dependency bytes are reused where supported. Interrupt an install
  and show that retry works without damaging an existing prepared environment.
- Exercise a compatible alternate large dependency and observe degraded-mode
  reporting; exercise a truly unsupported native stack and observe an accurate
  failure. Do not call a CPU-only import test proof of GPU execution.
- Run the same qualification through the ordinary local `cozy` CLI and private
  worker path. Record request numbers and results in the normal client history.
  Direct Runtime tests supplement this evidence; they do not replace it.

Use `cozy package install <package>` and `cozy run <package>/<function>`
for the public package path; use `cozy run ./script.py --on <name>` for a private
script. Preserve user inputs, existing packages and active rentals during testing.

These are acceptance requirements, not a claim that isolation already passes.
The 2026-09-15 audit found rental-specific base-version refusal and an incomplete
dependency repair path. Keep that remediation open until the stale-base and
two-package tests pass on the installed CLI and a real worker environment.
