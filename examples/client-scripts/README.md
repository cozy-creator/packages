# Client scripts

A script is an ordinary package in one file (xs-099). Its PEP 723 block is its `pyproject.toml`
(`dependencies`, `requires-python`, `[tool]`), and it declares `app = App()` and `@app.entrypoint`
functions as any package does: request and result structs, model slots, `defaults=`, `weights=`,
`accelerator=`. Run it with `cozy run ./path/to/script.py[/function] --on <machine>`; its one
function runs bare. It installs editable as `local/<file name>`: edit it and run it again.

Library operations look up compatible completed results and validate that their bytes still
exist, so a caller-only edit does not invalidate their keys. Changed operation inputs, parameters,
plan or relevant implementation produce a new key. The native conversion and quantization writers
can also adopt compatible partial work. A Runtime upgrade can conservatively invalidate more work
than a caller edit.

Each Store's `.cozy-workspace/journal.sqlite3` records operation identities, results, progress and
retention; TensorFS stores the artifact bytes. Reuse requires that Store and its retained bytes,
not a previous run ID or a Tensorhub publication. Inference runs fresh; checkpoint upload and
release creation are explicit effects.

Models bind with ordinary `model.<slot>=<ref>` terms; a function's `defaults=` sets its defaults.
