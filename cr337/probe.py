# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.47", "msgspec>=0.19"]
# ///
"""cr-337, proof only, never merged: does a pinned host tensor freed inside a CUDA graph capture break later
device-to-host copies? Capture as varena's `vt.graph` does: `capture_begin` on a side stream, thread-local, no
collection before it. `arm`: `inside` collects the cyclic garbage holding the pinned tensor inside the capture,
`outside` before it, `after` only probes (a later call in the same process)."""

import gc

import msgspec
import torch

from cozy_runtime.author import App

app = App()


class Ask(msgspec.Struct, forbid_unknown_fields=True):
    arm: str = "inside"


class Said(msgspec.Struct):
    torch: str
    cuda: str
    arm: str
    steps: list[str]


def _probe(device: torch.device) -> str:
    try:
        return f"ok {int(torch.ones(1, dtype=torch.int64, device=device).sum())}"
    except Exception as exc:  # noqa: BLE001 - the result is the error
        return f"FAIL {type(exc).__name__}: {str(exc).splitlines()[0]}"


def _try(say: list[str], what: str, op: object) -> None:
    try:
        op()  # type: ignore[operator]
        say.append(f"{what}: ok")
    except Exception as exc:  # noqa: BLE001
        say.append(f"{what}: FAIL {type(exc).__name__}: {str(exc).splitlines()[0]}")


@app.entrypoint(accelerator=True)
def probe(payload: Ask) -> Said:
    device = torch.device("cuda", torch.cuda.current_device())
    say: list[str] = [f"start: {_probe(device)}"]
    if payload.arm != "after":
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        x = torch.zeros(8, device=device)
        with torch.cuda.stream(side):
            pinned = torch.empty(8, dtype=torch.int64, pin_memory=True)
            pinned.copy_(torch.arange(8, device=device), non_blocking=True)  # its use is recorded on `side`
        side.synchronize()
        cycle: list[object] = [pinned]
        cycle.append(cycle)  # cyclic garbage holding it, as the decode check's trace class did
        del pinned, cycle
        gc.disable()  # only the arm's own collection frees it
        graph = torch.cuda.CUDAGraph()
        try:
            if payload.arm == "outside":
                say.append(f"collected outside the capture: {gc.collect()} objects")
            with torch.cuda.stream(side):
                graph.capture_begin(capture_error_mode="thread_local")
                try:
                    x.add_(1)
                    if payload.arm == "inside":
                        say.append(f"collected inside the capture: {gc.collect()} objects")
                finally:
                    graph.capture_end()
            say.append("captured")
            _try(say, "replay", lambda: (graph.replay(), torch.cuda.synchronize(device)))
        except Exception as exc:  # noqa: BLE001
            say.append(f"capture FAIL {type(exc).__name__}: {str(exc).splitlines()[0]}")
        finally:
            gc.enable()
        say.append(f"after: {_probe(device)}")
        say.append(f"again: {_probe(device)}")
        _try(say, "synchronize", lambda: torch.cuda.synchronize(device))
        say.append(f"after synchronize: {_probe(device)}")
        torch.cuda.empty_cache()
        say.append(f"after empty_cache: {_probe(device)}")
        _try(say, "host empty_cache", torch._C._host_emptyCache)
        say.append(f"after host empty_cache: {_probe(device)}")
        del graph
        gc.collect()
        say.append(f"after the graph goes: {_probe(device)}")
        _try(say, "a fresh pinned tensor", lambda: torch.empty(8, pin_memory=True))
    return Said(torch=torch.__version__, cuda=str(torch.version.cuda), arm=payload.arm, steps=say)
