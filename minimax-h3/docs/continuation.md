# H3 piece continuation

A piece carries a **picture window** (22, 39 or 56 frames) and/or **one second of sound**
from its predecessor, chosen independently. Both are sliced from the predecessor's denoised
latents: there is no decode/re-encode round trip, whose ~2.4% darkening per link compounded
into colour drift and softening down a chain.

## Picture

H3's video VAE packs frames 1/4/4/4/4 per five latents, so 22/39/56 frames are whole cycles
(7/12/17 latents). A sample is `17g + 5` frames (`5g + 2` latents), so the last window always
starts on a cycle boundary: the predecessor's own tokens, in phase. They are appended as the
last condition rows (no Qwen slot, no public label) and their rotary times moved onto the
target head they condition. The head is regenerated beside them and cut: the owner's double
form (2026-10-10), never a single frozen copy (tag `archive/h3-single-copy-context-20261010`).
The native scheduler never writes condition rows, so the carry is hard-locked.

The predecessor's picture therefore ends at its last sampled frame. Each piece samples
its carried window plus its seconds snapped up to the grid and delivers everything after the
head: 15 s first = 362 frames, then 14 s + 22 carried = 362 sampled, 340 delivered.

## Sound

The last 40 audio ticks (24 frames, exactly one second at 40 Hz, whatever the picture
window), ending on the first whole tick at or after the predecessor's picture end (its
last sampled tick at most: H3's `round(5f/3)` grid can fall a third short). They are placed
on the new clip's timeline to end exactly where they ended on the predecessor's: at the
picture join plus that offset, within a third of a tick.
Rotary positions are float, so the sound is in true time. It always overlaps a regenerated,
cut head: the picture window's, or for sound alone a one-second head. Placed immediately
before the target start instead, H3 read it as a reference to imitate and did not continue it
(run 5485, join 4: 20 dB quieter, NCC at cover-band level).

Each piece delivers sound from where its predecessor's ended to the end of its own (when
its successor carries sound) or to its picture end. Concatenated, the soundtrack is
continuous and the clocks telescope: never more than a frame from the picture at a join, in
sync at the end. A successor carrying sound alone needs the sound to reach the join, so the
predecessor's picture ends one frame early when its grid rounds the sound down (362 frames →
603 ticks → 361 frames).

## API

`minimax_h3/continuation.py`: `Carry`, `ContinuationPlan(seconds, carried, successor)`,
`AVContext(video, audio, audio_offset, provenance)` with `encode_context`/`decode_context`
(schema 3, ≤ 16 MiB, ~2.7 MB for 22 frames at 1344×768), `export_context`,
`align_context_rows`. `OfficialH3Pipeline.start_ref2va` takes `context`, `delivery` and
`expected_context_provenance`; `export_context(state, provenance=...)` returns the
successor's carry after denoise, before decode. Contexts always come from the first,
native-canvas pass; an upscale of the delivered video never feeds a successor.
