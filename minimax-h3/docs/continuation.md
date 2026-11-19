# H3 segment continuation

A segment carries a **picture window** (22, 39 or 56 frames) and/or **one second of sound**
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

The predecessor's picture therefore ends at its last sampled frame. Each segment samples
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

Each segment delivers sound from where its predecessor's ended to the end of its own (when
its successor carries sound) or to its picture end. Concatenated, the soundtrack is
continuous and the clocks telescope: never more than a frame from the picture at a join, in
sync at the end. A successor carrying sound alone needs the sound to reach the join, so the
predecessor's picture ends one frame early when its grid rounds the sound down (362 frames →
603 ticks → 361 frames).

## Resuming mid-video

A pickup's `out_frame` cuts a whole video's latents at one of its seams (`shot_latents.cut`, on the `17k + 5` grid): the same tokens, in
phase, whichever segment made them, so the last 22 frames there are the carry and the new segment joins the kept latents and
decodes with them as one. The file keeps no audio latents before a video's end, so the carried second is encoded from its
soundtrack by the audio VAE, ending on the tick nearest the picture join, where the kept sound stops. That is the one
decoded-and-encoded link in a resume, and only sound: its one job is to condition the new sound. As at any join, the new
segment's head is regenerated beside the carried frames and cut, so no new frame is lost. The kept latents don't change, and the
video decodes again as one, so only the last few frames before the join differ, where the VAE's overlap reads the new latents.

## Any other video

A video without H3 latents carries its last 22/39/56 frames encoded (`h3.HeldFrames`, at the shot's canvas, on the
24-fps clock) and its last second of sound encoded from its soundtrack at 32 kHz, as a cut's sound is. The encoder runs
each 17-frame chunk alone, so the window is the same kind of tokens as a carry, in phase. The new part's first segment
saves its latents led by that window (`lead` in their metadata), and its last decodes the part from it, so the part
decodes in phase without the regenerated head. The new part is spliced into the video (`shot.join_kept`, `Outputs.splice_video`): its last 4 frames
cross-fade into the part's decode of the carried window, and its last 40 ms of sound, equal power, into the regenerated
head's sound. The kept sound is resampled to the new part's. The result has no latents.

## Redo a range

`retake` cuts a take at a window and makes only the middle (`minimax_h3/retake.py`). The window goes on from the 22
frames before it, from the take's latents when they're on its `17k + 5` grid (`shot_latents.cut`), else encoded. At its
end, a shot change within one chunk of the asked end (picture and colour change both outliers among the frames around
it) ends it: the last segment samples a few frames past it, cut. Otherwise the take's next 39 frames and their second of
sound are a suffix, held over the sample's last 12 latents as the carry is over its head
(`ContinuationPlan.suffix`, `align_context_rows`, `unprotect_carried`); the sample is exactly `22 + L + 39`, so the
window is `L ≡ 12 (mod 17)` frames, and with latents it ends on a chunk, where the take's own latents follow in phase.
The sound starts on the nearest tick to the suffix's first frame, encoded from 5 s before to 1 s after its second. A
window longer than one sample chains segments of whole chunks, the suffix only on the last. The part decodes from the
carried window to the suffix's in place of its regenerated head and tail, and joins by pixels at both ends as above
(no picture cross-fade at a shot change).

Outside the window every packet is the take's, byte for byte; the window and the frames up to the take's next IDR are encoded
again at its size and clock. A take that can't be cut by packet copy is encoded again whole (`shot._joined`).

## API

`minimax_h3/continuation.py`: `Carry`, `ContinuationPlan(frames, carried, successor, suffix, tail)`,
`AVContext(video, audio, audio_offset, provenance)` with `encode_context`/`decode_context`
(schema 3, ≤ 16 MiB, ~2.7 MB for 22 frames at 1344×768), `export_context`,
`align_context_rows`. `OfficialH3Pipeline.start_ref2va` takes `context`, `delivery` and
`expected_context_provenance`; `export_context(state, provenance=...)` returns the
successor's carry after denoise, before decode. Contexts always come from the first,
native-canvas pass; an upscale of the delivered video never feeds a successor.

A carried window's picture rows aren't references: they take the video's own sparsity, dense for the dense steps and routed after, while text, references and audio stay protected (Paul, 11-12; `reference_rows.unprotect_carried` moves them out of the protected prefix, to just before the target rows).
