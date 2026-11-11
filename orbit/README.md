# Orbit

A picture's subject seen from all the way round: one clockwise camera turn around it, frozen in time, as a video that
loops. On a canvas, put **Orbit** on a picture box; the picture goes in its `picture` slot.

`orbit` takes:

- `picture`: the view the turn starts and ends on.
- `subject`: what it shows, in a word or two (`motorcycle`); the caption names it.
- `duration`: one turn's length, 5 to 15 s (15 s by default; the clip is H3's `17n + 5` frames for those seconds).
- `seed`: unset, a new one each render.

It returns `video`: one turn, no sound, frame 0 at 0 degrees. Frame `n` would be frame 0 again, so it isn't kept.

## How

MiniMax H3 renders the turn on Ref2VA with MATLOWAI's ORB360 LoRA (`matlowai/orb360@0.1.0`, a mirror of
[MiniMax-H3-ORB360-CardSpin](https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin)'s step 1500), Turbo, through
`minimax-h3/render_shot`. ORB360 turns only clockwise, and only from the caption it was trained on, which
`caption()` writes: the subject, its picture, and the angle every third of the turn at its frame. Plain words make a
half turn, and without the LoRA the camera makes a quarter turn. A picture whose edges are one even color (a character
drawn on white) gets the caption's plain-backdrop wording; a real scene's wording grows snow and ground there.

The turn's end has to meet its start (`ENDS`):

- `pinned` (default): the picture is also the shot's first and last frame (Ref2VA keyframes), so ORB360 stays on the
  DiT it was trained on and the last frame is the first again. It follows the picture's shape (a square picture makes
  a 768x768 turn).
- `bridged` (se-114's B): a Ref2VA turn at 1344x768, cut where its tail comes closest to frame 0, its last twelfth (from
  about 330 degrees) redrawn by a 5 s FL2VA shot without the LoRA, from there to frame 0, fitted into those frames.
  Two calls, and the last 1.25 s is a little busier.

Every frame a keyframe, and the `orbit` metadata (`TURN`) that makes the canvas turn it instead of playing it, wait on
the Runtime's output metadata (xs-180).

## Measured

2026-11-11, one RTX 5090, Turbo, 15 s, seed 20260926 (se-116). Seam: the dropped last frame against frame 0, PSNR on
96x54 grey; wrap: the step the loop shows (frame n-1 to 0) against the turn's ordinary steps.

| Picture | Ends | Size | Full turn | Seam | Wrap / step | Time |
|---|---|---|---|---|---|---|
| motorcycle (drawn, alley) | pinned | 768x768 | yes | 36.5 dB | 26.1 / 29.7 dB | 147 s |
| swordswoman (drawn, white) | pinned | 768x768 | yes | 38.8 dB | 35.8 / 35.2 dB | 122 s |
| white cat (photo, snow) | pinned | 1024x768 | yes | 40.1 dB | 27.9 / 26.7 dB | 211 s |
| motorcycle | bridged | 1344x768 | yes | 34.1 dB | 28.0 / 32.4 dB | 384 s |
| swordswoman | bridged | 1344x768 | yes, on a gold backdrop it made up | 35.2 dB | 25.3 / 29.8 dB | 369 s |

Before, the demo's Ref2VA turns cut at their closest tail frame: 18.8-20.2 dB.
