# Orbit

A picture's subject seen from all the way round: one clockwise camera turn around it, frozen in time, as a video that
loops. **Orbit** is a video tool: on a canvas it goes on a video node and makes that node's video. It takes its pictures
as qwen-image-2's Image does, by mention in its main text.

`orbit` takes:

- `instructions`: the pictures it mentions and what's said of them. The first picture mentioned (`<swordswoman>`) is the
  view the turn starts and ends on; every other one is a reference H3 reads by its mention (a face up close, a back
  view), up to H3's nine. Words beside the mentions go into the shot's description.
- `seed`: unset, a new one each render.

A turn has no length to choose: it is 15 s (`DURATION`), H3's longest shot, which turns smoothest (Measured).

It returns `video`: one turn, no sound, frame 0 at 0 degrees, its file marked an orbit. Frame `n` would be frame 0 again, so it isn't kept.

## How

MiniMax H3 renders the turn on Ref2VA with MATLOWAI's ORB360 LoRA (`matlowai/orb360@0.1.1`, a mirror of
[MiniMax-H3-ORB360-CardSpin](https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin)'s step 1500), Turbo, through
`minimax-h3/render_shot`. ORB360 turns only clockwise, and only from the caption it was trained on, which
`caption()` writes: the first picture by its mention, and the angle every third of the turn at its frame. Plain words make a
half turn, and without the LoRA the camera makes a quarter turn. A picture whose edges are one even color (a character
drawn on white) gets the caption's plain-backdrop wording; a real scene's wording grows snow and ground there.

The turn's end has to meet its start (`ENDS`):

- `pinned` (default): the picture is also the shot's first and last frame (Ref2VA keyframes), so ORB360 stays on the
  DiT it was trained on and the last frame is the first again. It follows the picture's shape (a square picture makes
  a 768x768 turn).
- `bridged` (se-114's B): a Ref2VA turn at 1344x768, cut where its tail comes closest to frame 0, its last twelfth (from
  about 330 degrees) redrawn by a 5 s FL2VA shot without the LoRA, from there to frame 0, fitted into those frames.
  Two calls, and the last 1.25 s is a little busier.

Its video says it is an orbit (`orbit: {degrees: 360, direction: cw, loops: true}`, `TURN`), so the canvas turns it on
drag instead of playing it, and every frame is a keyframe, so any angle shows at once.

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

Orbit calls H3's `render_shot` through `orbit/_h3.py`, typed references generated from H3's interface. After
changing H3's interface, regenerate them from the packages root:
`cozy-runtime --dir minimax-h3 --json describe > h3.json && cozy-runtime --package-interface h3.json calls minimax-h3 orbit/orbit/_h3.py`.
