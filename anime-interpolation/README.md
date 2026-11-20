# Anime Frame Interpolation

A `video` → `video` tool that uses Practical-RIFE 4.26 to fill short holds in animation. **Keep source frame rate** fills the repeated slots while preserving the original drawing times. **Double source frame rate** also adds half-frame slots, for example from 24 fps to 48 fps.

At detected hard cuts, the tool keeps the last drawing on screen until the first frame of the new shot. Holds longer than about ⅛ second stay held. The final frame occupies its full duration at either rate. The tool streams decoded media and passes the existing soundtrack to Runtime for encoding without changing its speed. It accepts constant-rate video because Runtime writes a constant-rate MP4 clock.

The hold-aware workflow was informed by [TheAnimeScripter](https://github.com/NevermindNilas/TheAnimeScripter). The timing code is Cozy's own. The network is adapted from [Practical-RIFE](https://github.com/hzwer/Practical-RIFE#trained-model), whose author distributes the inference code and weights under the MIT license included in this package.

For package maintainers, download the [pinned upstream release archive](https://drive.google.com/file/d/1gViYvvQrtETBgU1w8axZSsr7YUuw31uy/view) and run `<author-environment>/bin/python anime-interpolation/prepare_checkpoint.py RIFEv4.26_0921.zip rife.cozytensors` in an author environment with PyTorch, Cozy Runtime, and TensorFS. This verifies the archive hash before reading only its tensor state and writes the native `rife` component with the license. Workers only load the resulting TensorFS checkpoint.
