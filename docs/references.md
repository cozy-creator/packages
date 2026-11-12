# References and `<name>` mentions

A prompt is its text, as typed, plus the references that text mentions (`cozy_runtime.author.Prompt`):

```yaml
prompt: "<amy> standing in the pose of <pose>"
```

```sh
cozy run qwen-image-2/generate --input request.yaml     # in a project: each <name> is that box
```

The function receives `Prompt(text, references)`: `references` maps each key, spelled exactly as its `<key>` in the
text, to what it names, `Image(file, description)` (or `Video`, `Audio`), in order of first mention. Calcifer fills it
and checks every mention names something, so a package only replaces `<key>` with its model's words.

- `<amy>` is the box's approved picture, else its newest; `<amy.2>` is version 2; `<walk@48-120>` frames 48 to 120 of
  a video, cut before sending.
- A text box is put in the text as its words; it is not a reference.
- A picture model takes `Prompt[Image]`: a mentioned video stays as its name in the text, with a warning.
- Each package words a reference in its model's own words for the Nth one (below). A dotted key a model reads by name
  reads `kyogre.2` as `kyogre 2`. The result's `prompt` is the text the model read.

| package | the Nth picture reads as | source |
|---|---|---|
| qwen-image | `Picture N` | Edit-2511's template writes `Picture N:` before each image |
| qwen-image-2 | `<imageN>`; one picture reads as its name | 2.1's template writes `<imageN>`, and Qwen's rewriter tags only two or more |
| fibo-2 | `<image_N>`; one picture reads as its name | the pipeline's `image_N` keys and Bria's instructions |
| flux.3-klein | "the first image", …; `ref_image_N-1` with boxes or regions; `<ref_image_N-1>` with `prompt_upsampling: false` | BFL's upsampler scaffold and Diffusers' Flux3 docs |
| flux-2 | `image N` | BFL's prompting guide |
| hunyuan-image-3 | `图N` | Tencent's documented prompts |
| hidream-o1 | its name | no convention upstream |
| minimax-h3 | `<Picture N>`, `<Video N>`, `<Audio N>` per segment; see its README | MiniMax's reference guide |
| moss-image | its name; several as `image N` | no convention upstream: Moss-Image-Edit's instructions say "the image" |
| anima, ideogram-4, krea-2, sdxl | (takes no pictures) | text to image only |

```python
from cozy_runtime.author import Image, ImageRequest, Prompt

class GenerateInput(ImageRequest):
    prompt: Prompt[Image] = Prompt()

text = payload.prompt.text
for n, key in enumerate(payload.prompt.references, 1):
    text = text.replace(f"<{key}>", f"Picture {n}")   # how the model refers to the Nth picture
```
