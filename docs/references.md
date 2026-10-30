# References and `<Name>` mentions

Every first-party package that takes reference media takes them in one shape. Its prompt names
them the same way:

```yaml
references:
  - {name: Person 1, image: ./persona.png}
  - {name: Pose,     image: ./pose.png}
prompt: "<Person 1> standing in the pose of <Pose>"
```

```sh
cozy run qwen-image-2/generate --input request.yaml
cozy run qwen-image-2/generate prompt='<Person 1> standing in the pose of <Pose>' \
  references:='[{"name": "Person 1", "image": "persona.png"}, {"name": "Pose", "image": "pose.png"}]'
```

## The rules

- **A reference** is `{name, <medium>, kind?}`.
  - `name` is required and unique.
  - The medium is one field named for it: `image`, `video` or `audio`.
  - `kind` is an optional free string. Only a package that reads it uses it (minimax-h3: `character`,
    `background`, …).
- **A mention** is `<Name>` anywhere in the prompt.
  - Names match in any case, with `_`, `-` and space treated alike, so `<person_1>` names `Person 1`.
  - A mention of no reference stays as written, with a warning.
  - Two references with one name are refused.
- **The package renders each mention** in its model's own words for the Nth reference: the wording
  the model was trained on (table below). A model with no such wording reads the name itself.
  The result's `prompt` is the text the model read.
- **Image packages** (the shared image request):
  - The first reference sets the default aspect ratio and pixel count.
  - `box` (`[x0, y0, x1, y1]`, 0–1) places a reference. Only models with layout control use it; the
    others warn.
  - A video or audio reference is refused: an image reference has no such field.
  - Text-only packages take no references.

| package | the Nth reference reads as | source |
|---|---|---|
| qwen-image | `Picture N` | Edit-2511's template writes `Picture N:` before each image |
| qwen-image-2 | `<imageN>`; one reference reads as its name | 2.1's template writes `<imageN>`, and Qwen's rewriter tags only two or more |
| fibo-2 | `<image_N>`; one reference reads as its name | the pipeline's `image_N` keys and Bria's instructions |
| flux-3-klein | "the first image", …; `ref_image_N-1` with boxes or regions; `<ref_image_N-1>` with `prompt_upsampling: false` | BFL's upsampler scaffold and Diffusers' Flux3 docs |
| flux-2 | `image N` | BFL's prompting guide |
| hunyuan-image-3 | `图N` | Tencent's documented prompts |
| hidream-o1 | its name | no convention upstream |
| anima, ideogram-4, krea-2, moss-image, sdxl | (takes no references) | text to image only |
| minimax-h3 | `<Picture N>`, `<Audio N>`, `<Video N>`, counted per medium and per segment | MiniMax's reference guide |

## For package authors

The convention lives in `cozy_runtime.author`, and it is opt-in. Nothing below the author library
knows it, so a package may declare any input shape it likes.

```python
from cozy_runtime.author import ImageReference, ImageRequest, Reference, VideoAsset, render

class GenerateInput(ImageRequest):  # references: list[ImageReference]; narrow the count with Meta(max_length=N)
    ...

def label(n: int, reference: ImageReference) -> str:
    return f"Picture {n}"  # how the model refers to the Nth reference

prompt = payload.rendered(adj, label)  # or render(text, references, label, adj=adj) for any list

class Clip(Reference):  # a new reference type: its medium and fields are its own
    kind: Literal["motion", "style"]
    video: VideoAsset
    strength: float = 1.0
```

Each package README states its label and how its first reference sets the size.
