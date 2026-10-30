# Assets and `<Name>` mentions

An asset is a production element, in VFX terms: a character, a prop or an environment. Every
first-party package that takes assets takes them in one shape. Its prompt names them the same way:

```yaml
assets:
  - {name: Person 1, image: ./persona.png}
  - {name: Pose,     image: ./pose.png}
prompt: "<Person 1> standing in the pose of <Pose>"
```

```sh
cozy run qwen-image-2/generate --input request.yaml
cozy run qwen-image-2/generate prompt='<Person 1> standing in the pose of <Pose>' \
  assets:='[{"name": "Person 1", "image": "persona.png"}, {"name": "Pose", "image": "pose.png"}]'
```

A file is the attached file itself (`image`, `video`, `audio`, `--file`).

## The rules

- **An asset** is `{name, <medium>, kind?}`.
  - `name` is required and unique.
  - The medium is one file field named for it: `image`, `video` or `audio`.
  - `kind` is an optional free string, such as `character`, `prop` or `environment`. Only a
    package that reads it uses it.
- **A mention** is `<Name>` anywhere in the prompt.
  - Names match in any case, with `_`, `-` and space treated alike, so `<person_1>` names `Person 1`.
  - A mention of no asset stays as written, with a warning.
  - Two assets with one name are refused.
- **The package renders each mention** in its model's own words for the Nth asset: the wording
  the model was trained on (table below). A model with no such wording reads the name itself.
  The result's `prompt` is the text the model read.
- **Image packages** (the shared image request):
  - The first asset sets the default aspect ratio and pixel count.
  - `box` (`[x0, y0, x1, y1]`, 0–1) places an asset. Only models with layout control use it; the
    others warn.
  - A video or audio asset is refused: a picture asset has no such field.
  - Text-only packages take no assets.

| package | the Nth asset reads as | source |
|---|---|---|
| qwen-image | `Picture N` | Edit-2511's template writes `Picture N:` before each image |
| qwen-image-2 | `<imageN>`; one asset reads as its name | 2.1's template writes `<imageN>`, and Qwen's rewriter tags only two or more |
| fibo-2 | `<image_N>`; one asset reads as its name | the pipeline's `image_N` keys and Bria's instructions |
| flux-3-klein | "the first image", …; `ref_image_N-1` with boxes or regions; `<ref_image_N-1>` with `prompt_upsampling: false` | BFL's upsampler scaffold and Diffusers' Flux3 docs |
| flux-2 | `image N` | BFL's prompting guide |
| hunyuan-image-3 | `图N` | Tencent's documented prompts |
| hidream-o1 | its name | no convention upstream |
| anima, ideogram-4, krea-2, moss-image, sdxl | (takes no assets) | text to image only |

## For package authors

The convention lives in `cozy_runtime.author`, and it is opt-in. Nothing below the author library
knows it, so a package may declare any input shape it likes.

```python
from cozy_runtime.author import Asset, ImageRequest, PictureAsset, VideoFile, render

class GenerateInput(ImageRequest):  # assets: list[PictureAsset]; narrow the count with Meta(max_length=N)
    ...

def label(n: int, asset: PictureAsset) -> str:
    return f"Picture {n}"  # how the model refers to the Nth asset

prompt = payload.rendered(adj, label)  # or render(text, assets, label, adj=adj) for any list

class Clip(Asset):  # a new asset type: its media and fields are its own
    kind: Literal["motion", "style"]
    video: VideoFile
    strength: float = 1.0
```

Each package README states its label and how its first asset sets the size.
