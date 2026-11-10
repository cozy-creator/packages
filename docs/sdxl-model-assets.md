# SDXL tokenizer assets

SDXL loads its two CLIP tokenizers with the pipeline, `StableDiffusionXLPipeline.from_pretrained(loader.path)`,
from the checkpoint's `tokenizer/` and `tokenizer_2/` folders.

A checkpoint that ships neither folder for a tokenizer gets the package's standard CLIP vocabulary and merges,
`sdxl/clip/vocab.json` and `sdxl/clip/merges.txt` (one copy: both encoders share those bytes, restored unchanged
from the package's original tokenizer data at `d9ef626^`). The first pads with `<|endoftext|>`, the second with
`!`; both use replacement errors and a 77-token maximum. A folder missing one of its files fails the load, as
Transformers' own loader does: the fallback never mixes a checkpoint's vocabulary with the package's merges.
