# SDXL tokenizer assets

SDXL reads both CLIP tokenizers from the selected CozyTensors checkpoint. Each is a
vocabulary and merges pair, and all four are required:

```text
tokenizer/vocab.json
tokenizer/merges.txt
tokenizer_2/vocab.json
tokenizer_2/merges.txt
```

A missing member raises Runtime's `model_asset_missing` error naming it; the package
carries no substitute. Invalid bytes and other Runtime asset errors also propagate.

The first CLIP tokenizer pads with `<|endoftext|>`; the second pads with `!`.
Both use replacement errors and a 77-token maximum. Unused tokenizer settings
sidecars do not change those model-family constants.

Loading reads verified model bytes without network access or filesystem writes,
including inside Runtime's admission sandbox. The constructed tokenizers stay
attached to the loaded model and are reused by warmup and every request.
