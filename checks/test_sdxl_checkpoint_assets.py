"""Required SDXL tokenizer data must be supplied by its checkpoint."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdxl"))
from cozy_runtime.author import ModelAssets
from cozy_runtime.author._errors import InvalidRequest
import sdxl


class SdxlCheckpointAssetsTests(unittest.TestCase):
    def assets(self, data):
        return ModelAssets(data, lambda _: None)

    def test_checkpoint_supplies_the_tokenizer_without_loose_files(self):
        data = {
            "tokenizer/vocab.json": json.dumps({"<|startoftext|>": 0, "<|endoftext|>": 1, "h": 2, "i</w>": 3}).encode(),
            "tokenizer/merges.txt": b"#version: 0.2\n",
        }
        tokenizer = sdxl._tokenizer(self.assets(data), "tokenizer")
        self.assertEqual(tokenizer("hi").input_ids, [0, 2, 3, 1])

    def test_absent_checkpoint_tokenizer_is_not_replaced_by_package_data(self):
        with self.assertRaises(InvalidRequest) as failure:
            sdxl._tokenizer(self.assets({}), "tokenizer")
        self.assertEqual(failure.exception.code, "model_asset_missing")
        self.assertIn("tokenizer/vocab.json", str(failure.exception))

    def test_partial_checkpoint_tokenizer_names_the_missing_member(self):
        with self.assertRaises(InvalidRequest) as failure:
            sdxl._tokenizer(self.assets({"tokenizer/vocab.json": b"{}"}), "tokenizer")
        self.assertIn("tokenizer/merges.txt", str(failure.exception))


if __name__ == "__main__":
    unittest.main()
