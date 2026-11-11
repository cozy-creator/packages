"""Stable consumers read native checkpoint data, including in a fresh owned store."""

from collections.abc import Iterator
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import tensorfs
from cozy_runtime.author import InvalidRequest, ModelAssets
from tokenizers import Tokenizer, models, pre_tokenizers

from anima.model import _tokenizer as anima_tokenizer
from sdxl import _tokenizer as sdxl_tokenizer


@contextmanager
def checkpoint_assets(components: dict[str, dict[str, bytes]]) -> Iterator[ModelAssets]:
    # The published 0.23 SDK has a thin file. Copy its header plus explicitly owned
    # asset bytes into a fresh metadata view; do not imply 0.24 portable-file support.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        header = root / "source.cozytensors"
        writer = tensorfs.Writer(header, cache_dir=source)
        for name, files in components.items():
            component = writer.component(name, config={"fixture": True})
            for filename, data in files.items():
                component.asset(filename, data)
        saved = writer.commit()
        copied = root / "copied.cozytensors"
        shutil.copyfile(header, copied)
        payloads = [component.asset(path) for component in saved.components.values() for path in component.assets]
        saved.close()
        del saved, writer, component
        gc.collect()
        shutil.rmtree(source)
        checkpoint = tensorfs.view(copied.read_bytes(), own=payloads, cache_dir=root / "fresh", local_files_only=True)
        try:
            assert checkpoint.missing() == []
            data = {
                f"{name}/{path}": component.asset(path)
                for name, component in checkpoint.components.items()
                for path in component.assets
            }
            yield ModelAssets(data, lambda _: None)
        finally:
            checkpoint.close()


def tiny_tokenizer() -> dict[str, bytes]:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "checkpoint": 1, "only": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return {
        "tokenizer_config.json": json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"}).encode(),
        "tokenizer.json": tokenizer.to_str().encode(),
    }


class AnimaCheckpointDataTests(unittest.TestCase):
    def test_both_tokenizers_use_the_selected_checkpoint(self) -> None:
        with checkpoint_assets({"tokenizer": tiny_tokenizer(), "t5_tokenizer": tiny_tokenizer()}) as assets:
            for folder in ("tokenizer", "t5_tokenizer"):
                self.assertEqual(anima_tokenizer(assets, folder).encode("checkpoint only"), [1, 2])

    def test_checkpoint_template_is_used(self) -> None:
        files = tiny_tokenizer()
        files["chat_template.jinja"] = b"checkpoint {{ messages[0].content }}"
        with checkpoint_assets({"tokenizer": files}) as assets:
            tokenizer = anima_tokenizer(assets, "tokenizer")
            self.assertEqual(tokenizer.apply_chat_template([{"role": "user", "content": "only"}], tokenize=False), "checkpoint only")

    def test_missing_config_does_not_load_package_tokenizer(self) -> None:
        with checkpoint_assets({"tokenizer": {}}) as assets, self.assertRaises(InvalidRequest) as failure:
            anima_tokenizer(assets, "tokenizer")
        self.assertEqual(failure.exception.code, "model_asset_missing")
        self.assertIn("tokenizer/tokenizer_config.json", str(failure.exception))

    def test_missing_tokenizer_names_the_required_member(self) -> None:
        files = tiny_tokenizer()
        del files["tokenizer.json"]
        with checkpoint_assets({"tokenizer": files}) as assets, self.assertRaises(InvalidRequest) as failure:
            anima_tokenizer(assets, "tokenizer")
        self.assertEqual(failure.exception.code, "model_asset_missing")
        self.assertIn("tokenizer/tokenizer.json", str(failure.exception))


class SdxlCheckpointDataTests(unittest.TestCase):
    def test_both_vocabularies_come_from_the_checkpoint(self) -> None:
        vocabulary = json.dumps({"<|startoftext|>": 0, "<|endoftext|>": 1, "h": 2, "i</w>": 3, "!": 4}).encode()
        with checkpoint_assets({name: {"vocab.json": vocabulary, "merges.txt": b"#version: 0.2\n"} for name in ("tokenizer", "tokenizer_2")}) as assets:
            for name in ("tokenizer", "tokenizer_2"):
                self.assertEqual(sdxl_tokenizer(assets, name)("hi").input_ids, [0, 2, 3, 1])

    def test_absent_checkpoint_pair_is_not_replaced_by_package_data(self) -> None:
        with checkpoint_assets({"tokenizer": {}}) as assets, self.assertRaises(InvalidRequest) as failure:
            sdxl_tokenizer(assets, "tokenizer")
        self.assertEqual(failure.exception.code, "model_asset_missing")
        self.assertIn("tokenizer/vocab.json", str(failure.exception))

    def test_partial_pair_names_the_missing_member(self) -> None:
        with checkpoint_assets({"tokenizer": {"vocab.json": b"{}"}}) as assets, self.assertRaises(InvalidRequest) as failure:
            sdxl_tokenizer(assets, "tokenizer")
        self.assertEqual(failure.exception.code, "model_asset_missing")
        self.assertIn("tokenizer/merges.txt", str(failure.exception))


if __name__ == "__main__":
    unittest.main()
