"""Loading data comes from a copied checkpoint, including its tokenizer and learned statistics."""

from array import array
import asyncio
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "anima"), str(ROOT / "minimax-h3")]

import tensorfs
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from cozy_runtime.author import Config, InvalidRequest, ModelAssets, UnsupportedInput
from cozy_runtime import vlm

from anima.model import _tokenizer
from minimax_h3.official import checkpoint_processor
from minimax_h3.latent_upscaler import H3LatentUpscaler, _normalization
from minimax_h3.preparation.job import _produce_lane, _require_processor_assets, _validate_lane_config, retable


@contextmanager
def checkpoint_data(components, configs=None):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        file = root / "copied.cozytensors"
        with tensorfs.Writer(file, cache_dir=source) as writer:
            for name, assets in components.items():
                component = writer.component(name, config=(configs or {}).get(name, {}))
                component.put("fixture", array("f", [0]), dtype="F32", shape=[1])
                for path, raw in assets.items():
                    component.asset(path, raw)
        saved = writer.commit()
        saved.close()
        shutil.rmtree(source)
        checkpoint = tensorfs.open(file, cache_dir=root / "fresh", local_files_only=True)
        try:
            assert checkpoint.missing() == []
            data = {
                f"{name}/{path}": component.asset(path)
                for name, component in checkpoint.components.items()
                for path in component.assets
            }
            yield ModelAssets(data, lambda _: None), checkpoint
        finally:
            checkpoint.close()


def tiny_tokenizer():
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "checkpoint": 1, "only": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return {
        "tokenizer_config.json": json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"}).encode(),
        "tokenizer.json": tokenizer.to_str().encode(),
    }


class AnimaCheckpointDataTests(unittest.TestCase):
    def test_selected_checkpoint_tokenizers_win_over_installed_package_data(self):
        with checkpoint_data({"tokenizer": tiny_tokenizer(), "t5_tokenizer": tiny_tokenizer()}) as (assets, _):
            for folder in ("tokenizer", "t5_tokenizer"):
                self.assertEqual(_tokenizer(assets, folder).encode("checkpoint only"), [1, 2])

    def test_missing_tokenizer_is_not_replaced_by_installed_package_data(self):
        with checkpoint_data({"tokenizer": {}}) as (assets, _):
            with self.assertRaises(InvalidRequest) as failure:
                _tokenizer(assets, "tokenizer")
            self.assertEqual(failure.exception.code, "model_asset_missing")
            self.assertIn("tokenizer/tokenizer_config.json", str(failure.exception))


class H3CheckpointDataTests(unittest.TestCase):
    def test_checkpoint_processor_uses_its_own_template_and_preserves_tokenization(self):
        folder = ROOT / "minimax-h3/minimax_h3/processor"
        original = vlm.processor(folder)
        files = {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}
        config = json.loads(files["tokenizer_config.json"])
        config["chat_template"] = "checkpoint template {{ messages }}"
        files["tokenizer_config.json"] = json.dumps(config).encode()
        with checkpoint_data({"text_encoder": {f"processor/{name}": raw for name, raw in files.items()}}) as (assets, _):
            selected = checkpoint_processor(assets)
            self.assertEqual(selected.chat_template, config["chat_template"])
            self.assertEqual(selected.tokenizer.encode("a fox in fresh snow"), original.tokenizer.encode("a fox in fresh snow"))

    def test_missing_processor_names_the_checkpoint_asset(self):
        with checkpoint_data({"text_encoder": {}}) as (assets, _):
            with self.assertRaises(InvalidRequest) as failure:
                checkpoint_processor(assets)
            self.assertEqual(failure.exception.code, "model_asset_missing")
            self.assertIn("text_encoder/processor/tokenizer_config.json", str(failure.exception))


class UpscalerCheckpointDataTests(unittest.TestCase):
    def test_learned_normalization_changes_the_actual_cpu_forward(self):
        config = {"architecture": "minimax-h3-latent-upscaler-3d-conv-v1", "latent_normalization": {"enabled": True, "mean": [3] * 24, "std": [2] * 24}}
        with checkpoint_data({"upscaler": {}, "model": {}}, {"model": config}) as (_, checkpoint):
            stats = _normalization(Config(checkpoint.components["model"].config))

        class Double(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def forward(self, value, **_):
                return value * 2

        model = SimpleNamespace(graph=SimpleNamespace(components={"upscaler": Double()}, normalization=stats))
        result = inspect.unwrap(H3LatentUpscaler.upscale)(model, torch.full((1, 24, 1, 1, 1), 7), height=1, width=1)
        self.assertTrue(torch.equal(result, torch.full_like(result, 11)))

    def test_missing_normalization_does_not_use_packaged_statistics(self):
        with self.assertRaises(InvalidRequest) as failure:
            _normalization(Config({"architecture": "minimax-h3-latent-upscaler-3d-conv-v1"}))
        self.assertEqual(failure.exception.code, "config_missing")
        self.assertIn("latent_normalization", str(failure.exception))

    def test_malformed_normalization_refuses_before_model_construction(self):
        for field, values in (("mean", [float("nan")] * 24), ("mean", [0] * 23), ("std", [0] * 24)):
            config = {"enabled": True, "mean": [0] * 24, "std": [1] * 24, field: values}
            with self.subTest(field=field, values=values[:1]), self.assertRaises(InvalidRequest) as failure:
                _normalization(Config({"latent_normalization": config}))
            self.assertEqual(failure.exception.code, "config_invalid")
            self.assertIn(field, str(failure.exception))


class H3PreparationConfigTests(unittest.TestCase):
    def sections(self):
        return json.loads((ROOT / "minimax-h3/minimax_h3/preparation/assets/model-config.json").read_bytes())

    def test_supported_full_raw_and_curve_configs_remain_accepted(self):
        for variant in ("full", "raw", "curve"):
            config = self.sections()
            if variant == "full":
                for component, task in (("transformer", "fl2va"), ("transformer_ref", "ref2va")):
                    config[component]["cozy_h3"] = {"task": task, "modulation": "full"}
            else:
                config["text_encoder"].pop("cozy_h3")
            # Library bookkeeping does not alter construction or learned values.
            config["vae"]["_diffusers_version"] = "different-library-metadata"
            with self.subTest(variant=variant), checkpoint_data({name: {} for name in config}, config) as (_, checkpoint):
                _validate_lane_config(checkpoint, curve=variant == "curve")

    def test_changed_learned_statistics_refuse_before_the_producer_opens_a_writer(self):
        for component, field in (("vae", "latents_mean"), ("audio_vae", "latents_std")):
            config = self.sections()
            config[component][field][0] += 0.125
            with self.subTest(component=component, field=field), checkpoint_data({name: {} for name in config}, config) as (_, checkpoint):
                output = Mock(side_effect=AssertionError("a refused source opened a writer"))
                context = SimpleNamespace(tensorfs_source=lambda _: checkpoint, output=output)
                with self.assertRaises(UnsupportedInput) as failure:
                    _produce_lane(context, None, None, "bf16-full")
                self.assertEqual(failure.exception.code, "h3_lane_config")
                self.assertIn(f"{component}.{field}", str(failure.exception))
                output.assert_not_called()

    def processor_assets(self):
        folder = ROOT / "minimax-h3/minimax_h3/processor"
        return {f"processor/{path.name}": path.read_bytes() for path in folder.iterdir() if path.is_file()}

    def test_complete_source_processor_assets_are_accepted(self):
        config = self.sections()
        components = {name: {} for name in config}
        components["text_encoder"] = self.processor_assets()
        with checkpoint_data(components, config) as (_, checkpoint):
            _require_processor_assets(checkpoint)

    def test_missing_processor_asset_refuses_lane_production_before_a_writer(self):
        config = self.sections()
        components = {name: {} for name in config}
        components["text_encoder"] = self.processor_assets()
        del components["text_encoder"]["processor/vocab.json"]
        with checkpoint_data(components, config) as (_, checkpoint):
            output = Mock(side_effect=AssertionError("an incomplete source opened a writer"))
            context = SimpleNamespace(tensorfs_source=lambda _: checkpoint, output=output)
            with self.assertRaises(UnsupportedInput) as failure:
                _produce_lane(context, None, None, "bf16-full")
            self.assertEqual(failure.exception.code, "h3_processor_assets")
            self.assertIn("text_encoder/processor/vocab.json", str(failure.exception))
            output.assert_not_called()

    def test_missing_pruned_processor_refuses_retable_before_a_writer(self):
        config = self.sections()
        components = {name: {} for name in config}
        components["text_encoder"] = self.processor_assets()
        del components["text_encoder"]["processor/merges.txt"]
        with checkpoint_data(components, config) as (_, checkpoint):
            output = Mock(side_effect=AssertionError("an incomplete retable opened a writer"))
            context = SimpleNamespace(tensorfs_source=lambda _: checkpoint, output=output)
            with self.assertRaises(UnsupportedInput) as failure:
                asyncio.run(retable(context, pruned="pruned", full="full", tel=None))
            self.assertEqual(failure.exception.code, "h3_processor_assets")
            self.assertIn("text_encoder/processor/merges.txt", str(failure.exception))
            output.assert_not_called()

    def test_retable_refuses_drift_in_either_source_before_opening_a_writer(self):
        valid = self.sections()
        changed = self.sections()
        changed["vae"]["latents_std"][0] += 0.125
        components = {name: {} for name in valid}
        with checkpoint_data(components, valid) as (_, correct), checkpoint_data(components, changed) as (_, incorrect):
            for name in ("pruned", "full"):
                sources = {"pruned": correct, "full": correct, name: incorrect}
                output = Mock(side_effect=AssertionError("a refused retable opened a writer"))
                context = SimpleNamespace(tensorfs_source=sources.__getitem__, output=output)
                with self.subTest(source=name), self.assertRaises(UnsupportedInput) as failure:
                    asyncio.run(retable(context, pruned="pruned", full="full", tel=None))
                self.assertEqual(failure.exception.code, "h3_lane_config")
                self.assertIn("vae.latents_std", str(failure.exception))
                output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
