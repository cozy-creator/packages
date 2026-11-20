"""Check guide packing in the isolated candidate selected by PYTHONPATH."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from minimax_h3.guide_patch_selector import plate_patch_offsets
from minimax_h3.guide_rows import GUIDE, GUIDE_LATENTS, GUIDE_SAMPLING_FACTS, guide_first_prunable
from minimax_h3.official import OfficialH3Pipeline, patchify_video_latents


class State(SimpleNamespace):
    def get(self, key):
        return getattr(self, key)

    def set(self, key, value):
        setattr(self, key, value)


class PlatePatchCandidate(unittest.TestCase):
    def test_guide_packing_preserves_positions_other_modalities_and_rng(self):
        frames, height, width = 20, 3, 4
        patch = (1, 2, 2)
        full_rows = frames * height * width
        prefix_frames = 22
        torch_rng = torch.Generator().manual_seed(4)
        clean = torch.randn(1, 1, frames, 2 * height, 2 * width, generator=torch_rng)
        audio = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        text_rows, condition_video, condition_audio, target_audio = 2, 3, 1, 12
        rows = text_rows + condition_video + full_rows + condition_audio + target_audio

        pipeline = SimpleNamespace(
            patch_size=patch,
            keyframe_noise_aug=0.1,
            scheduler=SimpleNamespace(scale_noise=lambda value, sigma, noise: (1 - sigma) * value + sigma * noise),
        )

        def pack(sampling):
            state = State(
                position_ids=torch.arange(3 * rows, dtype=torch.float32).reshape(rows, 3),
                token_tags=torch.arange(rows),
                text_indices=torch.arange(text_rows),
                video_indices=torch.arange(text_rows, text_rows + condition_video + full_rows),
                audio_indices=torch.arange(text_rows + condition_video + full_rows, rows),
                latents=torch.rand(full_rows + condition_video, 4, generator=torch.Generator().manual_seed(12)),
                audio_latents=torch.arange((condition_audio + target_audio) * 2, dtype=torch.float32).reshape(-1, 2),
                num_condition_video_rows=condition_video,
                num_condition_audio_rows=condition_audio,
                num_audio_latents=6,
                generator=torch.Generator().manual_seed(198),
                continuation_delivery=SimpleNamespace(prefix_frames=prefix_frames),
            )
            state.set(GUIDE, SimpleNamespace(sampling=sampling))
            state.set(GUIDE_LATENTS, (clean, audio))
            count = OfficialH3Pipeline._pack_guide(pipeline, state)
            return state, count

        full, expected_full_count = pack("full")
        _, uniform_count = pack("uniform")
        selected, patch_count = pack("patch")
        self.assertEqual(full_rows, expected_full_count)
        self.assertEqual(patch_count, uniform_count)
        self.assertEqual(selected.get(GUIDE_SAMPLING_FACTS), (frames, frames, full_rows, patch_count))
        self.assertEqual(selected.num_condition_video_rows, condition_video + patch_count)
        self.assertEqual(selected.num_condition_audio_rows, full.num_condition_audio_rows)

        offsets = plate_patch_offsets(
            patchify_video_latents(clean, patch), (frames, height, width),
            first_prunable=guide_first_prunable(frames, prefix_frames),
            omit_frames=(full_rows - uniform_count) // (height * width),
        )
        self.assertTrue(torch.equal(selected.latents[:patch_count], full.latents[offsets]))
        self.assertTrue(torch.equal(selected.latents[patch_count:], full.latents[full_rows:]))
        self.assertTrue(torch.equal(selected.audio_latents, full.audio_latents))
        self.assertTrue(torch.equal(selected.generator.get_state(), full.generator.get_state()))

        old_video = torch.cat((full.video_indices[offsets], full.video_indices[full_rows:]))
        for current, original in (
            (selected.video_indices, old_video),
            (selected.audio_indices, full.audio_indices),
            (selected.text_indices, full.text_indices),
        ):
            self.assertTrue(torch.equal(selected.position_ids[current], full.position_ids[original]))
            self.assertTrue(torch.equal(selected.token_tags[current], full.token_tags[original]))


if __name__ == "__main__":
    unittest.main()
