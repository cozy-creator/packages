"""Check H3 Plate guide rows on CPU without loading model weights."""

from types import SimpleNamespace
import unittest

import torch

from minimax_h3.guide_rows import GUIDE, GUIDE_LATENTS, GUIDE_SAMPLING_FACTS, guide_frame_selection
from minimax_h3.official import OfficialH3Pipeline


class State(SimpleNamespace):
    def get(self, key):
        return getattr(self, key)

    def set(self, key, value):
        setattr(self, key, value)


class PlateSampling(unittest.TestCase):
    def test_matched_budget_and_coverage(self):
        latents = torch.zeros(1, 2, 107, 8, 8)
        self.assertEqual(guide_frame_selection(latents, "full", 0), tuple(range(107)))
        for prefix, head in ((0, 1), (22, 8), (24, 9), (56, 18)):
            uniform = guide_frame_selection(latents, "uniform", prefix)
            motion = guide_frame_selection(latents, "motion", prefix)
            self.assertEqual(uniform, motion)
            self.assertEqual(len(motion), 86)
            self.assertEqual(motion[:head], tuple(range(head)))
            self.assertEqual(motion[-1], 106)
            self.assertLessEqual(max(b - a for a, b in zip(motion, motion[1:])), 2)

    def test_local_change_preserves_both_sides_of_a_cut(self):
        latents = torch.zeros(1, 2, 20, 8, 8)
        uniform = guide_frame_selection(latents, "uniform", 0)
        cut = next(frame for frame in range(2, 18) if frame not in uniform)
        latents[:, :, cut:, 0, 0] = 1.0
        motion = guide_frame_selection(latents, "motion", 0)
        self.assertIn(cut - 1, motion)
        self.assertIn(cut, motion)
        self.assertEqual(len(motion), len(uniform))

    def test_short_guide_covered_by_join(self):
        latents = torch.zeros(1, 2, 8, 8, 8)
        self.assertEqual(guide_frame_selection(latents, "motion", 56), tuple(range(8)))

    def test_current_packer_keeps_exact_positions_audio_and_rng(self):
        frames, per_frame = 20, 4
        clean = torch.zeros(1, 1, frames, 4, 4)
        clean[:, :, 4:, 0, 0] = 1
        full_rows = frames * per_frame
        text_rows, condition_video, condition_audio, target_audio = 2, 3, 1, 12
        guide_audio = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        rows = text_rows + condition_video + full_rows + condition_audio + target_audio
        pipeline = SimpleNamespace(
            patch_size=(1, 2, 2), keyframe_noise_aug=0.5,
            scheduler=SimpleNamespace(scale_noise=lambda latents, _, noise: latents + noise),
        )

        def pack(sampling):
            state = State(
                position_ids=torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3),
                token_tags=torch.arange(rows),
                text_indices=torch.arange(text_rows),
                video_indices=torch.arange(text_rows, text_rows + condition_video + full_rows),
                audio_indices=torch.arange(text_rows + condition_video + full_rows, rows),
                latents=torch.arange((full_rows + condition_video) * 4, dtype=torch.float32).reshape(-1, 4),
                audio_latents=torch.arange((condition_audio + target_audio) * 2, dtype=torch.float32).reshape(-1, 2),
                num_condition_video_rows=condition_video,
                num_condition_audio_rows=condition_audio,
                num_audio_latents=6,
                generator=torch.Generator().manual_seed(5581),
                continuation_delivery=SimpleNamespace(prefix_frames=22),
            )
            state.set(GUIDE, SimpleNamespace(sampling=sampling))
            state.set(GUIDE_LATENTS, (clean, guide_audio))
            count = OfficialH3Pipeline._pack_guide(pipeline, state)
            return state, count

        full, full_count = pack("full")
        for sampling in ("uniform", "motion"):
            sampled, count = pack(sampling)
            kept = guide_frame_selection(clean, sampling, 22)
            offsets = [frame * per_frame + patch for frame in kept for patch in range(per_frame)]
            self.assertEqual(sampled.get(GUIDE_SAMPLING_FACTS), (frames, len(kept), full_count, count))
            self.assertEqual(count, 64)
            self.assertEqual(sampled.num_condition_video_rows, condition_video + count)
            self.assertEqual(sampled.num_condition_audio_rows, full.num_condition_audio_rows)
            self.assertTrue(torch.equal(sampled.position_ids[sampled.video_indices[:count]],
                                        full.position_ids[full.video_indices[offsets]]))
            self.assertTrue(torch.equal(sampled.token_tags[sampled.video_indices[:count]],
                                        full.token_tags[full.video_indices[offsets]]))
            self.assertTrue(torch.equal(sampled.latents[:count], full.latents[offsets]))
            self.assertTrue(torch.equal(sampled.latents[count:], full.latents[full_count:]))
            self.assertTrue(torch.equal(sampled.position_ids[sampled.video_indices[count:]],
                                        full.position_ids[full.video_indices[full_count:]]))
            self.assertTrue(torch.equal(sampled.audio_latents, full.audio_latents))
            self.assertTrue(torch.equal(sampled.position_ids[sampled.audio_indices],
                                        full.position_ids[full.audio_indices]))
            self.assertTrue(torch.equal(sampled.position_ids[sampled.text_indices],
                                        full.position_ids[full.text_indices]))
            self.assertTrue(torch.equal(sampled.generator.get_state(), full.generator.get_state()))


if __name__ == "__main__":
    unittest.main()
