"""CPU checks for the experimental H3 Plate patch selector."""

from __future__ import annotations

import unittest

import torch

from minimax_h3.guide_patch_selector import plate_patch_offsets


def keep_mask(video: torch.Tensor, *, first_prunable: int, omit_frames: int) -> torch.Tensor:
    grid = tuple(video.shape[:3])
    kept = plate_patch_offsets(video.flatten(0, 2), grid, first_prunable=first_prunable, omit_frames=omit_frames)
    mask = torch.zeros(grid, dtype=torch.bool)
    mask.flatten()[kept] = True
    return mask


class PlatePatchSelector(unittest.TestCase):
    def test_exact_budget_original_order_and_coverage(self):
        frames, height, width = 107, 5, 7
        video = torch.ones(frames, height, width, 4)
        prefix = 8
        omitted_frames = 21
        offsets = plate_patch_offsets(
            video.flatten(0, 2), (frames, height, width), first_prunable=prefix, omit_frames=omitted_frames
        )
        keep = keep_mask(video, first_prunable=prefix, omit_frames=omitted_frames)
        self.assertEqual(offsets.tolist(), keep.flatten().nonzero(as_tuple=True)[0].tolist())
        self.assertEqual(int(offsets.numel()), (frames - omitted_frames) * height * width)
        self.assertTrue(bool(keep[:prefix].all()))
        self.assertTrue(bool(keep[-1].all()))
        self.assertTrue(bool(keep.unfold(0, 3, 1).any(dim=-1).all()))
        self.assertTrue(bool(keep.unfold(1, 2, 1).unfold(2, 2, 1).any(dim=(-2, -1)).all()))
        self.assertGreater(int((~keep[prefix:-1]).any(dim=(1, 2)).sum()), 80)

    def test_moving_patches_cut_and_static_detail_outlive_flat_background(self):
        video = torch.ones(30, 12, 12, 4)
        video[3:23, 3:5, 3:5] = torch.arange(20)[:, None, None, None].remainder(2) * 5 + 1
        video[:, 8, 8] = 8
        video[25:] *= 4
        keep = keep_mask(video, first_prunable=1, omit_frames=6)
        self.assertTrue(bool(keep[3:23, 3:5, 3:5].all()))
        self.assertTrue(bool(keep[:, 8, 8].all()))
        self.assertTrue(bool(keep[24:26].all()))
        self.assertGreater(int((~keep[:, :2, :2]).sum()), 0)

    def test_drift_tiebreak_keeps_patches_changed_since_previous_anchor(self):
        # Every adjacent pair has the same relative L1 change, and spatial
        # detail is zero. Two frames of accumulated drift raise the score.
        video = (2.0 ** torch.arange(5)).view(5, 1, 1, 1).expand(-1, 2, 2, -1)
        keep = keep_mask(video, first_prunable=1, omit_frames=1)
        self.assertTrue(bool(keep[2, 0, 0]))
        self.assertTrue(bool(keep[3, 0, 1]))
        self.assertTrue(bool(keep[3, 1, 0]))

    def test_short_guide_with_no_budget_keeps_all_rows(self):
        video = torch.zeros(2, 2, 3, 4)
        keep = keep_mask(video, first_prunable=1, omit_frames=0)
        self.assertTrue(bool(keep.all()))


if __name__ == "__main__":
    unittest.main()
