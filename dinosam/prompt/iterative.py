import numpy as np
import torch
from typing import Tuple


class DepthIterativePromptGenerator:
    """Generate error-based point prompts for iterative DepthSam training.

    Compares predicted mask with GT mask to find:
    - Positive points: in GT but not in prediction (missed foreground)
    - Negative points: in prediction but not in GT (false positive)

    Adapted from micro-sam's IterativePromptGenerator, simplified for 2D single-object case.
    """

    def __call__(
        self,
        gt_masks: torch.Tensor,
        pred_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 1 positive + 1 negative point per image.

        Args:
            gt_masks: (B, 1, H, W) binarized ground truth masks
            pred_masks: (B, 1, H, W) binarized predicted masks

        Returns:
            new_point_coords: (B, 2, 2) -- 1 positive + 1 negative point per image, (x, y) order
            new_point_labels: (B, 2) -- [1, 0]
        """
        assert gt_masks.shape == pred_masks.shape
        assert gt_masks.ndim == 4
        B = gt_masks.shape[0]

        # Error regions
        expected_diff = pred_masks - gt_masks  # 1 = false positive, -1 = missed foreground
        pos_region = (expected_diff == -1)   # missed foreground (need positive point)
        neg_region = (expected_diff == 1)    # false positive (need negative point)
        overlap_region = torch.logical_and(pred_masks == 1, gt_masks == 1)

        all_coords = []
        all_labels = []

        for i in range(B):
            # Positive point: sample from missed foreground
            pos_loc = torch.where(pos_region[i, 0])
            if len(pos_loc[0]) > 0:
                idx = np.random.choice(len(pos_loc[0]))
                # pos_loc[0] = y, pos_loc[1] = x; SAM expects (x, y)
                pos_coord = [pos_loc[1][idx].item(), pos_loc[0][idx].item()]
            else:
                # Fallback: sample from overlap region
                ovlp_loc = torch.where(overlap_region[i, 0])
                if len(ovlp_loc[0]) > 0:
                    idx = np.random.choice(len(ovlp_loc[0]))
                    pos_coord = [ovlp_loc[1][idx].item(), ovlp_loc[0][idx].item()]
                else:
                    # Last resort: sample from GT foreground
                    gt_loc = torch.where(gt_masks[i, 0] > 0)
                    if len(gt_loc[0]) > 0:
                        idx = np.random.choice(len(gt_loc[0]))
                        pos_coord = [gt_loc[1][idx].item(), gt_loc[0][idx].item()]
                    else:
                        h, w = gt_masks.shape[-2:]
                        pos_coord = [w / 2, h / 2]

            # Negative point: sample from false positive region
            neg_loc = torch.where(neg_region[i, 0])
            if len(neg_loc[0]) > 0:
                idx = np.random.choice(len(neg_loc[0]))
                neg_coord = [neg_loc[1][idx].item(), neg_loc[0][idx].item()]
            else:
                # Fallback: sample from background near object boundary
                neg_coord = self._sample_negative_near_object(gt_masks[i, 0])
                if neg_coord is None:
                    # Last resort: sample from any background
                    bg_loc = torch.where(gt_masks[i, 0] == 0)
                    if len(bg_loc[0]) > 0:
                        idx = np.random.choice(len(bg_loc[0]))
                        neg_coord = [bg_loc[1][idx].item(), bg_loc[0][idx].item()]
                    else:
                        h, w = gt_masks.shape[-2:]
                        neg_coord = [0.0, 0.0]

            all_coords.append([pos_coord, neg_coord])
            all_labels.append([1, 0])

        new_point_coords = torch.tensor(all_coords, dtype=torch.float32, device=gt_masks.device)
        new_point_labels = torch.tensor(all_labels, dtype=torch.int64, device=gt_masks.device)

        return new_point_coords, new_point_labels

    def _sample_negative_near_object(self, gt_mask: torch.Tensor, dilation: int = 3):
        """Sample a negative point near the object boundary in the background.

        Args:
            gt_mask: (H, W) binary ground truth
            dilation: pixels to expand the bounding box

        Returns:
            [x, y] coordinate or None if no valid location found
        """
        gt_loc = torch.where(gt_mask > 0)
        if len(gt_loc[0]) == 0:
            return None

        h, w = gt_mask.shape
        y_min = max(torch.min(gt_loc[0]).item() - dilation, 0)
        y_max = min(torch.max(gt_loc[0]).item() + dilation + 1, h)
        x_min = max(torch.min(gt_loc[1]).item() - dilation, 0)
        x_max = min(torch.max(gt_loc[1]).item() + dilation + 1, w)

        # Background in the dilated bounding box
        bbox_region = gt_mask[y_min:y_max, x_min:x_max]
        bg_in_bbox = (bbox_region == 0)

        bg_loc = torch.where(bg_in_bbox)
        if len(bg_loc[0]) == 0:
            return None

        idx = np.random.choice(len(bg_loc[0]))
        # Convert to global coordinates
        y = bg_loc[0][idx].item() + y_min
        x = bg_loc[1][idx].item() + x_min
        return [float(x), float(y)]
