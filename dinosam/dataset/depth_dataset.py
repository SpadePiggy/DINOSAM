import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy import ndimage


def _folder_to_depth(folder_name: str) -> float:
    if folder_name.startswith("f"):
        return -float(folder_name[1:])
    return float(folder_name)


class DepthDataset(Dataset):
    """Dataset for depth-prediction SAM fine-tuning.

    Directory layout:
        root/
          0/   (depth=0)
          10/  (depth=10)
          f10/ (depth=-10)
          ...
        Each folder contains paired .jpg images and .png masks with matching names.
    """

    def __init__(self, root: str, max_points: int = 3):
        self.root = root
        self.max_points = max_points
        self.samples = []
        self._scan_folders()

    def _scan_folders(self):
        for folder_name in sorted(os.listdir(self.root)):
            folder_path = os.path.join(self.root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            depth = _folder_to_depth(folder_name)
            for fname in sorted(os.listdir(folder_path)):
                if not fname.endswith(".jpg"):
                    continue
                mask_fname = fname.replace(".jpg", ".png")
                mask_path = os.path.join(folder_path, mask_fname)
                if not os.path.exists(mask_path):
                    continue
                self.samples.append({
                    "image": os.path.join(folder_path, fname),
                    "mask": mask_path,
                    "depth": depth,
                })

    def __len__(self):
        return len(self.samples)

    def _get_point_prompts(self, mask_arr: np.ndarray):
        """Extract centroids of the largest connected components as point prompts.

        Returns:
            point_coords: (N, 2) array of (x, y) coordinates
            point_labels: (N,) array of 1s
        """
        binary = mask_arr > 0
        labeled, num_features = ndimage.label(binary)
        if num_features == 0:
            # Fallback: use image center
            h, w = mask_arr.shape
            point_coords = np.array([[w / 2, h / 2]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int64)
            return point_coords, point_labels

        # Compute area per component and sort descending
        component_ids = np.arange(1, num_features + 1)
        areas = ndimage.sum(binary, labeled, component_ids)
        sorted_ids = component_ids[np.argsort(areas)[::-1]]
        selected = sorted_ids[:min(self.max_points, num_features)]

        coords = []
        for cid in selected:
            ys, xs = np.where(labeled == cid)
            cx = xs.mean()
            cy = ys.mean()
            coords.append([cx, cy])

        point_coords = np.array(coords, dtype=np.float32)
        point_labels = np.ones(len(coords), dtype=np.int64)
        return point_coords, point_labels

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image"]).convert("RGB")
        image_arr = np.array(image, dtype=np.float32)  # (H, W, 3), [0, 255]
        image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1)  # (3, H, W)

        mask = Image.open(sample["mask"]).convert("L")
        mask_arr = np.array(mask)  # (H, W), uint8

        point_coords, point_labels = self._get_point_prompts(mask_arr)
        point_coords_tensor = torch.from_numpy(point_coords)  # (N, 2)
        point_labels_tensor = torch.from_numpy(point_labels)  # (N,)

        depth = torch.tensor(sample["depth"] / 100.0, dtype=torch.float32)

        # Binary mask for loss computation
        binary_mask = (mask_arr > 0).astype(np.float32)
        gt_mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)  # (1, H, W)

        h, w = image_arr.shape[:2]
        original_size = torch.tensor([h, w], dtype=torch.int64)

        return {
            "image": image_tensor,
            "point_coords": point_coords_tensor,
            "point_labels": point_labels_tensor,
            "depth": depth,
            "gt_mask": gt_mask_tensor,
            "original_size": original_size,
        }


def collate_fn(batch):
    """Custom collate to handle variable-length point prompts."""
    images = torch.stack([b["image"] for b in batch])
    depths = torch.stack([b["depth"] for b in batch])
    gt_masks = torch.stack([b["gt_mask"] for b in batch])
    original_sizes = torch.stack([b["original_size"] for b in batch])

    # Pad point_coords and point_labels to max length in batch
    max_points = max(b["point_coords"].shape[0] for b in batch)
    point_coords_list = []
    point_labels_list = []
    for b in batch:
        n = b["point_coords"].shape[0]
        pad_size = max_points - n
        if pad_size > 0:
            pc = torch.cat([b["point_coords"], torch.zeros(pad_size, 2)])
            pl = torch.cat([b["point_labels"], torch.zeros(pad_size, dtype=torch.int64)])
        else:
            pc = b["point_coords"]
            pl = b["point_labels"]
        point_coords_list.append(pc)
        point_labels_list.append(pl)

    point_coords = torch.stack(point_coords_list)  # (B, max_N, 2)
    point_labels = torch.stack(point_labels_list)   # (B, max_N)

    return {
        "image": images,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "depth": depths,
        "gt_mask": gt_masks,
        "original_size": original_sizes,
    }
