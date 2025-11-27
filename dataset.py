import os
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class StarDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        self.root = root
        self.split = split
        self.samples = self._collect_pairs()

    def _collect_pairs(self) -> List[Tuple[str, str]]:
        split_dir = os.path.join(self.root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        input_dir = os.path.join(split_dir, "input")
        target_dir = os.path.join(split_dir, "target")
        if not os.path.isdir(input_dir) or not os.path.isdir(target_dir):
            raise FileNotFoundError(f"Expected subdirectories 'input' and 'target' inside {split_dir}")

        allowed_ext = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}

        def collect(dir_path):
            mapping = {}
            for fname in os.listdir(dir_path):
                stem, ext = os.path.splitext(fname)
                if ext.lower() not in allowed_ext:
                    continue
                mapping[stem.lower()] = os.path.join(dir_path, fname)
            return mapping

        input_map = collect(input_dir)
        target_map = collect(target_dir)

        common = sorted(set(input_map.keys()) & set(target_map.keys()))
        pairs = [(input_map[k], target_map[k]) for k in common]

        if not pairs:
            raise RuntimeError(f"No matching input/target pairs found in {input_dir} and {target_dir}")
        return pairs

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor

    def __getitem__(self, idx: int):
        input_path, target_path = self.samples[idx]
        inp = self._load_image(input_path)
        target = self._load_image(target_path)
        return inp, target
