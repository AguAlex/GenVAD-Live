"""Perechi (cadre t-k..t-1) -> cadru t pentru antrenament."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import VideoSequence, bgr_to_tensor_chw, read_frame_bgr


class FutureFrameTrainDataset(Dataset):
    def __init__(
        self,
        sequences: List[VideoSequence],
        n_input: int,
        size_wh: Tuple[int, int] | None,
    ):
        self.n_input = n_input
        self.size_wh = size_wh
        self.samples: list[tuple[list[str], int]] = []
        for seq in sequences:
            n = len(seq.frame_paths)
            for t in range(n_input, n):
                self.samples.append((seq.frame_paths, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        paths, t = self.samples[idx]
        past = []
        for i in range(t - self.n_input, t):
            past.append(bgr_to_tensor_chw(read_frame_bgr(paths[i], self.size_wh)))
        target = bgr_to_tensor_chw(read_frame_bgr(paths[t], self.size_wh))
        inp = np.concatenate(past, axis=0)
        return torch.from_numpy(inp), torch.from_numpy(target)
