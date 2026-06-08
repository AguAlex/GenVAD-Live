from __future__ import annotations

import numpy as np
import torch
import cv2

from inference_config import InferenceConfig


def _bgr_bytes_to_bgr_array(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Imagine invalidă sau format necunoscut.")
    return bgr


def image_bytes_to_sample_batch(image_bytes: bytes, cfg: InferenceConfig) -> torch.Tensor:
    """
    Aceeași logică ca în v2/data/test_dataset.py::__getitem__ pentru `img` -> tensore `samples`.
    """
    img = _bgr_bytes_to_bgr_array(image_bytes)
    h, w = cfg.input_size

    if img.shape[0] != h or img.shape[1] != w:
        # cv2.resize primește (lățime, înălțime) = (W, H) = input_size[::-1]
        img = cv2.resize(img, cfg.input_size[::-1], interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32)
    img = (img - 127.5) / 127.5
    img = np.swapaxes(img, 0, -1).swapaxes(1, -1)  # HWC -> CHW

    tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 3, H, W)
    return tensor