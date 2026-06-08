"""MOG2: învățare fundal pe train, scor anomalie pe test (learningRate=0)."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class MOG2AnomalyDetector:
    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = False,
        fg_threshold: int = 200,
        morph_kernel: int = 3,
        score_mode: str = "fg_ratio",
    ):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.detect_shadows = detect_shadows
        self.fg_threshold = fg_threshold
        self.morph_kernel = morph_kernel
        self.score_mode = score_mode
        self._kernel = None
        if morph_kernel > 0:
            k = morph_kernel | 1  # impar
            self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def learn(self, frame_bgr: np.ndarray) -> None:
        """Actualizează modelul de fundal (faza train)."""
        self.bg.apply(frame_bgr)

    def _postprocess_mask(self, fg_raw: np.ndarray) -> np.ndarray:
        if self.detect_shadows:
            mask = (fg_raw >= self.fg_threshold).astype(np.uint8) * 255
        else:
            _, mask = cv2.threshold(fg_raw, self.fg_threshold, 255, cv2.THRESH_BINARY)
        if self._kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        return mask

    def score(self, frame_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Scor anomalie + mască foreground (fără update model).
        learningRate=0 îngheață fundalul învățat.
        """
        fg_raw = self.bg.apply(frame_bgr, learningRate=0)
        mask = self._postprocess_mask(fg_raw)
        h, w = mask.shape[:2]
        total = float(h * w) if h and w else 1.0

        if self.score_mode == "fg_ratio":
            s = float(np.count_nonzero(mask)) / total
        elif self.score_mode == "fg_pixels":
            s = float(np.count_nonzero(mask))
        elif self.score_mode == "max_blob_area":
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if n <= 1:
                s = 0.0
            else:
                areas = stats[1:, cv2.CC_STAT_AREA]
                s = float(np.max(areas)) / total
        else:
            raise ValueError(f"score_mode necunoscut: {self.score_mode}")

        return s, mask
