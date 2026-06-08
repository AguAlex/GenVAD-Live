"""Evaluare frame-level AUC (micro / macro) — aliniat la v1/v2."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn import metrics

from util.abnormal_utils import filt


def compute_auc_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    video_ids: np.ndarray,
    filt_range: int = 302,
    filt_mu: int = 21,
    use_filt: bool = True,
) -> Dict[str, float]:
    predictions = np.nan_to_num(predictions.astype(np.float64), nan=0.0)
    labels = labels.astype(np.float64)
    video_ids = np.asarray(video_ids)

    per_video_auc: List[float] = []
    filtered_preds: List[np.ndarray] = []
    filtered_labels: List[np.ndarray] = []

    for vid in np.unique(video_ids):
        mask = video_ids == vid
        pred = predictions[mask]
        lbl = labels[mask]
        if use_filt:
            pred = filt(pred, range=filt_range, mu=filt_mu)
        filtered_preds.append(pred)
        filtered_labels.append(lbl)

        lbl_auc = np.array([0.0] + list(lbl) + [1.0])
        pred_auc = np.array([0.0] + list(pred) + [1.0])
        fpr, tpr, _ = metrics.roc_curve(lbl_auc, pred_auc)
        per_video_auc.append(metrics.auc(fpr, tpr))

    macro_auc = float(np.nanmean(per_video_auc))
    all_pred = np.concatenate(filtered_preds)
    all_lbl = np.concatenate(filtered_labels)
    fpr, tpr, _ = metrics.roc_curve(all_lbl, all_pred)
    micro_auc = float(np.nan_to_num(metrics.auc(fpr, tpr), nan=1.0))

    return {
        "micro": micro_auc,
        "macro": macro_auc,
        "per_video_auc": dict(zip(np.unique(video_ids).tolist(), per_video_auc)),
    }
