"""Antrenare + inferență future frame prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import VideoSequence, bgr_to_tensor_chw, read_frame_bgr
from evaluate import compute_auc_metrics
from model.future_predictor import FutureFrameUNet


def train_one_epoch(
    model: FutureFrameUNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int,
) -> float:
    model.train()
    losses = []
    for step, (inp, target) in enumerate(loader):
        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred = model(inp)
        loss = torch.nn.functional.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % print_freq == 0:
            print(
                f"  epoch {epoch} step {step + 1}/{len(loader)} "
                f"loss {np.mean(losses[-print_freq:]):.6f}"
            )
    return float(np.mean(losses))


@torch.no_grad()
def score_test_sequences(
    model: FutureFrameUNet,
    sequences: List[VideoSequence],
    n_input: int,
    size_wh: Tuple[int, int] | None,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[float] = []
    labels: list[float] = []
    video_ids: list[str] = []

    for seq in sequences:
        paths = seq.frame_paths
        buffer: list[np.ndarray] = []
        for i, path in enumerate(paths):
            frame = bgr_to_tensor_chw(read_frame_bgr(path, size_wh))
            if i < n_input:
                predictions.append(0.0)
            else:
                inp = np.concatenate(buffer[-n_input:], axis=0)
                inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
                target_t = torch.from_numpy(frame).unsqueeze(0).to(device)
                pred = model(inp_t)
                score = FutureFrameUNet.anomaly_score(pred, target_t).item()
                predictions.append(score)
            buffer.append(frame)
            labels.append(float(seq.labels[i]))
            video_ids.append(seq.video_id)

    return (
        np.array(predictions, dtype=np.float64),
        np.array(labels, dtype=np.float64),
        np.array(video_ids),
    )


def save_checkpoint(
    path: Path,
    model: FutureFrameUNet,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    args,
    metrics: Dict | None = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "args": vars(args),
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: FutureFrameUNet,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return start_epoch


def run_evaluation(
    model: FutureFrameUNet,
    test_sequences: List[VideoSequence],
    args,
    size_wh,
    device: torch.device,
) -> Dict:
    preds, lbls, vids = score_test_sequences(
        model, test_sequences, args.n_input_frames, size_wh, device
    )
    metrics = compute_auc_metrics(
        preds,
        lbls,
        vids,
        filt_range=args.filt_range,
        filt_mu=args.filt_mu,
        use_filt=not args.no_filt,
    )
    return metrics


def append_log(output_dir: Path, record: dict) -> None:
    with open(output_dir / "log.txt", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
