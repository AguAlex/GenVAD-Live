"""
GenVAD v4 — Future Frame Prediction (UCSD Ped2 / CUHK Avenue).

Antrenează un U-Net să prezică cadru t din cadrele t-k..t-1.
Scor anomalie = MSE(predicție, cadru real) — aliniat la v3 (micro/macro AUC).

Colab UCSD:
  !pip install -q torch opencv-python-headless scikit-learn numpy scipy
  %cd /content/drive/MyDrive/Licenta/GenVAD-Live
  !python v4_future/main.py --dataset ucsd --data_path /content/UCSD_Dataset \\
      --output_dir /content/v4_ucsd --device cuda

Colab Avenue:
  !python v4_future/main.py --dataset avenue --data_path /content/Avenue_Dataset \\
      --output_dir /content/v4_avenue --device cuda
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_V4_ROOT = Path(__file__).resolve().parent
if str(_V4_ROOT) not in sys.path:
    sys.path.insert(0, str(_V4_ROOT))

from configs.configs import get_configs
from data.dataset import discover_extension, load_sequences, parse_resize
from data.temporal_dataset import FutureFrameTrainDataset
from engine import (
    append_log,
    load_checkpoint,
    run_evaluation,
    save_checkpoint,
    train_one_epoch,
)
from model.future_predictor import FutureFrameUNet

DATASET_TITLES = {"ucsd": "UCSD Ped2", "avenue": "CUHK Avenue"}


def main():
    args = get_configs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.isdir(args.data_path):
        raise SystemExit(f"❌ Dataset inexistent: {args.data_path}")

    size_wh = parse_resize(args.resize)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    print(f"Device: {device} | Dataset: {args.dataset} | Resize: {args.resize}")

    extension = discover_extension(
        os.path.join(args.data_path, "train", "frames")
    )
    train_seqs = load_sequences(
        args.data_path, "train", args.dataset, extension
    )
    test_seqs = load_sequences(
        args.data_path,
        "test",
        args.dataset,
        extension,
        avenue_gt_path=args.avenue_gt_path,
    )
    print(f"Train: {len(train_seqs)} secvențe | Test: {len(test_seqs)} secvențe")

    model = FutureFrameUNet(
        n_input_frames=args.n_input_frames,
        base_channels=args.base_channels,
    ).to(device)

    t0 = time.time()
    best_macro = -1.0
    start_epoch = 0

    if args.run_type in ("train", "train_eval"):
        train_ds = FutureFrameTrainDataset(train_seqs, args.n_input_frames, size_wh)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        print(f"Samples antrenare: {len(train_ds)} (n_input={args.n_input_frames})")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        if args.resume:
            start_epoch = load_checkpoint(args.resume, model, device, optimizer)
            print(f"Reluare de la epoch {start_epoch}")

        for epoch in range(start_epoch, args.epochs):
            avg_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch,
                args.print_freq,
            )
            print(f"Epoch {epoch} train loss: {avg_loss:.6f}")
            save_checkpoint(
                output_dir / "checkpoint-last.pth",
                model,
                optimizer,
                epoch,
                args,
            )

            log_entry = {"epoch": epoch, "train_loss": avg_loss}

            if args.eval_each_epoch and args.run_type == "train_eval":
                metrics = run_evaluation(model, test_seqs, args, size_wh, device)
                log_entry.update(
                    {f"test_{k}": v for k, v in metrics.items() if k != "per_video_auc"}
                )
                print(
                    f"  test Micro: {metrics['micro']:.4f} "
                    f"Macro: {metrics['macro']:.4f}"
                )
                if metrics["macro"] > best_macro:
                    best_macro = metrics["macro"]
                    save_checkpoint(
                        output_dir / "checkpoint-best.pth",
                        model,
                        optimizer,
                        epoch,
                        args,
                        metrics,
                    )
            append_log(output_dir, log_entry)

    elif args.run_type == "eval":
        ckpt = args.resume or str(output_dir / "checkpoint-best.pth")
        if not os.path.isfile(ckpt):
            ckpt = str(output_dir / "checkpoint-last.pth")
        if not os.path.isfile(ckpt):
            raise SystemExit(f"❌ Lipsește checkpoint: {ckpt}")
        load_checkpoint(ckpt, model, device)
        print(f"Checkpoint încărcat: {ckpt}")

    results = {
        "method": "FutureFramePrediction",
        "dataset": args.dataset,
        "data_path": args.data_path,
        "n_input_frames": args.n_input_frames,
        "epochs": args.epochs,
        "resize": args.resize,
        "filt_range": args.filt_range,
        "filt_mu": args.filt_mu,
        "base_channels": args.base_channels,
    }

    if args.run_type in ("train_eval", "eval"):
        if not args.eval_each_epoch and args.run_type == "train_eval":
            best_path = output_dir / "checkpoint-best.pth"
            if best_path.is_file():
                load_checkpoint(str(best_path), model, device)
            else:
                load_checkpoint(str(output_dir / "checkpoint-last.pth"), model, device)

        metrics = run_evaluation(model, test_seqs, args, size_wh, device)
        results.update(metrics)
        title = DATASET_TITLES.get(args.dataset, args.dataset)
        print(f"\n=== {title} — Future Frame Prediction ===")
        print(f"Micro-AUC: {metrics['micro']:.4f}")
        print(f"Macro-AUC: {metrics['macro']:.4f}")
        for vid, auc_v in sorted(metrics["per_video_auc"].items()):
            print(f"  {vid}: {auc_v:.4f}")

        if args.run_type == "train_eval" and not args.eval_each_epoch:
            save_checkpoint(
                output_dir / "checkpoint-best.pth",
                model,
                None,
                args.epochs - 1,
                args,
                metrics,
            )

    results["elapsed_sec"] = int(time.time() - t0)
    out_json = output_dir / "results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRezultate: {out_json} | Timp: {results['elapsed_sec']} s")


if __name__ == "__main__":
    main()
