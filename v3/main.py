"""
GenVAD v3 — MOG2 pe UCSD Ped2 sau CUHK Avenue.

UCSD (Colab):
  !python v3/main.py --dataset ucsd --data_path /content/UCSD_Dataset

Avenue (Colab):
  !python v3/main.py --dataset avenue --data_path /content/Avenue_Dataset \\
      --resize 640x320 --output_dir /content/v3_mog2_avenue
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_V3_ROOT = Path(__file__).resolve().parent
if str(_V3_ROOT) not in sys.path:
    sys.path.insert(0, str(_V3_ROOT))

from configs.configs import get_configs
from data.dataset import (
    discover_extension,
    load_sequences,
    parse_resize,
    read_frame,
)
from evaluate import compute_auc_metrics
from mog2_detector import MOG2AnomalyDetector

DATASET_TITLES = {
    "ucsd": "UCSD Ped2",
    "avenue": "CUHK Avenue",
}


def fit_mog2_on_train(
    detector: MOG2AnomalyDetector, data_path: str, args, size_wh
) -> int:
    extension = discover_extension(os.path.join(data_path, "train", "frames"))
    train_seqs = load_sequences(data_path, "train", args.dataset, extension)
    n_frames = 0
    print(
        f"Antrenare MOG2 [{args.dataset}] pe {len(train_seqs)} secvențe train ({extension})..."
    )
    for seq in train_seqs:
        paths = seq.frame_paths
        if args.train_warmup_frames > 0:
            paths = paths[: args.train_warmup_frames]
        for path in paths:
            detector.learn(read_frame(path, size_wh))
            n_frames += 1
    print(f"  → {n_frames} cadre procesate pentru fundal.")
    return n_frames


def score_test_videos(
    detector: MOG2AnomalyDetector, data_path: str, args, size_wh
):
    extension = discover_extension(os.path.join(data_path, "test", "frames"))
    gt_path = args.avenue_gt_path if args.dataset == "avenue" else None
    test_seqs = load_sequences(
        data_path,
        "test",
        args.dataset,
        extension,
        avenue_gt_path=gt_path,
    )

    predictions: list[float] = []
    labels: list[float] = []
    video_ids: list[str] = []

    debug_dir = None
    if args.save_debug_masks:
        debug_dir = Path(args.output_dir) / "debug_masks"
        debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scorare test: {len(test_seqs)} secvențe...")
    for seq in test_seqs:
        for i, path in enumerate(seq.frame_paths):
            frame = read_frame(path, size_wh)
            score, mask = detector.score(frame)
            predictions.append(score)
            labels.append(float(seq.labels[i]))
            video_ids.append(seq.video_id)

            if debug_dir is not None and i % args.debug_every == 0:
                out_path = debug_dir / f"{seq.video_id}_{i:05d}.png"
                cv2.imwrite(str(out_path), mask)

    return (
        np.array(predictions),
        np.array(labels),
        np.array(video_ids),
    )


def main():
    args = get_configs()
    data_path = args.data_path
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    title = DATASET_TITLES.get(args.dataset, args.dataset)

    if not os.path.isdir(data_path):
        raise SystemExit(
            f"❌ Nu există dataset-ul: {data_path}\n"
            f"   Folosește --dataset {args.dataset} --data_path <cale> "
            "(Colab: /content/UCSD_Dataset sau /content/Avenue_Dataset)."
        )

    train_frames = os.path.join(data_path, "train", "frames")
    test_frames = os.path.join(data_path, "test", "frames")
    if not os.path.isdir(train_frames):
        raise SystemExit(f"❌ Lipsește {train_frames}")
    if args.run_type == "eval" and not os.path.isdir(test_frames):
        raise SystemExit(f"❌ Lipsește {test_frames}")

    if args.dataset == "avenue":
        has_mat = os.path.isdir(
            os.path.join(data_path, "testing_label_mask")
        ) or os.path.isdir(
            os.path.join(data_path, "ground_truth_demo", "testing_label_mask")
        )
        has_txt = os.path.isdir(args.avenue_gt_path) and any(
            f.endswith(".txt")
            for f in os.listdir(args.avenue_gt_path)
            if os.path.isfile(os.path.join(args.avenue_gt_path, f))
        )
        if args.run_type == "eval" and not has_mat and not has_txt:
            print(
                "⚠️ Avenue: nu am găsit testing_label_mask/*.mat nici ground_truth/*.txt — "
                "AUC poate fi 0. Copiază GT oficial sau rulează Scripts/convert_mat_to_txt.py."
            )

    size_wh = parse_resize(args.resize)
    if size_wh:
        print(f"Redimensionare: {size_wh[0]}x{size_wh[1]}")

    detector = MOG2AnomalyDetector(
        history=args.history,
        var_threshold=args.var_threshold,
        detect_shadows=args.detect_shadows,
        fg_threshold=args.fg_threshold,
        morph_kernel=args.morph_kernel,
        score_mode=args.score_mode,
    )

    t0 = time.time()
    n_train = fit_mog2_on_train(detector, data_path, args, size_wh)

    results = {
        "method": "MOG2",
        "dataset": args.dataset,
        "data_path": data_path,
        "n_train_frames": n_train,
        "history": args.history,
        "var_threshold": args.var_threshold,
        "score_mode": args.score_mode,
        "detect_shadows": args.detect_shadows,
        "fg_threshold": args.fg_threshold,
        "resize": args.resize or "native",
        "filt_range": args.filt_range,
        "filt_mu": args.filt_mu,
    }

    if args.run_type == "eval":
        preds, lbls, vids = score_test_videos(detector, data_path, args, size_wh)
        metrics = compute_auc_metrics(
            preds,
            lbls,
            vids,
            filt_range=args.filt_range,
            filt_mu=args.filt_mu,
            use_filt=not args.no_filt,
        )
        results.update(metrics)
        print(f"\n=== {title} — MOG2 ===")
        print(f"Micro-AUC: {metrics['micro']:.4f}")
        print(f"Macro-AUC: {metrics['macro']:.4f}")
        for vid, auc_v in sorted(metrics["per_video_auc"].items()):
            print(f"  {vid}: {auc_v:.4f}")

    results["elapsed_sec"] = int(time.time() - t0)
    out_json = Path(args.output_dir) / "results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRezultate salvate: {out_json}")
    print(f"Timp total: {results['elapsed_sec']} s")


if __name__ == "__main__":
    main()
