"""UCSD Ped2 + CUHK Avenue — aceeași structură ca v3."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

IMG_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


@dataclass
class VideoSequence:
    video_id: str
    frame_paths: List[str]
    labels: np.ndarray | None


def discover_extension(frames_root: str) -> str:
    for ext in IMG_EXTENSIONS:
        if glob.glob(os.path.join(frames_root, "*", f"*{ext}")):
            return ext
    raise FileNotFoundError(
        f"Nu s-au găsit cadre sub {frames_root} (extensii: {IMG_EXTENSIONS})"
    )


def list_video_dirs(frames_root: str) -> List[str]:
    dirs = sorted(glob.glob(os.path.join(frames_root, "*")))
    return [d for d in dirs if os.path.isdir(d)]


def _labels_from_avenue_mat(mat_path: str, num_frames: int) -> np.ndarray:
    try:
        import scipy.io as sio
    except ImportError as e:
        raise ImportError("Pentru Avenue .mat: pip install scipy") from e

    d = sio.loadmat(mat_path)
    if "volLabel" not in d:
        raise KeyError(f"Lipsește 'volLabel' în {mat_path}")
    vl = d["volLabel"]
    if vl.dtype == np.object_:
        lbls = [
            1.0 if np.sum(np.asarray(vl.flat[i])) > 0 else 0.0
            for i in range(vl.size)
        ]
        lbls = np.array(lbls, dtype=np.float64)
    else:
        lbls = np.asarray(vl, dtype=np.float64).reshape(-1)
    if lbls.size < num_frames:
        lbls = np.concatenate([lbls, np.zeros(num_frames - lbls.size)])
    elif lbls.size > num_frames:
        lbls = lbls[:num_frames]
    return lbls


def _labels_ucsd_bmps(gt_folder: str, num_frames: int) -> np.ndarray:
    lbls = []
    for gt_path in sorted(glob.glob(os.path.join(gt_folder, "*.bmp"))):
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        lbls.append(1.0 if mask is not None and np.sum(mask) > 0 else 0.0)
    lbls = np.array(lbls, dtype=np.float64)
    if lbls.size < num_frames:
        lbls = np.concatenate([lbls, np.zeros(num_frames - lbls.size)])
    elif lbls.size > num_frames:
        lbls = lbls[:num_frames]
    return lbls


def load_frame_labels(
    data_path: str,
    video_name: str,
    num_frames: int,
    dataset: str,
    avenue_gt_path: str | None = None,
) -> np.ndarray:
    if dataset == "ucsd":
        gt_folder = os.path.join(data_path, "test", f"{video_name}_gt")
        if os.path.isdir(gt_folder):
            return _labels_ucsd_bmps(gt_folder, num_frames)
        print(f"⚠️ Lipsește {gt_folder} — etichete 0.")
        return np.zeros(num_frames, dtype=np.float64)

    if dataset == "avenue":
        try:
            vid_id = int(str(video_name))
        except ValueError:
            vid_id = None
        if vid_id is not None:
            mat_name = f"{vid_id}_label.mat"
            for sub in (
                os.path.join(data_path, "testing_label_mask", mat_name),
                os.path.join(
                    data_path, "ground_truth_demo", "testing_label_mask", mat_name
                ),
            ):
                if os.path.isfile(sub):
                    try:
                        return _labels_from_avenue_mat(sub, num_frames)
                    except (OSError, KeyError, ImportError, ValueError) as e:
                        print(f"⚠️ .mat eșuat {sub}: {e}")

        gt_root = avenue_gt_path or os.path.join(data_path, "ground_truth")
        gt_txt = os.path.join(gt_root, f"{video_name}.txt")
        if os.path.isfile(gt_txt):
            raw = np.loadtxt(gt_txt)
            lbls = np.atleast_1d(np.squeeze(raw)).astype(np.float64)
            if lbls.size < num_frames:
                lbls = np.concatenate([lbls, np.zeros(num_frames - lbls.size)])
            elif lbls.size > num_frames:
                lbls = lbls[:num_frames]
            return lbls
        print(f"⚠️ GT lipsă pentru Avenue {video_name}.")
        return np.zeros(num_frames, dtype=np.float64)

    raise ValueError(f"Dataset necunoscut: {dataset}")


def load_sequences(
    data_path: str,
    split: str,
    dataset: str,
    extension: str | None = None,
    avenue_gt_path: str | None = None,
) -> List[VideoSequence]:
    frames_root = os.path.join(data_path, split, "frames")
    if extension is None:
        extension = discover_extension(frames_root)

    sequences = []
    for video_dir in list_video_dirs(frames_root):
        paths = sorted(glob.glob(os.path.join(video_dir, f"*{extension}")))
        if not paths:
            continue
        video_id = os.path.basename(video_dir)
        labels = None
        if split == "test":
            labels = load_frame_labels(
                data_path, video_id, len(paths), dataset, avenue_gt_path
            )
        sequences.append(VideoSequence(video_id, paths, labels))
    if not sequences:
        raise FileNotFoundError(f"Nicio secvență în {frames_root}")
    return sequences


def parse_resize(resize: str) -> Tuple[int, int] | None:
    if not resize or not resize.strip():
        return None
    parts = resize.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise ValueError(f'--resize invalid: "{resize}"')
    return int(parts[0]), int(parts[1])


def read_frame_bgr(path: str, size_wh: Tuple[int, int] | None) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Nu pot citi: {path}")
    if size_wh is not None:
        img = cv2.resize(img, size_wh)
    return img


def bgr_to_tensor_chw(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> float32 CHW [0,1]."""
    x = img_bgr.astype(np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))
