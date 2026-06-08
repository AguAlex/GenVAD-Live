"""
Generează `train/frames_abnormal` și `train/masks_abnormal` fără Gemini:
tu furnizezi bbox-ul (mască) și prompt-ul; scriptul alege cadre aleatoare din train/frames.

Exemple:
  python Scripts/generate_anomalies_manual.py --dataset ucsd --data_root ./UCSD_Dataset \\
    --bbox 0.35,0.55,0.55,0.75 \\
    --prompt "Broken mirror or glass shards on sidewalk, photorealistic CCTV, ground level." \\
    --num_runs 5 --seed 42

  # Prompt din catalog (ID din generate_anomalies.py):
  python Scripts/generate_anomalies_manual.py --dataset avenue \\
    --bbox 0.2,0.65,0.38,0.82 --anomaly_id C10 --num_runs 3

  # Cadru fix (fără random):
  python Scripts/generate_anomalies_manual.py --dataset ucsd \\
    --subfolder Train001 --frame 000.tif --bbox 0.1,0.5,0.25,0.7 --anomaly_id A2

  # Mai multe specificații dintr-un JSON (fiecare intrare = o generare):
  python Scripts/generate_anomalies_manual.py --dataset ucsd --config specs.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from generate_anomalies import (  # noqa: E402
    ANOMALY_CATALOG,
    DATASET_DEFAULTS,
    DEFAULT_NEGATIVE_PROMPT,
    create_mask_image,
    discover_train_frames,
    setup_inpaint_pipe,
)


def parse_bbox(s: str) -> list[float]:
    """`x_min,y_min,x_max,y_max` normalizat 0.0–1.0."""
    parts = [float(x.strip()) for x in str(s).split(",")]
    if len(parts) != 4:
        raise ValueError("bbox trebuie să aibă exact 4 valori: x_min,y_min,x_max,y_max")
    x_min, y_min, x_max, y_max = parts
    for name, v in zip(("x_min", "y_min", "x_max", "y_max"), parts):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name}={v} în afara intervalului [0, 1]")
    if x_min >= x_max:
        raise ValueError("x_min trebuie < x_max")
    if y_min >= y_max:
        raise ValueError("y_min trebuie < y_max")
    return [x_min, y_min, x_max, y_max]


def resolve_prompt(anomaly_id: str | None, prompt: str | None) -> str:
    if prompt and str(prompt).strip():
        return str(prompt).strip()
    if anomaly_id:
        aid = str(anomaly_id).strip().upper()
        if aid not in ANOMALY_CATALOG:
            raise ValueError(f"ID necunoscut: {aid}. Vezi ANOMALY_CATALOG în generate_anomalies.py.")
        return ANOMALY_CATALOG[aid]
    raise ValueError("Furnizează --prompt sau --anomaly_id.")


def load_specs_from_config(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "runs" in data:
        items = data["runs"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("JSON: listă de intrări sau obiect cu cheia 'runs'.")
    if not items:
        raise ValueError("Fișierul de config nu conține intrări.")
    return items


def normalize_spec(raw: dict, cli_defaults: argparse.Namespace) -> dict:
    """Unifică o intrare din config sau valorile CLI într-un dict de rulare."""
    bbox_raw = raw.get("bbox") or raw.get("bounding_box")
    if bbox_raw is None:
        if not cli_defaults.bbox:
            raise ValueError("Lipsește bbox în config și nu ai trecut --bbox.")
        bbox = parse_bbox(cli_defaults.bbox)
    elif isinstance(bbox_raw, str):
        bbox = parse_bbox(bbox_raw)
    else:
        bbox = [float(x) for x in bbox_raw]
        if len(bbox) != 4:
            raise ValueError("bbox din config trebuie să aibă 4 numere.")

    sd_prompt = raw.get("sd_prompt") or raw.get("prompt")
    anomaly_id = raw.get("anomaly_id")
    sd_prompt = resolve_prompt(anomaly_id, sd_prompt or cli_defaults.prompt)

    neg = (raw.get("negative_prompt") or cli_defaults.negative_prompt or "").strip()
    neg = neg or DEFAULT_NEGATIVE_PROMPT

    sub = raw.get("subfolder") or cli_defaults.subfolder
    frame = raw.get("frame") or cli_defaults.frame

    return {
        "bbox": bbox,
        "sd_prompt": sd_prompt,
        "negative_prompt": neg,
        "subfolder": sub,
        "frame": frame,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inpainting manual (fără Gemini): bbox + prompt, cadru aleator din train/frames."
    )
    p.add_argument("--dataset", type=str, choices=("avenue", "ucsd"), required=True)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Câte cadre aleatoare (ignorat dacă --subfolder + --frame sau --config cu N intrări).",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model_id", type=str, default="RunDiffusion/Juggernaut-XL-v9")
    p.add_argument("--frames_abnormal_dir", type=str, default=None)
    p.add_argument("--masks_dir", type=str, default=None)

    p.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Mască dreptunghiulară normalizată: x_min,y_min,x_max,y_max (ex. 0.35,0.55,0.55,0.75).",
    )
    p.add_argument("--prompt", type=str, default=None, help="Prompt inpainting (sd_prompt).")
    p.add_argument(
        "--anomaly_id",
        type=str,
        default=None,
        help="Alternativ la --prompt: ID din ANOMALY_CATALOG (ex. C10).",
    )
    p.add_argument("--negative_prompt", type=str, default=None)
    p.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder din train/frames (ex. Train001). Cu --frame, fără random.",
    )
    p.add_argument(
        "--frame",
        type=str,
        default=None,
        help="Nume fișier cadru (ex. 000.tif). Necesită --subfolder.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help='JSON: listă [{"bbox": [...], "prompt": "..."}, ...] sau {"runs": [...]}.',
    )
    return p.parse_args()


def pick_frame(
    all_pairs: list[tuple[str, str]],
    rng: random.Random,
    subfolder: str | None,
    frame: str | None,
) -> tuple[str, str]:
    if subfolder or frame:
        if not subfolder or not frame:
            raise ValueError("Pentru cadru fix trebuie ambele: --subfolder și --frame.")
        key = (subfolder, frame)
        if key not in all_pairs:
            raise ValueError(f"Cadru inexistent în train/frames: {subfolder}/{frame}")
        return key
    return rng.choice(all_pairs)


def run_inpaint(
    pipe,
    original_frame_path: str,
    bbox: list[float],
    sd_prompt: str,
    neg_prompt: str,
    abnormal_frames_dir: str,
    abnormal_masks_dir: str,
    sub: str,
    frame_name: str,
) -> bool:
    init_image = Image.open(original_frame_path).convert("RGB")
    width, height = init_image.size
    mask_image = create_mask_image(width, height, bbox)

    result_image = pipe(
        prompt=sd_prompt,
        negative_prompt=neg_prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=30,
        guidance_scale=7.0,
    ).images[0]

    os.makedirs(os.path.join(abnormal_frames_dir, sub), exist_ok=True)
    os.makedirs(os.path.join(abnormal_masks_dir, sub), exist_ok=True)

    dest_frame_path = os.path.join(abnormal_frames_dir, sub, frame_name)
    dest_mask_path = os.path.join(abnormal_masks_dir, sub, frame_name)

    result_image.save(dest_frame_path)
    mask_image.save(dest_mask_path)
    print(f"  ✅ frame: {dest_frame_path}")
    print(f"      mask: {dest_mask_path}")
    return True


def main() -> None:
    args = parse_args()

    defaults = DATASET_DEFAULTS[args.dataset]
    data_root = os.path.abspath(os.path.expanduser(args.data_root or defaults["data_root"]))
    extensions = defaults["extensions"]
    rng = random.Random(args.seed)

    train_frames_dir = os.path.join(data_root, "train", "frames")
    abnormal_frames_dir = os.path.abspath(
        os.path.expanduser(
            args.frames_abnormal_dir or os.path.join(data_root, "train", "frames_abnormal")
        )
    )
    abnormal_masks_dir = os.path.abspath(
        os.path.expanduser(args.masks_dir or os.path.join(data_root, "train", "masks_abnormal"))
    )

    os.makedirs(abnormal_frames_dir, exist_ok=True)
    os.makedirs(abnormal_masks_dir, exist_ok=True)

    all_pairs = discover_train_frames(train_frames_dir, extensions)
    if not all_pairs:
        print(f"❌ Nu am găsit cadre în {train_frames_dir} cu extensiile {extensions}")
        sys.exit(1)

    if args.config:
        try:
            raw_specs = load_specs_from_config(os.path.expanduser(args.config))
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"❌ Config invalid: {e}")
            sys.exit(1)
        try:
            specs = [normalize_spec(item, args) for item in raw_specs]
        except ValueError as e:
            print(f"❌ {e}")
            sys.exit(1)
    else:
        try:
            specs = [normalize_spec({}, args)]
        except ValueError as e:
            print(f"❌ {e}")
            print("   Exemplu: --bbox 0.35,0.55,0.55,0.75 --anomaly_id C10")
            sys.exit(1)

    print(f"Dataset: {args.dataset} | root: {data_root}")
    print(f"Cadre indexate: {len(all_pairs)} | specificații: {len(specs)}")
    print(f"Ieșire cadre anormale: {abnormal_frames_dir}")
    print(f"Ieșire măști:        {abnormal_masks_dir}")
    print("Mod: manual (fără Gemini)")

    print("Încărcăm modelul de inpainting...")
    pipe = AutoPipelineForInpainting.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    device_mode = setup_inpaint_pipe(pipe)
    print(f"Pipeline: {device_mode}")

    ok = 0
    total_planned = 0

    for spec_i, spec in enumerate(specs):
        sub_fix = spec.get("subfolder")
        frame_fix = spec.get("frame")
        runs = 1 if (sub_fix and frame_fix) else args.num_runs
        total_planned += runs

        for run_i in range(runs):
            try:
                sub, frame_name = pick_frame(all_pairs, rng, sub_fix, frame_fix)
            except ValueError as e:
                print(f"❌ {e}")
                continue

            original_frame_path = os.path.join(train_frames_dir, sub, frame_name)
            label = f"[spec {spec_i + 1}/{len(specs)} run {run_i + 1}/{runs}]"
            print(f"\n{label} -> {sub}/{frame_name}")
            print(f"  bbox: {spec['bbox']}")
            print(f"  prompt: {spec['sd_prompt'][:120]}{'...' if len(spec['sd_prompt']) > 120 else ''}")

            if run_inpaint(
                pipe,
                original_frame_path,
                spec["bbox"],
                spec["sd_prompt"],
                spec["negative_prompt"],
                abnormal_frames_dir,
                abnormal_masks_dir,
                sub,
                frame_name,
            ):
                ok += 1

    print(f"\nGata: {ok}/{total_planned} reușite.")
    print(f"  frames_abnormal → {abnormal_frames_dir}")
    print(f"  masks_abnormal  → {abnormal_masks_dir}")


if __name__ == "__main__":
    main()
