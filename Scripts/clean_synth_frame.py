"""
Șterge un cadru anormal (și masca lui) din Synth_Data_UCSD.

Structură așteptată:
  <synth_root>/train/frames_abnormal/<TrainXXX>/<cadru>.tif
  <synth_root>/train/masks_abnormal/<TrainXXX>/<cadru>.tif

Colab — o singură celulă (editează valorile):

    import sys
    sys.path.insert(0, "/content/drive/MyDrive/Licenta/GenVAD-Live/Scripts")
    from clean_synth_frame import delete_abnormal_pair

    delete_abnormal_pair(
        synth_root="/content/drive/MyDrive/Licenta/GenVAD-Live/Synth_Data_UCSD",
        train_folder="Train004",
        frame=42,  # sau "000.tif" / "0042"
    )

CLI local:

    python Scripts/clean_synth_frame.py --train Train004 --frame 42
    python Scripts/clean_synth_frame.py --train Train004 --frames 12,42,108
"""

from __future__ import annotations

import argparse
import os
import sys

FRAME_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# Editează aici dacă rulezi fișierul direct în Colab fără argumente CLI.
COLAB_SYNTH_ROOT = "/content/drive/MyDrive/Licenta/GenVAD-Live/Synth_Data_UCSD"
COLAB_TRAIN_FOLDER = "Train001"
COLAB_FRAME: int | str = 0


def _is_image_name(name: str) -> bool:
    low = name.lower()
    return any(low.endswith(ext) for ext in FRAME_EXTENSIONS)


def _frame_index_from_name(name: str) -> int | None:
    stem = os.path.splitext(name)[0]
    return int(stem) if stem.isdigit() else None


def find_frame_filename(directory: str, frame_spec: int | str) -> str | None:
    """
    Găsește numele fișierului din folder după:
      - nume complet (ex. 000.tif)
      - index numeric (ex. 42 → 0042.tif sau 042.tif)
    """
    if not os.path.isdir(directory):
        return None

    spec = frame_spec if isinstance(frame_spec, str) else str(frame_spec)
    spec = spec.strip()

    if _is_image_name(spec):
        return spec if os.path.isfile(os.path.join(directory, spec)) else None

    if not spec.isdigit():
        return None

    target = int(spec)
    matches: list[str] = []
    for name in os.listdir(directory):
        if not _is_image_name(name):
            continue
        idx = _frame_index_from_name(name)
        if idx is not None and idx == target:
            matches.append(name)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        matches.sort()
        print(
            f"⚠️  Mai multe fișiere pentru indexul {target}: {matches}. "
            f"Folosesc {matches[0]} (specifică numele complet, ex. {matches[0]})."
        )
        return matches[0]
    return None


def delete_abnormal_pair(
    synth_root: str,
    train_folder: str,
    frame: int | str,
    *,
    dry_run: bool = False,
) -> bool:
    """
    Șterge cadru + mască pentru un TrainXXX și un index / nume de fișier.
    Returnează True dacă s-a șters cel puțin un fișier.
    """
    synth_root = os.path.abspath(os.path.expanduser(synth_root))
    train_folder = train_folder.strip()

    frames_dir = os.path.join(synth_root, "train", "frames_abnormal", train_folder)
    masks_dir = os.path.join(synth_root, "train", "masks_abnormal", train_folder)

    frame_name = find_frame_filename(frames_dir, frame)
    if frame_name is None:
        frame_name = find_frame_filename(masks_dir, frame)

    if frame_name is None:
        print(f"❌ Nu am găsit cadrul {frame!r} în {train_folder}.")
        print(f"   Căutat în: {frames_dir}")
        if os.path.isdir(frames_dir):
            sample = sorted(n for n in os.listdir(frames_dir) if _is_image_name(n))[:8]
            if sample:
                print(f"   Exemple existente: {', '.join(sample)}")
        return False

    frame_path = os.path.join(frames_dir, frame_name)
    mask_path = os.path.join(masks_dir, frame_name)

    deleted_any = False
    for label, path in (("frame", frame_path), ("mask", mask_path)):
        if os.path.isfile(path):
            if dry_run:
                print(f"  [dry-run] ar șterge {label}: {path}")
            else:
                os.remove(path)
                print(f"  🗑️  șters {label}: {path}")
            deleted_any = True
        else:
            print(f"  ⚠️  lipsește {label}: {path}")

    if not deleted_any:
        print("❌ Niciun fișier de șters (ambele căi lipsesc).")
    elif not dry_run:
        print(f"✅ Gata: {train_folder}/{frame_name}")
    return deleted_any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Șterge perechi frame+mască din Synth_Data_UCSD (frames_abnormal / masks_abnormal)."
    )
    p.add_argument(
        "--synth_root",
        type=str,
        default=None,
        help="Rădăcina Synth_Data_UCSD. Implicit: COLAB_SYNTH_ROOT sau ./Synth_Data_UCSD",
    )
    p.add_argument("--train", type=str, required=False, help="Subfolder, ex. Train004")
    p.add_argument(
        "--frame",
        type=str,
        default=None,
        help="Index cadru (ex. 42) sau nume fișier (ex. 0042.tif)",
    )
    p.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Mai multe cadre, separate prin virgulă (ex. 12,42,108)",
    )
    p.add_argument("--dry-run", action="store_true", help="Doar afișează ce s-ar șterge")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    synth_root = args.synth_root
    if not synth_root:
        local_default = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Synth_Data_UCSD",
        )
        synth_root = COLAB_SYNTH_ROOT if os.path.isdir(COLAB_SYNTH_ROOT) else local_default

    train = args.train or COLAB_TRAIN_FOLDER

    specs: list[int | str] = []
    if args.frames:
        for part in args.frames.split(","):
            part = part.strip()
            if part:
                specs.append(int(part) if part.isdigit() else part)
    elif args.frame is not None:
        f = args.frame.strip()
        specs.append(int(f) if f.isdigit() else f)
    else:
        specs.append(COLAB_FRAME)

    ok = 0
    for spec in specs:
        print(f"\n--- {train} / {spec} ---")
        if delete_abnormal_pair(synth_root, train, spec, dry_run=args.dry_run):
            ok += 1

    if len(specs) > 1:
        print(f"\nRezumat: {ok}/{len(specs)} șterse.")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
