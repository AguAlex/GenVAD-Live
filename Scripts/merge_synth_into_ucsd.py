"""
Copiază frames_abnormal + masks_abnormal din Synth_Data_UCSD în UCSD_Dataset (Colab).

Colab — o celulă:

    !python /content/drive/MyDrive/Licenta/GenVAD-Live/Scripts/merge_synth_into_ucsd.py

sau rulează blocul din docstringul de la final (fără fișier).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

SUBDIRS = ("frames_abnormal", "masks_abnormal")

# Căi implicite Colab
DEFAULT_SYNTH = "/content/drive/MyDrive/Licenta/GenVAD-Live/Synth_Data_UCSD"
DEFAULT_UCSD = "/content/UCSD_Dataset"


def merge_tree(src_root: Path, dst_root: Path, dry_run: bool = False) -> tuple[int, int]:
    """Copiază recursiv fișiere din src_root în dst_root (păstrează structura TrainXXX/)."""
    copied = 0
    skipped = 0
    if not src_root.is_dir():
        print(f"  ⚠️  Lipsă sursă: {src_root}")
        return 0, 0

    for src in sorted(src_root.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        if dry_run:
            print(f"  [dry-run] {rel}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied += 1
    return copied, skipped


def main() -> None:
    p = argparse.ArgumentParser(description="Merge Synth_Data_UCSD → UCSD_Dataset/train")
    p.add_argument("--synth", type=str, default=DEFAULT_SYNTH, help="Rădăcină Synth_Data_UCSD")
    p.add_argument("--ucsd", type=str, default=DEFAULT_UCSD, help="Rădăcină UCSD_Dataset")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    synth = Path(args.synth).expanduser()
    ucsd = Path(args.ucsd).expanduser()
    train_synth = synth / "train"
    train_ucsd = ucsd / "train"

    if not train_ucsd.is_dir():
        raise SystemExit(f"❌ Nu există {train_ucsd} — montează/copiază mai întâi UCSD_Dataset în /content.")

    print(f"Sursă:  {train_synth}")
    print(f"Dest:   {train_ucsd}\n")

    total_copied = 0
    total_skipped = 0
    for name in SUBDIRS:
        src = train_synth / name
        dst = train_ucsd / name
        print(f"--- {name} ---")
        c, s = merge_tree(src, dst, dry_run=args.dry_run)
        print(f"  copiate: {c} | deja prezente (skip): {s}")
        total_copied += c
        total_skipped += s

    print(f"\n✅ Gata. Total copiate: {total_copied}, skip: {total_skipped}")
    if not args.dry_run and total_copied == 0 and total_skipped == 0:
        print("   Verifică că Synth_Data_UCSD/train/frames_abnormal conține fișiere.")


if __name__ == "__main__":
    main()
