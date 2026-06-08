"""
Înregistrează trafic normal de la cameră RTSP într-un singur fișier video per sesiune
(ex. Dataset/train_normal/Train001/normal_recording.mp4).

Pentru mai puține cadre la antrenament, folosește extract_frames.py --every 3
(fără să pierzi materialul sursă din MP4).
"""
import argparse
import os
import re

import cv2
from dotenv import load_dotenv

load_dotenv()


def _resolve_video_path(output_dir: str, basename: str, overwrite: bool) -> str:
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, f"{basename}.mp4")
    if overwrite or not os.path.isfile(base_path):
        return base_path

    existing = [
        f
        for f in os.listdir(output_dir)
        if re.match(rf"^{re.escape(basename)}(_\d+)?\.mp4$", f)
    ]
    if not existing:
        return base_path

    suffixes = []
    for name in existing:
        match = re.match(rf"^{re.escape(basename)}_(\d+)\.mp4$", name)
        if match:
            suffixes.append(int(match.group(1)))
        elif name == f"{basename}.mp4":
            suffixes.append(0)

    next_idx = max(suffixes) + 1
    return os.path.join(output_dir, f"{basename}_{next_idx:02d}.mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Înregistrare continuă trafic normal (un MP4 per sesiune)."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        #default="Dataset/train/frames/Train006",
        default="Dataset/test/frames/Test005",
        help="Folder sesiune (ex. Dataset/train_normal/Train001).",
    )
    parser.add_argument(
        "--basename",
        default="normal_recording",
        help="Numele fișierului video (fără .mp4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Suprascrie normal_recording.mp4 dacă există (altfel creează _01, _02, ...).",
    )
    parser.add_argument(
        "--duration-min",
        type=float,
        default=5.0,
        help="Durata maximă în minute; 0 = fără limită (oprire doar cu Q). Implicit: 10.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    user = os.getenv("CAM_USER")
    password = os.getenv("CAM_PASS")
    ip = os.getenv("CAM_IP")
    stream = os.getenv("CAM_STREAM", "stream1")
    url = f"rtsp://{user}:{password}@{ip}:554/{stream}"

    print(f"Încercare de conectare la {ip}...")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Eroare: Nu s-a putut deschide fluxul video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 15.0

    output_path = _resolve_video_path(args.output_dir, args.basename, args.overwrite)
    max_frames = (
        int(args.duration_min * 60 * fps) if args.duration_min > 0 else None
    )

    print(f"Conectat: {width}x{height} @ {fps:.0f} FPS")
    print(f"Salvare continuă în: {output_path}")
    if max_frames:
        print(f"Se oprește automat după {args.duration_min:g} min (~{max_frames} cadre).")
    print("Apasă 'q' în fereastra video pentru oprire manuală.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Eroare: Nu s-a putut crea fișierul video.")
        cap.release()
        return

    frames_written = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Eroare: S-a pierdut conexiunea cu camera.")
                break

            out.write(frame)
            frames_written += 1

            if frames_written % int(fps * 5) == 0:
                minutes = frames_written / fps / 60
                print(f"  … {frames_written} cadre (~{minutes:.1f} min)")

            cv2.imshow("Inregistrare (Q = oprire)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Înregistrare oprită de utilizator.")
                break

            if max_frames and frames_written >= max_frames:
                print(f"Durata de {args.duration_min:g} minute a fost atinsă.")
                break
    finally:
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        duration_sec = frames_written / fps if fps else 0
        print(
            f"Salvat: {output_path} ({frames_written} cadre, ~{duration_sec / 60:.1f} min)"
        )


if __name__ == "__main__":
    main()
