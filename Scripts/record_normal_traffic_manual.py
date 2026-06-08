"""
Înregistrează trafic normal de la cameră RTSP până la oprire manuală (Q sau Ctrl+C).
Salvează un singur MP4 per sesiune (ex. Dataset/train/Train006/normal_recording.mp4).

Pentru înregistrare cu durată fixă, folosește record_normal_traffic.py.
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
        description="Înregistrare trafic normal până la oprire manuală (fără limită de timp)."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="Dataset/train/Train008",
        help="Folder sesiune (ex. Dataset/train/Train006).",
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

    print(f"Conectat: {width}x{height} @ {fps:.0f} FPS")
    print(f"Salvare continuă în: {output_path}")
    print("Înregistrare fără limită de timp.")
    print("Oprire: 'q' în fereastra video sau Ctrl+C în terminal.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Eroare: Nu s-a putut crea fișierul video.")
        cap.release()
        return

    frames_written = 0
    stopped_by_user = False
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
                print("Înregistrare oprită de utilizator (Q).")
                stopped_by_user = True
                break
    except KeyboardInterrupt:
        print("\nÎnregistrare oprită (Ctrl+C).")
        stopped_by_user = True
    finally:
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        duration_sec = frames_written / fps if fps else 0
        print(
            f"Salvat: {output_path} ({frames_written} cadre, ~{duration_sec / 60:.1f} min)"
        )
        if not stopped_by_user and frames_written > 0:
            print("(Oprire: pierdere conexiune sau eroare la citire.)")


if __name__ == "__main__":
    main()
