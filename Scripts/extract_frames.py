# Cadre în același folder Train001 (recomandat cu un singur MP4):
#   python Scripts/extract_frames.py -i Dataset/train_normal/Train001 -o Dataset/train_normal --merge --every 3
import argparse
import cv2
import glob
import os
import re
# python Scripts/extract_frames.py -i Dataset/train/frames/Train003 -o Dataset/train/frames --merge --every 3
# python Scripts/extract_frames.py -i Dataset/test/frames/Test002 -o Dataset/test/frames --merge --every 3
from tqdm import tqdm


def _clip_sort_key(path: str) -> int:
    name = os.path.basename(path)
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


def _extract_video_frames(
    video_path: str,
    save_folder: str,
    start_index: int = 1,
    every: int = 1,
) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Eroare la deschiderea {video_path}")
        return start_index

    if every < 1:
        raise ValueError("--every trebuie să fie >= 1")

    frame_count = start_index
    read_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        read_index += 1
        if (read_index - 1) % every != 0:
            continue

        frame_filename = f"{str(frame_count).zfill(4)}.png"
        cv2.imwrite(os.path.join(save_folder, frame_filename), frame)
        frame_count += 1

    cap.release()
    return frame_count


def extract_frames_from_train_folder(
    input_folder: str,
    output_folder: str,
    merge_clips: bool = False,
    every: int = 1,
) -> None:
    """
    Extrage cadre din toate clipurile .mp4 dintr-un folder de train (ex. Train001).

    Implicit: fiecare clip devine subfolder propriu, cadre numerotate de la 1 per clip
    (ex. frames/Train001_normal_clip_0000/0001.png).

    Cu --merge: toate clipurile intr-un singur subfolder, numerotare continuă
    (ex. frames/Train001/0001.png, 0002.png, ...).
    """
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)
    train_name = os.path.basename(os.path.normpath(input_folder))

    video_paths = sorted(
        glob.glob(os.path.join(input_folder, "*.mp4")),
        key=_clip_sort_key,
    )

    if not os.path.isdir(input_folder):
        print(f"❌ Folder inexistent: {input_folder}")
        raise SystemExit(1)

    if not video_paths:
        print(f"❌ Nu am găsit niciun clip .mp4 în {input_folder}")
        raise SystemExit(1)

    print(f"📁 Train: {train_name}")
    print(f"🎥 Video(uri) găsite: {len(video_paths)}")
    if every > 1:
        print(f"⏭️  Se salvează 1 din {every} cadre (~{100 // every}% din total)")
    os.makedirs(output_folder, exist_ok=True)

    if merge_clips:
        save_folder = os.path.join(output_folder, train_name)
        os.makedirs(save_folder, exist_ok=True)

        next_index = 1
        for video_path in tqdm(video_paths, desc=f"Extragere {train_name}"):
            clip_name = os.path.splitext(os.path.basename(video_path))[0]
            next_index = _extract_video_frames(
                video_path, save_folder, start_index=next_index, every=every
            )
            print(f"   ✓ {clip_name}: cadre până la {next_index - 1:04d}")
    else:
        for video_path in tqdm(video_paths, desc=f"Extragere {train_name}"):
            clip_name = os.path.splitext(os.path.basename(video_path))[0]
            save_folder = os.path.join(output_folder, f"{train_name}_{clip_name}")
            os.makedirs(save_folder, exist_ok=True)
            last_index = _extract_video_frames(
                video_path, save_folder, start_index=1, every=every
            )
            print(f"   ✓ {clip_name}: {last_index - 1} cadre")


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.normpath(
        os.path.join(script_dir, "../Model_Training/Custom_Dataset/train/frames")
    )

    parser = argparse.ArgumentParser(
        description=(
            "Extrage cadre din clipurile normal_clip_XXXX.mp4 dintr-un folder de train "
            "(ex. Dataset/train_normal/Train001)."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Folderul de train cu clipuri (ex. Dataset/train_normal/Train001).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=default_output,
        help=f"Folderul rădăcină pentru cadre (implicit: {default_output}).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Unește toate clipurile într-un singur subfolder (Train001), numerotare continuă.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Salvează doar fiecare al N-lea cadru (ex. 3 => ~5 FPS din 15 FPS).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_frames_from_train_folder(
        args.input, args.output, merge_clips=args.merge, every=args.every
    )
    print("\n✅ Extragerea a fost finalizată cu succes!")
