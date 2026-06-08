import os
import cv2
import torch
import subprocess
import numpy as np
import glob
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import argparse

# --- CONFIGURARE MODALITĂȚI ---
# Poți schimba aceste variabile direct aici sau le poți lăsa ca default
CONFIG = {
    "ebsynth_path": "/content/ebsynth/bin/ebsynth",
    "device": "cuda",
    "sd_model": "runwayml/stable-diffusion-inpainting",
}

def setup_directories(baza_dir):
    dirs = [
        f"{baza_dir}/video_frames",
        f"{baza_dir}/ebsynth_output",
        f"{baza_dir}/ebsynth_masks"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs

def extract_frames(video_path, output_folder):
    print(f"--- [1/4] Extragere cadre din: {os.path.basename(video_path)} ---")

    if not os.path.exists(video_path):
        print(f"❌ EROARE CRITICĂ: Videoclipul nu există la calea specificată!")
        return 0

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ EROARE: OpenCV nu poate deschide fișierul video (codec incompatibil sau fișier corupt).")
        return 0

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(f"{output_folder}/{count:04d}.png", frame)
        count += 1
    cap.release()
    print(f"S-au extras {count} cadre.")
    return count

def create_anomaly_mask(image_path, output_path, anomaly_type="fire"):
    print(f"--- [2/4] Creare mască programatică ({anomaly_type}) ---")
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Setări dimensiuni în funcție de tip
    if anomaly_type == "fire":
        lw, lh = 200, 150
        offset_x, offset_y = -350, 250 # Pe asfalt/podea
    else: # intruder
        lw, lh = 180, 500
        offset_x, offset_y = -350, 0 # Stând în picioare
    
    cx, cy = w // 2 + offset_x, h // 2 + offset_y
    x1, y1 = max(0, cx - lw//2), max(0, cy - lh//2)
    x2, y2 = min(w, cx + lw//2), min(h, cy + lh//2)

    cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.imwrite(output_path, mask)
    return output_path

def generate_sd_keyframe(image_path, mask_path, output_path, prompt):
    print(f"--- [3/4] Generare Keyframe cu Stable Diffusion ---")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        CONFIG["sd_model"], torch_dtype=torch.float16, safety_checker=None
    ).to(CONFIG["device"])

    init_img = Image.open(image_path).convert("RGB").resize((512, 512))
    mask_img = Image.open(mask_path).convert("L").resize((512, 512))

    gen_img = pipe(
        prompt=prompt,
        negative_prompt="people, blurry, cartoon, artifacts, text",
        image=init_img, mask_image=mask_img,
        num_inference_steps=50, guidance_scale=15.0
    ).images[0]

    original_size = Image.open(image_path).size
    gen_img.resize(original_size).save(output_path)
    print(f"Keyframe generat salvat la: {output_path}")

def run_ebsynth_propagation(frames_dir, style_path, keyframe_path, output_dir, label):
    print(f"--- [4/4] Propagare EbSynth: {label} ---")
    frames = sorted(glob.glob(f"{frames_dir}/*.png"))
    for f in frames:
        out_f = f"{output_dir}/{os.path.basename(f)}"
        cmd = [CONFIG["ebsynth_path"], "-style", style_path, "-guide", keyframe_path, f, "-output", out_f]
        subprocess.run(cmd, stdout=subprocess.DEVNULL)
    print(f"Propagare {label} finalizată.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Calea către video normal")
    parser.add_argument("--output_baza", type=str, required=True, help="Folderul unde salvăm tot")
    parser.add_argument("--anomaly", type=str, default="fire", choices=["fire", "intruder"])
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()

    # 0. Setup
    f_cadre, f_video, f_masti = setup_directories(args.output_baza)
    
    # 1. Extracție
    total = extract_frames(args.video, f_cadre)
    key_idx = total // 2
    key_path = f"{f_cadre}/{key_idx:04d}.png"

    # 2. Mască
    m_path = f"{args.output_baza}/mask_ref.png"
    create_anomaly_mask(key_path, m_path, args.anomaly)

    # 3. Stable Diffusion
    sd_path = f"{args.output_baza}/sd_keyframe_ref.png"
    generate_sd_keyframe(key_path, m_path, sd_path, args.prompt)

    # 4. EbSynth
    run_ebsynth_propagation(f_cadre, sd_path, key_path, f_video, "VIDEO")
    run_ebsynth_propagation(f_cadre, m_path, key_path, f_masti, "MĂȘTI")

    print(f"\n🎉 GATA! Datele sintetice sunt în: {args.output_baza}")

if __name__ == "__main__":
    main()