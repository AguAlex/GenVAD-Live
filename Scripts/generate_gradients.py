import glob
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif"]

def compute_gradients(data_root_folder, step, split_type):
    """
    Calculează diferența dintre cadre pentru a extrage mișcarea (temporal gradients).
    data_root_folder: Calea către Custom_Dataset (ex: '../Model_Training/Custom_Dataset')
    split_type: 'train' sau 'test'
    """
    frames_dir = os.path.join(data_root_folder, split_type, "frames")
    
    # Am lăsat "gradients2" pentru că așa e scris original în train_dataset.py. 
    # Dacă ai redenumit în "gradients" acolo, schimbă și aici.
    gradients_dir = os.path.join(data_root_folder, split_type, "gradients2") 

    # Verificăm extensia pozelor tale
    extension = None
    for ext in IMG_EXTENSIONS:
        if len(list(glob.glob(os.path.join(frames_dir, f"*/*{ext}")))) > 0:
            extension = ext
            break
            
    if extension is None:
        print(f"❌ Nu am găsit imagini în {frames_dir}")
        return

    # Luăm toate folderele de clipuri (ex: clip_01, clip_02)
    video_dirs = list(glob.glob(os.path.join(frames_dir, "*")))
    print(f"📁 Am găsit {len(video_dirs)} clipuri în folderul {split_type}. Procesăm...")

    for video in tqdm(video_dirs):
        video_name = os.path.basename(video)
        img_paths = list(glob.glob(os.path.join(video, f"*{extension}")))
        # Sortare exactă (pentru a asigura ordinea cadrelor 1, 2, 3...)
        img_paths = sorted(img_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
        
        # Creăm folderul destinație pentru acest clip
        save_folder = os.path.join(gradients_dir, video_name)
        os.makedirs(save_folder, exist_ok=True)

        for i, img_path in enumerate(img_paths):
            previous = i - step
            if previous < 0:
                previous = i
            
            next_idx = i + step
            if next_idx >= len(img_paths):
                next_idx = i

            # Citire imagini
            previous_img = cv2.imread(img_paths[previous]).astype(np.int32)
            next_img = cv2.imread(img_paths[next_idx]).astype(np.int32)
            
            # Scăderea pentru a obține mișcarea (Temporal Gradient)
            gradient = np.abs(previous_img - next_img)
            gradient = gradient.astype(np.uint8)
            
            # Conversie BGR -> RGB (specific OpenCV către PIL)
            gradient = cv2.cvtColor(gradient, cv2.COLOR_BGR2RGB)
            
            # Salvare
            img_name = os.path.basename(img_path)
            Image.fromarray(gradient).save(os.path.join(save_folder, img_name))

if __name__ == "__main__":
    ROOT_FOLDER = "../Model_Training/Avenue_Dataset"
    
    compute_gradients(ROOT_FOLDER, step=1, split_type="test")
