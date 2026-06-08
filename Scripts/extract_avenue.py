import cv2
import os
import glob
from tqdm import tqdm

def extract_videos(input_folder, output_folder):
    video_paths = sorted(glob.glob(os.path.join(input_folder, "*.avi")))
    
    for video_path in tqdm(video_paths, desc=f"Extragere {os.path.basename(input_folder)}"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        clean_name = video_name.replace("training_video_", "").replace("testing_video_", "")
        
        save_folder = os.path.join(output_folder, clean_name)
        os.makedirs(save_folder, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 1
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Avenue folosește 4 cifre
            cv2.imwrite(os.path.join(save_folder, f"{str(frame_count).zfill(4)}.png"), frame)
            frame_count += 1
        cap.release()

# --- SETEAZĂ RUTELE TALE AICI ---
DIR_DESCĂRCAT_TRAIN = "C:/Users/Agu/Desktop/Avenue_Dataset/training_videos"
DIR_DESCĂRCAT_TEST = "C:/Users/Agu/Desktop/Avenue_Dataset/testing_videos"

DIR_IESIRE_TRAIN = "../Model_Training/Avenue_Dataset/train/frames"
DIR_IESIRE_TEST = "../Model_Training/Avenue_Dataset/test/frames"

if __name__ == "__main__":
    extract_videos(DIR_DESCĂRCAT_TRAIN, DIR_IESIRE_TRAIN)
    extract_videos(DIR_DESCĂRCAT_TEST, DIR_IESIRE_TEST)