import cv2
import os
import time
import re  # <-- Adăugat pentru a citi numerele din numele fișierelor
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURĂRI ---
USER = os.getenv("CAM_USER")
PASS = os.getenv("CAM_PASS")
IP = os.getenv("CAM_IP")
STREAM = os.getenv("CAM_STREAM", "stream1")

URL = f"rtsp://{USER}:{PASS}@{IP}:554/{STREAM}"

CLIP_DURATION = 15  # Durata în secunde pentru fiecare clip
OUTPUT_DIR = "dataset/train_normal"

# Creăm folderul dacă nu există
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print(f"Încercare de conectare la {IP}...")
    cap = cv2.VideoCapture(URL)

    if not cap.isOpened():
        print("Eroare: Nu s-a putut deschide fluxul video.")
        return

    # Extragem proprietățile camerei
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Uneori stream-urile RTSP nu raportează corect FPS-ul. 
    # Camerele Tapo rulează de obicei la 15 FPS pe stream1.
    if fps == 0 or fps != fps: 
        fps = 15.0

    print(f"Conectat cu succes! Rezoluție: {width}x{height} @ {fps} FPS")
    
    # --- LOGICĂ NOUĂ PENTRU CONTINUAREA NUMEROTĂRII ---
    # Căutăm clipurile deja existente în folder
    existing_files = [f for f in os.listdir(OUTPUT_DIR) if re.match(r'normal_clip_\d+\.mp4', f)]
    
    if existing_files:
        # Extragem numerele din nume și găsim maximul
        indices = [int(re.findall(r'\d+', f)[0]) for f in existing_files]
        clip_index = max(indices) + 1
        print(f"S-au găsit {len(existing_files)} clipuri vechi. Reluăm înregistrarea de la indexul: {clip_index:04d}")
    else:
        clip_index = 0
        print("Nu s-au găsit clipuri vechi. Începem de la indexul: 0000")
    # ---------------------------------------------------

    print(f"Salvare clipuri de {CLIP_DURATION} secunde în folderul '{OUTPUT_DIR}'...")
    print("Apasă 'q' în fereastra video pentru a opri înregistrarea.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frames_per_clip = int(fps * CLIP_DURATION)
    
    frame_count = 0
    out = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Eroare: S-a pierdut conexiunea cu camera.")
                break

            # Dacă am atins numărul de cadre pentru un clip, deschidem un fișier nou
            if frame_count % frames_per_clip == 0:
                if out is not None:
                    out.release() # Salvăm clipul anterior
                
                filename = os.path.join(OUTPUT_DIR, f"normal_clip_{clip_index:04d}.mp4")
                out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                print(f"Se înregistrează: {filename}")
                clip_index += 1

            # Scriem cadrul în fișierul video curent
            out.write(frame)
            frame_count += 1

            # Afișăm și fluxul live, ca să vezi ce se înregistrează
            cv2.imshow("Inregistrare Faza 1 (Apasa Q pt iesire)", frame)

            # Ieșire curată
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Înregistrare oprită de utilizator.")
                break

    finally:
        # Ne asigurăm că eliberăm resursele la final, chiar dacă apare o eroare
        if out is not None:
            out.release()
        cap.release()
        cv2.destroyAllWindows()
        print("Toate resursele au fost eliberate cu succes.")

if __name__ == "__main__":
    main()