import os
import glob
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def propagare_anomalie_statica(cale_cadre_normale, cale_cadru_ai, cale_masca, folder_iesire_cadre, folder_iesire_masti):
    """
    Propagă o anomalie generată pe un singur cadru peste tot restul videoclipului.
    Ideal pentru camere CCTV statice și obiecte nemișcate (persoane căzute, bagaje abandonate).
    """
    
    # 1. Verificăm dacă fișierele de bază există
    if not os.path.exists(cale_cadru_ai) or not os.path.exists(cale_masca):
        print("❌ Imaginea generată de AI sau masca lipsesc!")
        return

    print("🖼️ Se încarcă anomalia și masca de referință...")
    img_anomalie = Image.open(cale_cadru_ai).convert("RGBA")
    
    # Masca trebuie să fie în format "L" (Grayscale). 
    # Alb (255) = Arată anomalia, Negru (0) = Arată cadrul normal
    masca = Image.open(cale_masca).convert("L")

    # 2. Găsim toate cadrele normale din folderul videoclipului
    cadre_normale = sorted(glob.glob(os.path.join(cale_cadre_normale, "*.png")))
    if not cadre_normale:
        print(f"❌ Nu am găsit cadre normale în {cale_cadre_normale}")
        return

    print(f"🎬 Se propagă anomalia pe {len(cadre_normale)} cadre...")

    # Creăm folderele de ieșire dacă nu există
    os.makedirs(folder_iesire_cadre, exist_ok=True)
    os.makedirs(folder_iesire_masti, exist_ok=True)

    # 3. Bucla de propagare
    for cale_cadru in tqdm(cadre_normale, desc="Generare Cadre Anormale"):
        nume_fisier = os.path.basename(cale_cadru)
        
        # Încărcăm cadrul normal curent
        cadru_normal = Image.open(cale_cadru).convert("RGBA")
        
        # MAGIA ESTE AICI: Image.composite()
        # Unde masca e albă, ia pixelii din img_anomalie (omul căzut).
        # Unde masca e neagră, ia pixelii din cadru_normal (strada curată).
        # Unde masca e gri (marginile cu blur), face un blending perfect.
        cadru_anormal_final = Image.composite(img_anomalie, cadru_normal, masca)
        
        # Salvăm cadrul combinat (trecem înapoi în RGB)
        cale_salvare_cadru = os.path.join(folder_iesire_cadre, nume_fisier)
        cadru_anormal_final.convert("RGB").save(cale_salvare_cadru)
        
        # Salvăm și o copie a măștii pentru acest cadru (necesar pentru antrenament!)
        cale_salvare_masca = os.path.join(folder_iesire_masti, nume_fisier)
        masca.save(cale_salvare_masca)

    print(f"\n✅ Propagare finalizată! Verifică {folder_iesire_cadre}")


if __name__ == "__main__":
    # Numele videoclipului pe care lucrăm acum
    NUME_CLIP = "normal_clip_0010"
    
    # Rute de Intrare
    CADRE_NORMALE = f"../Model_Training/Custom_Dataset/train/frames/{NUME_CLIP}"
    CADRU_GENERAT_JUGGERNAUT = "/content/rezultat_final_persoana.png" # Calea către poza făcută cu scriptul tău SDXL
    MASCA_GENERATA_JUGGERNAUT = "/content/debug_mask_persoana.png"    # Masca ta cu margini soft (blur)
    
    # Rute de Ieșire
    CADRE_ANORMALE = f"../Model_Training/Custom_Dataset/train/frames_abnormal/{NUME_CLIP}"
    MASTI_ANORMALE = f"../Model_Training/Custom_Dataset/train/masks_abnormal/{NUME_CLIP}"
    
    propagare_anomalie_statica(
        cale_cadre_normale=CADRE_NORMALE,
        cale_cadru_ai=CADRU_GENERAT_JUGGERNAUT,
        cale_masca=MASCA_GENERATA_JUGGERNAUT,
        folder_iesire_cadre=CADRE_ANORMALE,
        folder_iesire_masti=MASTI_ANORMALE
    )