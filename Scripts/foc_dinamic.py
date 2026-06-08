import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

def compozitare_foc_dinamic(cale_cadre_normale, cale_cadre_foc, folder_iesire_cadre, folder_iesire_masti):
    cadre_normale = sorted(glob.glob(os.path.join(cale_cadre_normale, "*.png")))
    cadre_foc = sorted(glob.glob(os.path.join(cale_cadre_foc, "*.png"))) # Extrase din clipul de pe net
    
    os.makedirs(folder_iesire_cadre, exist_ok=True)
    os.makedirs(folder_iesire_masti, exist_ok=True)

    # Ne asigurăm că avem destule cadre de foc (dacă nu, le luăm de la capăt)
    for i, cale_cadru in enumerate(tqdm(cadre_normale, desc="Foc Dinamic")):
        img_bg = cv2.imread(cale_cadru)
        
        # Luăm cadrul de foc corespunzător (cu loop dacă clipul de foc e mai scurt)
        img_foc = cv2.imread(cadre_foc[i % len(cadre_foc)])
        
        # Redimensionăm focul și îl punem unde vrem pe ecran (X, Y)
        # (Aici presupunem că l-ai pregătit deja la dimensiunea/poziția dorită, 
        # sau îl redimensionezi cu cv2.resize și îl pui peste un canvas negru de 1024x1024)
        img_foc = cv2.resize(img_foc, (img_bg.shape[1], img_bg.shape[0])) 

        # 1. GENERARE MASCĂ PENTRU ANTRENAMENT
        # Transformăm focul în grayscale. Orice e mai luminos de 20 (nu e negru) devine mască albă.
        gray_foc = cv2.cvtColor(img_foc, cv2.COLOR_BGR2GRAY)
        _, masca = cv2.threshold(gray_foc, 20, 255, cv2.THRESH_BINARY)
        
        # 2. COMPOZITARE (ADDITIVE BLENDING)
        # Adunăm valorile pixelilor. Focul luminează fundalul.
        cadru_anormal = cv2.add(img_bg, img_foc)

        # Salvare
        nume_fisier = os.path.basename(cale_cadru)
        cv2.imwrite(os.path.join(folder_iesire_cadre, nume_fisier), cadru_anormal)
        cv2.imwrite(os.path.join(folder_iesire_masti, nume_fisier), masca)

# Utilizează acest script pe folderele tale