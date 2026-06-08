import scipy.io
import os
import numpy as np

def convert_mat_to_txt(mat_folder, txt_folder):
    os.makedirs(txt_folder, exist_ok=True)
    mat_files = [f for f in os.listdir(mat_folder) if f.endswith('.mat')]
    
    for mat_file in mat_files:
        mat_path = os.path.join(mat_folder, mat_file)
        mat_data = scipy.io.loadmat(mat_path)
        
        # Extragem array-ul de label-uri (depinde exact de cheia din .mat, de obicei e 'vol')
        # Această logică funcționează de obicei pentru Avenue
        try:
            vol = mat_data['vol']
            labels = np.any(vol, axis=(0, 1)).astype(int) # Transformăm 3D în 1D (0 sau 1 per frame)
            
            # Salvăm sub același nume, dar cu .txt
            txt_name = mat_file.replace('.mat', '.txt').replace('vol', '')
            np.savetxt(os.path.join(txt_folder, txt_name), labels, fmt='%d')
            print(f"✅ Convertit {mat_file} -> {txt_name}")
        except Exception as e:
            print(f"Eroare la {mat_file}: {e}")

# --- SETEAZĂ RUTELE ---
MAT_FOLDER = "C:/Users/Agu/Desktop/Avenue_Dataset/testing_vol"
TXT_FOLDER = "../Model_Training/Avenue_Dataset/ground_truth"

if __name__ == "__main__":
    convert_mat_to_txt(MAT_FOLDER, TXT_FOLDER)