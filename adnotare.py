import glob, os
import numpy as np

frames_dir = "Dataset/test/frames/Test002"
out_txt = "Dataset/ground_truth/Test002.txt"

n = len(glob.glob(os.path.join(frames_dir, "*.png")))
labels = np.zeros(n, dtype=int)

labels[643:660] = 1

os.makedirs(os.path.dirname(out_txt), exist_ok=True)
np.savetxt(out_txt, labels, fmt="%d")
print(f"Salvat {out_txt}: {n} linii, {labels.sum()} anormale")