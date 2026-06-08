import glob
import os
import random
import cv2
import numpy as np
import torch.utils.data

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif"]

class AbnormalDatasetTrain(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        if args.dataset == "avenue":
            data_path = args.avenue_path
        elif args.dataset == "custom":
            data_path = args.custom_path
        else:
            raise Exception("Unknown dataset!")
            
        self.percent_abnormal = args.percent_abnormal
        self.abnormal_data, self.data, self.masks_abnormal = self._read_data(data_path)
        self.valid_abnormal_indices = self._index_valid_abnormal_pairs()
        self.valid_abnormal_set = set(self.valid_abnormal_indices)

        if self.percent_abnormal > 0:
            n = len(self.valid_abnormal_indices)
            total = len(self.data)
            print(
                f"[train] Perechi anormale valide: {n}/{total} "
                f"({100.0 * n / total:.1f}% din cadre normale)."
            )
            if n == 0:
                print(
                    "[train] ⚠️  Niciun frames_abnormal/masks_abnormal găsit — "
                    "antrenament doar pe cadre normale (percent_abnormal ignorat)."
                )

    @staticmethod
    def _paths_exist(frame_path: str, mask_path: str) -> bool:
        return os.path.isfile(frame_path) and os.path.isfile(mask_path)

    def _index_valid_abnormal_pairs(self) -> list[int]:
        """Indici unde există atât cadrul anormal cât și masca (generare reușită)."""
        valid: list[int] = []
        for i, (frame_path, mask_path) in enumerate(
            zip(self.abnormal_data, self.masks_abnormal)
        ):
            if self._paths_exist(frame_path, mask_path):
                valid.append(i)
        return valid

    def _read_data(self, data_path):
        data = []
        abnormal_data = []
        masks_abnormal = []
        extension = None
        
        for ext in IMG_EXTENSIONS:
            if len(list(glob.glob(os.path.join(data_path, "train/frames", f"*/*{ext}")))) > 0:
                extension = ext
                break
        self.extension = extension
        
        dirs = list(glob.glob(os.path.join(data_path, "train", "frames", "*")))
        for dir in dirs:
            imgs_path = list(glob.glob(os.path.join(dir, f"*{extension}")))
            data += imgs_path
            video_name = os.path.basename(dir)
            
            for img_path in imgs_path:
                img_name = os.path.basename(img_path)
                abnormal_data.append(os.path.join(data_path, "train", "frames_abnormal", video_name, img_name))
                masks_abnormal.append(os.path.join(data_path, "train", "masks_abnormal", video_name, img_name))
                
        return abnormal_data, data, masks_abnormal

    def __getitem__(self, index):
        use_abnormal = (
            self.percent_abnormal > 0
            and self.valid_abnormal_indices
            and random.uniform(0, 1) <= self.percent_abnormal
        )

        if use_abnormal:
            # Același index dacă există perechea; altfel un cadru anormal valid aleator
            ab_idx = (
                index
                if index in self.valid_abnormal_set
                else random.choice(self.valid_abnormal_indices)
            )
            img = cv2.imread(self.abnormal_data[ab_idx])
            mask = cv2.imread(self.masks_abnormal[ab_idx])
            target = cv2.imread(self.data[ab_idx])
            if img is None:
                # Fișier șters/corupt între init și load — revenim la normal
                img = cv2.imread(self.data[index])
                mask = None
                target = cv2.imread(self.data[index])
                ab_idx = index
        else:
            ab_idx = index
            img = cv2.imread(self.data[index])
            mask = None
            target = cv2.imread(self.data[index])

        if img is None or target is None:
            raise FileNotFoundError(
                f"Nu pot citi cadrul normal: {self.data[index]}"
            )

        if mask is not None:
            mask = mask[:, :, :1]
        else:
            mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

        if img.shape[:2] != self.args.input_size[::-1]:
            img = cv2.resize(img, self.args.input_size[::-1])
            mask = cv2.resize(mask, self.args.input_size[::-1])
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=-1)
            target = cv2.resize(target, self.args.input_size[::-1])

        # Concatenăm masca la target (3 channels img + 1 channel mask)
        target = np.concatenate((target, mask), axis=-1)
        
        img = img.astype(np.float32)
        target = target.astype(np.float32)
        
        img = (img - 127.5) / 127.5
        img = np.swapaxes(img, 0, -1).swapaxes(1, -1)
        
        target = (target - 127.5) / 127.5
        target = np.swapaxes(target, 0, -1).swapaxes(1, -1)
        
        # Returnăm DOAR imaginea antrenată și imaginea țintă curată (target)
        return img, target

    def __len__(self):
        return len(self.data)