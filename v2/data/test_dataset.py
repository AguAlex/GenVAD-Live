import glob
import os
import cv2
import numpy as np
import torch.utils.data

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif"]

class AbnormalDatasetTest(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        
        if args.dataset == "avenue":
            data_path = args.avenue_path
        elif args.dataset == "custom":
            data_path = args.custom_path
        else:
            raise Exception("Unknown dataset!")

        self.data, self.labels, self.video_names = self._read_data(data_path)

    @staticmethod
    def _labels_from_avenue_official_mat(mat_path):
        """
        Demo oficial Avenue: testing_label_mask/{n}_label.mat, câmp 'volLabel' shape (1, T)
        cu celule-mască (H, W); cadru anormal dacă suma pixelilor > 0.
        """
        try:
            import scipy.io as sio
        except ImportError as e:
            raise ImportError(
                "Pentru fișiere .mat de la Avenue ai nevoie de scipy: pip install scipy"
            ) from e
        d = sio.loadmat(mat_path)
        if "volLabel" not in d:
            raise KeyError(f"Lipsește 'volLabel' în {mat_path}")
        vl = d["volLabel"]
        if vl.dtype != np.object_:
            out = np.asarray(vl, dtype=np.float64).reshape(-1)
            return out
        lbls = []
        for i in range(vl.size):
            m = np.asarray(vl.flat[i])
            lbls.append(1.0 if m.size and np.sum(m) > 0 else 0.0)
        return np.array(lbls, dtype=np.float64)

    @staticmethod
    def _load_frame_labels(data_path, video_name, num_frames):
        """
        Avenue (oficial .mat): testing_label_mask/{n}_label.mat (n = 1..21, fără zero),
        sau același subfolder în ground_truth_demo/ lângă rădăcina datasetului.
        Avenue (.txt): ground_truth/{video}.txt — o etichetă (0/1) pe linie.
        UCSD: test/{video}_gt/*.bmp — anomalie dacă există pixeli > 0.
        """
        try:
            vid_id = int(str(video_name))
        except ValueError:
            vid_id = None

        if vid_id is not None:
            mat_name = f"{vid_id}_label.mat"
            for sub in (
                os.path.join(data_path, "testing_label_mask", mat_name),
                os.path.join(data_path, "ground_truth_demo", "testing_label_mask", mat_name),
            ):
                if os.path.isfile(sub):
                    try:
                        lbls = AbnormalDatasetTest._labels_from_avenue_official_mat(sub)
                    except (OSError, KeyError, ImportError, ValueError) as e:
                        print(f"⚠️ Citire .mat eșuată {sub}: {e}. Încerc alte surse GT.")
                    else:
                        if lbls.size < num_frames:
                            lbls = np.concatenate(
                                [lbls, np.zeros(num_frames - lbls.size)]
                            )
                        elif lbls.size > num_frames:
                            lbls = lbls[:num_frames]
                        return lbls

        gt_txt = os.path.join(data_path, "ground_truth", f"{video_name}.txt")
        if os.path.isfile(gt_txt):
            try:
                raw = np.loadtxt(gt_txt)
            except OSError as e:
                print(f"⚠️ Nu pot citi GT txt {gt_txt}: {e}. Punem 0.")
                return np.zeros(num_frames)
            lbls = np.atleast_1d(np.squeeze(raw)).astype(np.float64)
            if lbls.size < num_frames:
                pad = np.zeros(num_frames - lbls.size)
                lbls = np.concatenate([lbls, pad])
            elif lbls.size > num_frames:
                lbls = lbls[:num_frames]
            return lbls

        gt_folder = os.path.join(data_path, "test", f"{video_name}_gt")
        if os.path.isdir(gt_folder):
            gt_imgs = sorted(list(glob.glob(os.path.join(gt_folder, "*.bmp"))))
            lbls = []
            for gt_img_path in gt_imgs:
                mask = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None and np.sum(mask) > 0:
                    lbls.append(1)
                else:
                    lbls.append(0)
            lbls = np.array(lbls, dtype=np.float64)
            if lbls.size < num_frames:
                pad = np.zeros(num_frames - lbls.size)
                lbls = np.concatenate([lbls, pad])
            elif lbls.size > num_frames:
                lbls = lbls[:num_frames]
            return lbls

        print(
            f"⚠️ Nu am găsit GT (.mat Avenue, txt sau folder _gt) pentru {video_name}. Punem 0."
        )
        return np.zeros(num_frames)

    def _read_data(self, data_path):
        data = []
        labels = []
        video_names = []
        
        extension = None
        for ext in IMG_EXTENSIONS:
            if len(list(glob.glob(os.path.join(data_path, "test/frames", f"*/*{ext}")))) > 0:
                extension = ext
                break
        self.extension = extension
                
        if extension is None:
            raise ValueError(f"Nu s-au găsit imagini în: {os.path.join(data_path, 'test/frames')}")

        dirs = sorted(list(glob.glob(os.path.join(data_path, "test", "frames", "*"))))
        
        for dir in dirs:
            imgs_path = sorted(list(glob.glob(os.path.join(dir, f"*{extension}"))))
            data += imgs_path
            video_name = os.path.basename(dir)

            lbls = self._load_frame_labels(data_path, video_name, len(imgs_path))
            
            for i, img_path in enumerate(imgs_path):
                labels.append(float(lbls[i]))
                video_names.append(video_name) 
            
        return data, labels, video_names

    def __getitem__(self, index):
        img = cv2.imread(self.data[index])
        if img is None:
            raise ValueError(f"Nu pot citi imaginea: {self.data[index]}")

        target_img = img.copy()

        if img.shape[:2] != self.args.input_size[::-1]:
            img = cv2.resize(img, self.args.input_size[::-1])
            target_img = cv2.resize(target_img, self.args.input_size[::-1])
            
        mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        target = np.concatenate((target_img, mask), axis=-1)
        
        img = img.astype(np.float32)
        target = target.astype(np.float32)
        
        img = (img - 127.5) / 127.5
        target = (target - 127.5) / 127.5
        
        img = np.swapaxes(img, 0, -1).swapaxes(1, -1)
        target = np.swapaxes(target, 0, -1).swapaxes(1, -1)
        
        return img, target, self.labels[index], self.video_names[index], self.data[index]

    def __len__(self):
        return len(self.data)