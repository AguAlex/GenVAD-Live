import glob
import os
import cv2
import numpy as np
import torch.utils.data

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif"]

class AbnormalDatasetGradientsTest(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.input_3d = args.input_3d # Trebuie definit și aici!
        
        if args.dataset == "avenue":
            data_path = args.avenue_path
        elif args.dataset == "shanghai":
            data_path = args.shanghai_path
        elif args.dataset == "custom":
            data_path = args.custom_path
        else:
            raise Exception("Unknown dataset!")

        self.data, self.gradients, self.labels, self.video_names = self._read_data(data_path)

    @staticmethod
    def _labels_ucsd_bmps(gt_folder, num_frames):
        gt_imgs = sorted(glob.glob(os.path.join(gt_folder, "*.bmp")))
        lbls = []
        for gt_img_path in gt_imgs:
            mask = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and np.sum(mask) > 0:
                lbls.append(1.0)
            else:
                lbls.append(0.0)
        lbls = np.array(lbls, dtype=np.float64)
        if lbls.size < num_frames:
            lbls = np.concatenate([lbls, np.zeros(num_frames - lbls.size)])
        elif lbls.size > num_frames:
            lbls = lbls[:num_frames]
        return lbls

    @staticmethod
    def _labels_avenue_mat(mat_path, num_frames):
        try:
            import scipy.io as sio
        except ImportError as e:
            raise ImportError(
                "Pentru Avenue ground truth .mat trebuie scipy instalat."
            ) from e

        d = sio.loadmat(mat_path)
        if "volLabel" not in d:
            raise KeyError(f"Lipsește cheia 'volLabel' în {mat_path}")

        vl = d["volLabel"]
        # Format oficial Avenue: object array (1, T), fiecare element este o mască HxW.
        if vl.dtype == np.object_:
            lbls = []
            for i in range(vl.size):
                mask = np.asarray(vl.flat[i])
                lbls.append(1.0 if mask.size and np.sum(mask) > 0 else 0.0)
            lbls = np.array(lbls, dtype=np.float64)
        else:
            # Fallback pentru fișiere deja convertite în etichete per-cadru.
            lbls = np.asarray(vl, dtype=np.float64).reshape(-1)

        if lbls.size < num_frames:
            lbls = np.concatenate([lbls, np.zeros(num_frames - lbls.size)])
        elif lbls.size > num_frames:
            lbls = lbls[:num_frames]
        return lbls

    def _labels_for_sequence(self, data_path, video_name, num_frames):
        gt_folder_ucsd = os.path.join(data_path, "test", f"{video_name}_gt")
        if os.path.isdir(gt_folder_ucsd):
            bmp_paths = glob.glob(os.path.join(gt_folder_ucsd, "*.bmp"))
            if len(bmp_paths) > 0:
                return self._labels_ucsd_bmps(gt_folder_ucsd, num_frames)

        # Avenue oficial: ground_truth/{n}_label.mat, unde video_name poate fi "01".
        try:
            seq_id = int(str(video_name))
        except ValueError:
            seq_id = None
        if seq_id is not None:
            mat_path = os.path.join(self.args.avenue_gt_path, f"{seq_id}_label.mat")
            if os.path.isfile(mat_path):
                try:
                    return self._labels_avenue_mat(mat_path, num_frames)
                except (OSError, KeyError, ValueError, ImportError) as e:
                    print(f"⚠️ Nu pot citi GT mat {mat_path}: {e}. Încerc alte surse GT.")

        lbl_file = os.path.join(self.args.avenue_gt_path, f"{video_name}.txt")
        if os.path.isfile(lbl_file):
            try:
                raw = np.loadtxt(lbl_file)
            except OSError as e:
                print(f"⚠️ Nu pot citi GT txt {lbl_file}: {e}. Punem 0.")
                return np.zeros(num_frames)
            lbls = np.atleast_1d(np.squeeze(raw)).astype(np.float64)
            if lbls.size < num_frames:
                lbls = np.concatenate([lbls, np.zeros(num_frames - lbls.size)])
            elif lbls.size > num_frames:
                lbls = lbls[:num_frames]
            return lbls

        print(
            f"⚠️ Nu am găsit GT (folder {video_name}_gt, {video_name}.txt sau {video_name}_label.mat). Punem 0."
        )
        return np.zeros(num_frames)

    def _read_data(self, data_path):
        data = []
        gradients = []
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
            gradients_path = []

            lbls = self._labels_for_sequence(data_path, video_name, len(imgs_path))

            for i, img_path in enumerate(imgs_path):
                img_name = os.path.basename(img_path)
                gradients_path.append(os.path.join(data_path, "test", "gradients2", video_name, img_name))

                labels.append(float(lbls[i]))
                video_names.append(video_name) 
                
            gradients += gradients_path
            
        return data, gradients, labels, video_names

    def __getitem__(self, index):
        current_img = cv2.imread(self.data[index])
        
        # ⬅️ Verificare de siguranță adăugată:
        if current_img is None:
            raise ValueError(f"Nu pot citi imaginea: {self.data[index]}")

        dir_path, frame_no, len_frame_no = self.extract_meta_info(self.data, index)
        previous_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=-3, length=len_frame_no)
        next_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=3, length=len_frame_no)
        img = current_img
        
        if self.input_3d:
            img = np.concatenate([previous_img, current_img, next_img], axis=-1)

        gradient = cv2.imread(self.gradients[index])
        if gradient is None:
            raise ValueError(f"Nu pot citi gradientul: {self.gradients[index]}")

        if img.shape[:2] != self.args.input_size[::-1]:
            img = cv2.resize(img, self.args.input_size[::-1])
            current_img = cv2.resize(current_img, self.args.input_size[::-1])
            gradient = cv2.resize(gradient, self.args.input_size[::-1])
            
        mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        target = np.concatenate((current_img, mask), axis=-1)
        
        img = img.astype(np.float32)
        gradient = gradient.astype(np.float32)
        target = target.astype(np.float32)
        
        img = (img - 127.5) / 127.5
        target = (target - 127.5) / 127.5
        
        img = np.swapaxes(img, 0, -1).swapaxes(1, -1)
        target = np.swapaxes(target, 0, -1).swapaxes(1, -1)
        gradient = np.swapaxes(gradient, 0, 1).swapaxes(0, -1)
        
        return img, gradient, target, self.labels[index], self.video_names[index], self.data[index]

    def extract_meta_info(self, data, index):
        frame_no = int(data[index].split("/")[-1].split('.')[0])
        dir_path = "/".join(data[index].split("/")[:-1])
        len_frame_no = len(data[index].split("/")[-1].split('.')[0])
        return dir_path, frame_no, len_frame_no

    def read_prev_next_frame_if_exists(self, dir_path, frame_no, direction=-3, length=1):
        frame_path = dir_path + "/" + str(frame_no + direction).zfill(length) + self.extension
        if os.path.exists(frame_path):
            return cv2.imread(frame_path)
        else:
            return cv2.imread(dir_path + "/" + str(frame_no).zfill(length) + self.extension)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.__class__.__name__