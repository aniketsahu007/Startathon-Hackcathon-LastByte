import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

        self.mapping = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            27: 4,
            39: 5
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)

        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # ---- AUGMENTATION ----
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # ---- Normalize ----
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))

        # ---- Remap mask ----
        new_mask = np.zeros_like(mask)
        for k, v in self.mapping.items():
            new_mask[mask == k] = v

        mask = new_mask

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
