
import os
from torch.utils.data import Dataset
import cv2
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None, size=(512,512)):
        # if mask_dir is None, expect masks next to images with mask_ prefix
        self.img_paths = sorted([os.path.join(img_dir, p) for p in os.listdir(img_dir) if p.startswith('img_')])
        if mask_dir is None:
            self.mask_paths = [p.replace('img_', 'mask_') for p in self.img_paths]
        else:
            self.mask_paths = sorted([os.path.join(mask_dir, p) for p in os.listdir(mask_dir) if p.startswith('mask_')])
        self.transform = transform
        self.size = size

    def __len__(self):
        return min(len(self.img_paths), len(self.mask_paths))

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.size)
        mask = cv2.resize(mask, self.size)
        img = img.astype('float32') / 255.0
        mask = (mask>127).astype('float32')
        # HWC -> CHW
        img = np.transpose(img, (2,0,1))
        mask = np.expand_dims(mask, 0)
        return img, mask
