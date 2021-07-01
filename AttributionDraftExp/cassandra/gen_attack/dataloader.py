import pandas as pd
import skimage.io
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re

class TrojAIDataset(Dataset):
    def __init__(self, root_dir, filenames):
        """
        Args:
            root_dir (string): Directory with all the images.
            filenames
        """
        self.root_dir = root_dir
        self.filenames = filenames

        p = re.compile("class_(\d*)_.*")

        self.filenames = [f for f in self.filenames if p.search(f)]

        #the pain of having to specify this
        self.truelabels = [int(p.search(f).group(1)) for f in self.filenames]
        
    def __len__(self):
        return len(self.truelabels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        label = self.truelabels[idx]
        image = self.pre_process_image(img_path)
        return image, label, img_path

    @staticmethod
    def pre_process_image(img_path):
        img = skimage.io.imread(img_path)
        #only want to normalize 0-1, do not wants to rebias

        img = img/ np.max(img)
        img = np.transpose(img, (2, 0, 1))
        
        return img.astype("float32")



class TrojAIDatasetNumpy(TrojAIDataset):
    def __init__(self, root_dir, filenames):
        super().__init__(root_dir, filenames)

    @staticmethod
    def pre_process_image(img_path):
        filepath, ext = os.path.splitext(img_path)
        img = np.load(filepath+'.npy')
        return img


class TrojAIDatasetRound1(TrojAIDataset):
    def __init__(self, root_dir, filenames):
        super().__init__(root_dir, filenames)

    @staticmethod
    def pre_process_image(img_path):
        img = skimage.io.imread(img_path)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        image = np.stack((b, g, r), axis=2)
        image = np.transpose(image, (2, 0, 1))
        image = image - np.min(image)
        image = image/ np.max(image)
        return image



class TrojAIDatasetRound2(TrojAIDataset):
    def __init__(self, root_dir, filenames):
        super().__init__(root_dir, filenames)

    @staticmethod
    def pre_process_image(img_path):
        img = skimage.io.imread(img_path)

        # perform center crop to what the CNN is expecting 224x224
        h, w, c = img.shape
        dx = int((w - 224) / 2)
        dy = int((w - 224) / 2)
        img = img[dy:dy+224, dx:dx+224, :]

        # perform tensor formatting and normalization explicitly
        # convert to CHW dimension ordering
        img = np.transpose(img, (2, 0, 1))
        # convert to NCHW dimension ordering

        # normalize the image
        img = img - np.min(img)
        img = img / np.max(img)

        return img.astype("float32")
