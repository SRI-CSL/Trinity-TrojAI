import os
import json
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
import torch.nn as nn
import copy
from argparse import ArgumentParser
from torch.optim import Adam
from torch.autograd import Variable
import skimage.io
from decimal import Decimal
from torch.utils.data import Dataset, DataLoader, TensorDataset
from matplotlib.pyplot import imshow
from PIL import Image as pil_image


def init_data_path(dataset_name, use_kubernetes=False):
    if not use_kubernetes:
        troj_root = "/data/ksikka/projects_2020/trojAI"
        cvpr_root = "/data/isur/0.Work/trojAI/trinityTrojAI/users/ksikka/checkpoints_path_bkp/neural_clense_masks"
    else:
        troj_root = "/data/datasets/"
        cvpr_root = "/code/neural_clense_masks"

    if dataset_name == "round2":
        model_dir = os.path.join(troj_root, "round2-dataset-train")
        meta_data = pd.read_csv(os.path.join(model_dir, "METADATA.csv"))
        feature_dir = os.path.join(cvpr_root, "round2")
    elif dataset_name == "round1":
        model_dir = os.path.join(troj_root, "round1-dataset-train", "models")
        meta_data = pd.read_csv(
            os.path.join(troj_root, "round1-dataset-train", "METADATA.csv")
        )
        feature_dir = os.path.join(cvpr_root, "round1")
        meta_data = meta_data.rename(columns={"ground_truth": "poisoned"})
    elif dataset_name == "round1_holdout":
        model_dir = os.path.join(cvpr_root, "round1_holdout")
        meta_data = pd.read_csv(
            os.path.join(troj_root, "round1-holdout-dataset/METADATA.csv")
        )
        meta_data = holdout_metadata.rename(columns={"ground_truth": "poisoned"})
        feature_dir = os.path.join(cvpr_root, "round1_holdout")
    elif dataset_name == "mnist":
        model_dir = os.path.join(troj_root, "mnist-dataset")
        meta_data = pd.read_csv(os.path.join(model_dir, "METADATA.csv"))
        feature_dir = os.path.join(cvpr_root, "mnist")

    return model_dir, meta_data, feature_dir


def get_image(img_path):
    img = skimage.io.imread(img_path)
    # perform center crop to what the CNN is expecting 224x224
    h, w, c = img.shape
    dx = int((w - 224) / 2)
    dy = int((w - 224) / 2)
    img = img[dy : dy + 224, dx : dx + 224, :]
    # img = img.astype('float32') + np.random.randn(224, 224, 3) * noise
    img = np.uint8(img)
    return img


def get_dl(file_list, batch_size=32):
    labels = torch.tensor([int(en.split("/")[-1].split("_")[1]) for en in file_list])
    img_batch_load = []
    for img in file_list:
        img = get_image(img)
        batch_data = torch.FloatTensor(img)
        img_batch_load.append(batch_data)
    img_batch = torch.stack(img_batch_load, 0).squeeze()
    dataset = TensorDataset(torch.Tensor(img_batch), labels)
    dl = DataLoader(dataset, batch_size=batch_size)

    return dl


class MNISTdataset(Dataset):
    def __init__(self, root_dir, filenames, csv_file):
        self.root_dir = root_dir

        df = pd.read_csv(csv_file, header=None)

        self.filenames = []
        self.truelabels = []
        for f in filenames:
            _f = f.split("/")[-1]
            label = df[df[0] == _f][1].to_list()
            if len(label):
                self.filenames.append(f)
                self.truelabels.append(label[0])

    def __len__(self):
        return len(self.truelabels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        label = self.truelabels[idx]
        image = self.pre_process_image(img_path)
        return image, label

    @staticmethod
    def pre_process_image(img_path):
        img = skimage.io.imread(img_path)
        # img = img[None, ...]/255.
        img = img[..., None]
        return img.astype("float32")


def get_dl_mnist(root_dir, filenames, csv_file, batch_size=32):
    dataset = MNISTdataset(root_dir, filenames, csv_file)
    dl = DataLoader(dataset, batch_size=batch_size)

    return dl


def save_mask_pattern(mask, pattern, target_class, save_dir, show=True):
    mask[mask < 0] = 0
    mask[mask > 1] = 1

    pattern[pattern < 0] = 0
    pattern[pattern > 255] = 255

    fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
    pattern = pattern.squeeze()
    fusion = fusion.squeeze()

    if show:
        f = plt.figure(figsize=(10, 30))
        f.add_subplot(1, 3, 1)
        plt.imshow(mask * 255)
        f.add_subplot(1, 3, 2)
        plt.imshow(pattern.astype("uint8"))
        f.add_subplot(1, 3, 3)
        plt.imshow(fusion.astype("uint8"))
        plt.show()

    if pattern.ndim == 3:
        COL_TXT = "RGB"
    elif pattern.ndim == 2:
        COL_TXT = "L"
    mask_img = pil_image.fromarray((mask * 255).astype("uint8"), "L")
    mask_img.save(os.path.join(save_dir, "mask_{}.png".format(target_class)))
    pattern_img = pil_image.fromarray(pattern.astype("uint8"), COL_TXT)
    pattern_img.save(os.path.join(save_dir, "pattern_{}.png".format(target_class)))
    fusion_img = pil_image.fromarray(fusion.astype("uint8"), COL_TXT)
    fusion_img.save(os.path.join(save_dir, "fusion_{}.png".format(target_class)))

