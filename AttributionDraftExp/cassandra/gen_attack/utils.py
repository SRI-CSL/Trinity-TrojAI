import os
import numpy as np
import skimage.io
import random
import torch
import warnings
import csv
import torchvision
import pandas as pd
import torch.nn as nn
from argparse import ArgumentParser
from tqdm import tqdm
import sys

import dataloader
#################################################
def check_for_cuda():
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"Device is {device}")
    return device


#################################################
def load_model_round1(src_dir, model_name, device, meta_data):
    """
    Loading a specific model and meta-data around it from trojAI-round1 data
    src_dir: dir where round-1 data is available  
    """
    model_info = {}
    model_dir = os.path.join(src_dir, "models")
    model_info["model_dir"] = os.path.join(model_dir, model_name)
    model_info["image_dir"] = os.path.join(model_info["model_dir"], "example_data")
    model_info["csv_file"] = os.path.join(model_info["image_dir"], "data.csv")

    meta_data_row=(meta_data.loc[ meta_data["model_name"] == model_name]).iloc[0]

    model_info["model_img_size"]=meta_data_row["img_size"]
    model_info["num_classes"] = meta_data_row["number_classes"]
    print(f'Number of classes:{model_info["num_classes"]}')

    fs = []
    for (dirpath, dirnames, filenames) in os.walk(model_info["image_dir"]):
        fs.extend(filenames)
        break


    model_info["file_list"] = fs

    #model_info["file_list"] = [
    #            it.strip().split(",")[0:2]
    #            for it in open(os.path.join(model_info["image_dir"], "data.csv"))
    #        ][1:]
    model_info["model_filepath"] = os.path.join(model_info["model_dir"], "model.pt")
    model_info["label"] = meta_data_row["ground_truth"]

    # print(f"Number of benign images in {model_name} are {len(model_info["file_list"])}")
    model = torch.load(model_info["model_filepath"], map_location=device)
    model.eval()
    print(f"Loaded model {model_name} with ground-truth {model_info['label']}")

    dataset = dataloader.TrojAIDatasetRound1(model_info["image_dir"], model_info["file_list"])

    return model_info, model, dataset

#################################################
def load_model_round2(src_dir, model_name, device, meta_data):
    """
    Loading a specific model and meta-data around it from trojAI-round2 data
    src_dir: dir where round-1 data is available  
    """
    model_info = {}
    # model_dir = src_dir
    model_dir = os.path.join(src_dir, "models")
    model_info["model_dir"] = os.path.join(model_dir, model_name)
    model_info["image_dir"] = os.path.join(model_info["model_dir"], "example_data")

    meta_data_row=(meta_data.loc[ meta_data["model_name"] == model_name]).iloc[0]
    model_info["model_img_size"]=meta_data_row["cnn_img_size_pixels"]
    model_info["num_classes"] = meta_data_row["number_classes"]
    print(f'Number of classes:{model_info["num_classes"]}')

    fs = []
    for (dirpath, dirnames, filenames) in os.walk(model_info["image_dir"]):
        fs.extend(filenames)
        break
            
    model_info["file_list"] = fs
    
    model_info["model_filepath"] = os.path.join(model_info["model_dir"], "model.pt")
    model_info["label"] = meta_data_row["poisoned"]
    print(f"Poisoned:{meta_data_row['poisoned']}")
    
    
    model = torch.load(model_info["model_filepath"], map_location=device)
    model.eval()
    print(f"Loaded model {model_name} with ground-truth {model_info['label']}")

    dataset = dataloader.TrojAIDatasetRound2(model_info["image_dir"],model_info["file_list"])
    
    return model_info, model, dataset

######################################################
def argparser():
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="attack")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="id-00000020")
    # For attacked targets
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument(
        "--attack_type",
        type=str,
        default="pgd_t",
        choices=["pgd_t", "univ_pert_ut_1", "univ_pert_ut_2", "univ_pert_ut_n", "fgsm_ut"],
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default="/data/ksikka/projects_2020/trojAI/round1-dataset-train",
    )

    args = parser.parse_args()
    args.device = check_for_cuda()
    return args


def compute_accuracy(model, dataloader, device):

    with torch.no_grad():
        scores = []
        labels = []
        img_path = []

        for data in tqdm(dataloader):
            img, label, _img_path = data
            img = img.to(device)
            logits = model(img)
            labels.append(label.cpu().numpy())
            scores.append(logits.cpu().numpy())
            img_path.append(_img_path)

    scores = np.vstack(scores)

    labels = np.hstack(labels)
    
    img_path = np.hstack(img_path).tolist()
    pred_labels = scores.argmax(1)

    #s_indexes = sorted(range(len(img_path)), key=lambda k: img_path[k])
    #[print(img_path[i]) for i in s_indexes]
    #print(labels[s_indexes])
    #print(scores[s_indexes])
    #print(pred_labels[s_indexes])
    
    accuracy = (pred_labels == labels).mean() * 100

    print(f"Accuracy is {accuracy}")
    return img_path, pred_labels.tolist(), accuracy


def get_last_layer(dataloader, mlmodel, args):

    features = []
    labels = []

    def hook(mlmodel, input, output):
        features.append(
            input[0]
        )  # input is a tuple, get the first and only element for fc input
        # print(features)
        return hook

    moduleList = [x for x, y in mlmodel.named_modules()]
    lastlayermodule = moduleList[-1]
    with torch.no_grad():  # deactivate gradients
        mlmodel.eval()  # let batchnorm / dropout layers work in eval mode
        if lastlayermodule == "fc":
            handle = mlmodel.fc.register_forward_hook(hook)
        elif lastlayermodule == "classifier":
            handle = mlmodel.classifier.register_forward_hook(hook)
        else:
            print("Unknown last layer " + lastlayermodule + " in getNamedLayer.")
            sys.exit(0)
        # print('Number of samples: ', len(dataloader))


        for data in dataloader:
            sample, label, img_path = data
            sample = sample.to(args.device)
            output = mlmodel(sample)
            labels.append(label)

        handle.remove()
    features = torch.cat(features)
    #labels = torch.cat(labels)

    # print(features)
    return features

