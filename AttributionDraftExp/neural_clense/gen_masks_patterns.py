import os
import json
import numpy as np
import torch
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
import warnings

warnings.filterwarnings("ignore")

from utils import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

############################################################################
batch_size = 32

# input size  @Changes based on dataset
IMG_ROWS = 224
IMG_COLS = 224
IMG_COLOR = 3
NORMALIZE = True

INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

UPSAMPLE_SIZE = 1  # size of the super pixel
MASK_SHAPE = np.ceil(np.array(INPUT_SHAPE[0:2], dtype=float) / UPSAMPLE_SIZE)
MASK_SHAPE = MASK_SHAPE.astype(int)

# min/max of mask
MASK_MIN = 0
MASK_MAX = 1
# min/max of raw pixel intensity
COLOR_MIN = 0
COLOR_MAX = 255

EPSILON = 1e-07

# optimization
STEPS = 1000  # total optimization iterations
LR = 0.1
BETAS = (0.5, 0.9)
REGULARIZATION = "l1"  # reg term to control the mask's norm   # [None, 'l1', 'l2']

INIT_COST = 1e-3

VERBOSE = 2

ATTACK_SUCC_THRESHOLD = 0.99
PATIENCE = 5  # patience for adjusting weight, number of mini batches
COST_MULTIPLIER = 2  # multiplier for auto-control of weight (COST)
SAVE_LAST = False  # whether to save the last result or best result

EARLY_STOP = True  # whether to early stop
EARLY_STOP_THRESHOLD = 1.0  # loss threshold for early stop
EARLY_STOP_PATIENCE = 5 * PATIENCE  # patience for early stop

COST_MULTIPLIER_UP = 2
COST_MULTIPLIER_DOWN = COST_MULTIPLIER_UP ** 1.5


class hook_fn_nn:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs = module_out.squeeze()

    def clear(self):
        self.outputs = []


############################################################################
def generate_mask_pattern(model, dl, target_class):
    mask_tanh_t = Variable(
        torch.tensor(mask_tanh.copy()).to(device=device), requires_grad=True
    )
    pattern_tanh_t = Variable(
        torch.tensor(pattern_tanh.copy()).to(device=device), requires_grad=True
    )
    mask_upsapler = nn.Upsample(scale_factor=UPSAMPLE_SIZE, mode="nearest")

    # Define optimizer
    # if args.dataset == "mnist":
    #     criterion = nn.NLLLoss()
    # else:
    criterion = nn.CrossEntropyLoss()

    opt = Adam([mask_tanh_t, pattern_tanh_t], lr=LR, betas=BETAS)

    cost = INIT_COST

    # best optimization results
    mask_best = None
    mask_upsample_best = None
    pattern_best = None
    reg_best = float("inf")

    # logs and counters for adjusting balance cost
    logs = []
    cost_set_counter = 0
    cost_up_counter = 0
    cost_down_counter = 0
    cost_up_flag = False
    cost_down_flag = False

    # counter for early stop
    early_stop_counter = 0
    early_stop_reg_best = reg_best

    # loop start
    for step in range(STEPS):
        # record loss for all mini-batches
        loss_ce_list = []
        loss_reg_list = []
        loss_list = []
        loss_acc_list = []

        for img, _ in dl:
            # Forward
            label = torch.Tensor([target_class] * img.shape[0]).long().to(device=device)
            img = img.permute(0, 3, 1, 2).to(device=device)

            mask_t = torch.tanh(mask_tanh_t) / (2 - EPSILON) + 0.5
            mask_t = mask_t.repeat(1, 1, IMG_COLOR).unsqueeze(0)
            mask_t_t = mask_t.permute(0, 3, 1, 2)
            mask_t = mask_upsapler(mask_t_t)
            mask_t = mask_t[:, :, :IMG_ROWS, :IMG_COLS]
            rev_mask_t = 1 - mask_t

            pattern_t = (torch.tanh(pattern_tanh_t) / (2 - EPSILON) + 0.5) * 255.0
            pattern_t = pattern_t.unsqueeze(0)
            pattern_t = pattern_t.permute(0, 3, 1, 2)

            X_t = rev_mask_t * img + mask_t * pattern_t
            if NORMALIZE:
                X_t = X_t / 255.0

            if args.dataset == "mnist":
                _ = model(X_t.float())
                out = hook_fn_feat_layer.outputs
            else:
                out = model(X_t.float())
            loss_ce = criterion(out, label)

            if REGULARIZATION is None:
                loss_reg = 0
            elif REGULARIZATION is "l1":
                loss_reg = mask_t.abs().sum() / IMG_COLOR
            elif REGULARIZATION is "l2":
                loss_reg = torch.sqrt(torch.square(mask_t).sum()) / IMG_COLOR

            loss = loss_ce + cost * loss_reg
            loss_acc = (out.argmax(-1) == label).float().sum() / len(label)

            model.zero_grad()
            loss.backward()
            opt.step()

            loss_ce_list.append(loss_ce.item())
            loss_reg_list.append(loss_reg.item())
            loss_list.append(loss.item())
            loss_acc_list.append(loss_acc.item())

        avg_loss_ce = np.mean(loss_ce_list)
        avg_loss_reg = np.mean(loss_reg_list)
        avg_loss = np.mean(loss_list)
        avg_loss_acc = np.mean(loss_acc_list)

        # check to save best mask or not
        if avg_loss_acc >= ATTACK_SUCC_THRESHOLD and avg_loss_reg < reg_best:
            mask_best = mask_t_t[0, 0, ...].data.cpu().numpy()
            mask_upsample_best = mask_t[0, 0, ...].data.cpu().numpy()
            pattern_best = pattern_t.data.cpu().squeeze(0).permute(1, 2, 0).numpy()
            reg_best = avg_loss_reg

        _log_txt = (
            "step: %3d, cost: %.2E, attack: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f"
            % (
                step,
                Decimal(cost),
                avg_loss_acc,
                avg_loss,
                avg_loss_ce,
                avg_loss_reg,
                reg_best,
            )
        )
        # verbose
        if VERBOSE != 0:
            if VERBOSE == 2 or step % (STEPS // 10) == 0:
                print(_log_txt)

        # save log
        logs.append(_log_txt)

        # check early stop
        if EARLY_STOP:
            # only terminate if a valid attack has been found
            if reg_best < float("inf"):
                if reg_best >= EARLY_STOP_THRESHOLD * early_stop_reg_best:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
            early_stop_reg_best = min(reg_best, early_stop_reg_best)

            if (
                cost_down_flag
                and cost_up_flag
                and early_stop_counter >= EARLY_STOP_PATIENCE
            ):
                print("early stop")
                break

        # check cost modification
        if cost == 0 and avg_loss_acc >= ATTACK_SUCC_THRESHOLD:
            cost_set_counter += 1
            if cost_set_counter >= PATIENCE:
                cost = INIT_COST
                cost_up_counter = 0
                cost_down_counter = 0
                cost_up_flag = False
                cost_down_flag = False
                print("initialize cost to %.2E" % Decimal(self.cost))
        else:
            cost_set_counter = 0

        if avg_loss_acc >= ATTACK_SUCC_THRESHOLD:
            cost_up_counter += 1
            cost_down_counter = 0
        else:
            cost_up_counter = 0
            cost_down_counter += 1

        if cost_up_counter >= PATIENCE:
            cost_up_counter = 0
            if VERBOSE == 2:
                print(
                    "up cost from %.2E to %.2E"
                    % (Decimal(cost), Decimal(cost * COST_MULTIPLIER_UP))
                )
            cost *= COST_MULTIPLIER
            cost_up_flag = True
        elif cost_down_counter >= COST_MULTIPLIER_UP:
            cost_down_counter = 0
            if VERBOSE == 2:
                print(
                    "down cost from %.2E to %.2E"
                    % (Decimal(cost), Decimal(cost / COST_MULTIPLIER_DOWN))
                )
            cost /= COST_MULTIPLIER_DOWN
            cost_down_flag = True

    #         if self.save_tmp:
    #             self.save_tmp_func(step)

    # save the final version
    if mask_best is None:
        mask_best = mask_t_t[0, 0, ...].data.cpu().numpy()
        mask_upsample_best = mask_t[0, 0, ...].data.cpu().numpy()
        pattern_best = pattern_t.data.cpu().squeeze(0).permute(1, 2, 0).numpy()

    return pattern_best, mask_best, mask_upsample_best, logs


def outlier_detection(l1_norm_list, idx_mapping, consistency_constant=1.4826):
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad

    print("median: %f, MAD: %f" % (median, mad))
    print("anomaly index: %f" % min_mad)

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if np.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print(
        "flagged label list: %s"
        % ", ".join(["%d: %2f" % (y_label, l_norm) for y_label, l_norm in flag_list])
    )

    return {
        "l1_norm_lst": l1_norm_list,
        "flag_list": flag_list,
        "median": median,
        "mad": mad,
        "min_mad": min_mad,
    }


def analyze_pattern_norm_dist(save_dir, n_classes, consistency_constant=1.4826):
    mask_flatten = []
    idx_mapping = {}

    for y_label in range(n_classes):
        mask_filename = os.path.join(save_dir, "mask_{}.png".format(y_label))
        if os.path.exists(mask_filename):
            img = pil_image.open(mask_filename)
            # import ipdb; ipdb.set_trace()
            mask = np.array(img)
            mask = mask / 255
            # mask = mask[:, :, 0]

            mask_flatten.append(mask.flatten())

            idx_mapping[y_label] = len(mask_flatten) - 1

    l1_norm_list = [np.sum(np.abs(m)) for m in mask_flatten]
    print("%d labels found" % len(l1_norm_list))
    return outlier_detection(
        l1_norm_list, idx_mapping, consistency_constant=consistency_constant
    )


############################################################################
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="id-00000005")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["round1_holdout", "round1", "round2", "mnist"],
    )
    parser.add_argument("--use_kubernetes", action="store_true")
    parser.add_argument("--save_dir", type=str)

    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--attack_succ_thresh", type=float, default=0.9)
    parser.add_argument("--upsample", type=int, default=1)
    parser.add_argument("--init_cost", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    VERBOSE = args.verbose
    STEPS = args.steps
    ATTACK_SUCC_THRESHOLD = args.attack_succ_thresh
    UPSAMPLE_SIZE = args.upsample
    INIT_COST = args.init_cost
    PATIENCE = args.patience

    args.model_dir, args.meta_data, args.results_dir = init_data_path(
        args.dataset, use_kubernetes=args.use_kubernetes
    )
    if args.save_dir is not None:
        args.results_dir = args.save_dir

    if args.dataset == "mnist":
        IMG_ROWS = 28
        IMG_COLS = 28
        IMG_COLOR = 1
        NORMALIZE = False
        model_name = "model.pt.1"
        INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)
        MASK_SHAPE = np.ceil(np.array(INPUT_SHAPE[0:2], dtype=float) / UPSAMPLE_SIZE)
        MASK_SHAPE = MASK_SHAPE.astype(int)
    else:
        IMG_ROWS = 224
        IMG_COLS = 224
        IMG_COLOR = 3
        NORMALIZE = True
        model_name = "model.pt"
        INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)
        MASK_SHAPE = np.ceil(np.array(INPUT_SHAPE[0:2], dtype=float) / UPSAMPLE_SIZE)
        MASK_SHAPE = MASK_SHAPE.astype(int)

    save_dir = os.path.join(args.results_dir, args.model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, "result.json")
    if os.path.exists(save_file):
        exit()

    ############################################################################
    # Create Mask & Pattern:
    mask = np.random.random(MASK_SHAPE)
    pattern = np.random.random(INPUT_SHAPE) * 255.0

    mask = np.clip(mask, MASK_MIN, MASK_MAX)
    pattern = np.clip(pattern, COLOR_MIN, COLOR_MAX)
    mask = np.expand_dims(mask, axis=2)

    # convert to tanh space
    mask_tanh = np.arctanh((mask - 0.5) * (2 - EPSILON))
    pattern_tanh = np.arctanh((pattern / 255.0 - 0.5) * (2 - EPSILON))

    print("mask_tanh", np.min(mask_tanh), np.max(mask_tanh))
    print("pattern_tanh", np.min(pattern_tanh), np.max(pattern_tanh))
    ############################################################################

    meta_data = args.meta_data

    model_info = meta_data[meta_data.model_name == args.model_name].iloc[0]
    model_curr_dir = os.path.join(args.model_dir, model_info.model_name)

    # with open(os.path.join(model_curr_dir, "config.json"), "r") as f:
    #     model_config = json.load(f)

    model_filepath = os.path.join(model_curr_dir, model_name)
    model = torch.load(model_filepath, map_location=device)
    model = model.eval()

    if args.dataset == "mnist":
        hook_fn_feat_layer = hook_fn_nn()
        model.fc[-1].register_forward_hook(hook_fn_feat_layer)

        model_image_dir = os.path.join(args.model_dir, "example")
        file_list = [
            os.path.join(model_image_dir, en)
            for en in os.listdir(model_image_dir)
            if en.endswith(".png")
        ]
        csv_file = os.path.join(model_image_dir, "labels.txt")
        dl = get_dl_mnist(model_image_dir, file_list, csv_file, batch_size=batch_size)
    else:
        model_image_dir = os.path.join(model_curr_dir, "example_data")
        file_list = [
            os.path.join(model_image_dir, en)
            for en in os.listdir(model_image_dir)
            if en.endswith(".png")
        ]
        dl = get_dl(file_list, batch_size=batch_size)

    model_edit_dist_info = {}
    # model_edit_dist_info['trigger_type'] = model_info.trigger_type
    model_edit_dist_info["triggered_classes"] = model_info.triggered_classes
    model_edit_dist_info["trigger_target_class"] = model_info.trigger_target_class
    model_edit_dist_info["file_list"] = file_list

    print(model_info.poisoned)
    # print (model_info.trigger_type)
    print(model_info.triggered_classes)
    print(model_info.trigger_target_class)

    for target_class in range(model_info.number_classes):
        print("Target class: {}/{}".format(target_class, model_info.number_classes))
        log_file = os.path.join(save_dir, "log_{}.json".format(target_class))
        print(log_file)
        if not os.path.exists(log_file):
            pattern_best, mask_best, mask_up_best, logs = generate_mask_pattern(
                model, dl, target_class
            )
            save_mask_pattern(
                mask_up_best, pattern_best, target_class, save_dir, show=False
            )
            with open(log_file, "w") as f:
                f.write(json.dumps(logs, indent=4))

    ret = analyze_pattern_norm_dist(save_dir, model_info.number_classes)
    # print (ret)

    with open(save_file, "w") as wf:
        json.dump(ret, wf)

