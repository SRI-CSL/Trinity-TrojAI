import os
import pickle
import pandas as pd
import numpy as np
import scipy.interpolate as interpolate
from sklearn import metrics
import random
from sklearn import metrics
import tempfile
import pprint
import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append("../troj_round2")
sys.path.append("../troj_round2/utils_temporal_nn")

from utils_nn import (
    # load_data,
    # neuron_ds,
    # TrojNet,
    # get_members_list,
    # train_model,
    get_splits,
    get_splits_random,
    get_split_data,
    # initialize_wandb,
    # init_data_path,
    get_data_info,
)


from wrapper_nn import check_for_cuda
import argparse
from helper import compute_metrics, sigmoid

from util_cassandra import cassandra_ds, CassendraNet, init_data_path_cassendra, learner_cassendra, initialize_wandb


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_split", type=int, default=20)
    parser.add_argument("--val_split", type=int, default=20)
    # parser.add_argument(
    #     "--arch",
    #     type=str,
    #     default="conv",
    #     choices=["conv", "conv_posenc", "transformer"],
    # )
    parser.add_argument("--env_name", type=str, default="cassendra")
    parser.add_argument(
        "--dataset", type=str, default="mnist", choices=["round1", "round2", "mnist", "round3"]
    )
    parser.add_argument("--lr", type=float, default="1e-3")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    # parser.add_argument("--max_class", type=int, default=24)
    # parser.add_argument("--diff", action="store_true")
    # parser.add_argument("--p", type=float, default=0.5)
    # parser.add_argument("--p_tx", type=float, default=0.8)
    # parser.add_argument("--nhead", type=int, default=2)
    # parser.add_argument("--nhid", type=int, default=16)
    # parser.add_argument("--nlayers_tx", type=int, default=2)
    # parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--num_ensemble", type=int, default=5)
    parser.add_argument("--ckt_path", type=str)
    parser.add_argument("--logdir", type=str, default="/home/isur/troj/log")
    # parser.add_argument("--stocastic", action="store_true")
    # parser.add_argument("--T", type=int, default=10)
    # parser.add_argument("--hard", action="store_true")
    parser.add_argument("--nfolds", type=int, default=5)
    parser.add_argument("--dump_output", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_holdout", action="store_true")
    parser.add_argument("--use_kubernetes", action="store_true")
    parser.add_argument("--output_file", type=str, default="test.txt")
    parser.add_argument("--tags", type=str, default=" ")
    # parser.add_argument("--model_dir", type=str)
    # parser.add_argument("--meta_data", type=str)
    # parser.add_argument("--feature_dir", type=str)

    # Cassendra opts
    parser.add_argument("--use_fr", action="store_true")
    parser.add_argument('--feat_types', nargs='+', type=str, default=['univ_pert_ut_l1'])
    parser.add_argument("--crop_stride", type=int, default=2)
    parser.add_argument("--crop_size", type=int, default=50)
    parser.add_argument("--scale_mnist", action="store_true")
    parser.add_argument("--use_blank_noise", action="store_true")

    args = parser.parse_args()

    args.device = check_for_cuda()

    # Initialize paths
    model_dir, meta_data, feature_dir, holdout_dir, holdout_metadata = init_data_path_cassendra(
        args.dataset, use_holdout=args.use_holdout, use_kubernetes=args.use_kubernetes
    )

    if holdout_dir is not None:
        # For round-1 20% is kept for validation
        args.test_split = 0
        args.val_split = 20

    nfolds = args.nfolds
    if args.ckt_path is None:
        _, args.ckt_path = tempfile.mkstemp()

    if args.dump_output:
        sys.stdout = open(args.output_file, "w")

    now = datetime.datetime.now()
    args.env_name = args.env_name + "_" + now.strftime("%Y-%m-%d%H:%M:%S")

    if args.use_wandb:
        wandb = initialize_wandb(args)
        writer = SummaryWriter(os.path.join(args.logdir, args.env_name))

    pprint.pprint(vars(args))
    sys.stdout.flush()
    print("Current date and time : ")
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Holdout dir is {holdout_dir}")

    #############################################################################
    # Load data from all the models
    all_archs, models_filter, models_lst = get_data_info(args, meta_data)
    # all_data = load_data(feature_dir, meta_data, models_lst)
    if holdout_dir is not None:
        _, models_filter_holdout, models_lst_holdout = get_data_info(
            args, holdout_metadata
        )
        # holdout_data = load_data(holdout_dir, holdout_metadata, models_lst_holdout)

    print("Dataset: ", args.dataset)
    print("Running experiments on archs: ", all_archs)

    #############################################################################
    # Create splits
    partitions = []
    for _ in range(nfolds):
        partitions.append(
            get_splits_random(models_filter, all_archs, args.test_split, args.val_split)
        )

    #############################################################################
    # Setup Cross-validation and model
    np.random.seed(0)
    auc_all = []
    scores_all = {}


    model = learner_cassendra(args)

    auc_all = []
    acc_all = []
    ce_all = []
    val_auc_all = []
    val_acc_all = []
    val_ce_all = []
    for i in range(nfolds):
        train_split, val_split, test_split = partitions[i]
        print(
            "Partition sizes: Train-{}, Val-{}, Test-{}".format(
                len(train_split), len(val_split), len(test_split)
            )
        )

        ds_train = cassandra_ds(feature_dir,
                                meta_data,
                                models_list=train_split,
                                args=args)
        ds_val = cassandra_ds(feature_dir,
                                meta_data,
                                models_list=val_split,
                                args=args)
        if holdout_dir is not None:
            ds_test = cassandra_ds(holdout_dir,
                                    holdout_metadata,
                                    models_list=models_lst_holdout,
                                    args=args)
        else:
            ds_test = cassandra_ds(feature_dir,
                                    meta_data,
                                    models_list=test_split,
                                    args=args)

        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
        dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

        train_loader = dl_train
        valid_loader = dl_test

        # Building ensemble (average)
        test_info = []
        for num_models in range(args.num_ensemble):
            print(f"Training ensemble model {num_models} / {args.num_ensemble}")
            valid_acc, valid_auc, valid_ce, valid_info = model.train(dl_train, dl_val)
            acc, auc, ce, _test_info = model.test(dl_test)
            test_info.append(_test_info)
            print(f"validation auc = {valid_auc}\ntest auc = {auc}")
        prob = np.asarray([sigmoid(it[0]) for it in test_info]).mean(0)
        # Convert back to logits
        scores = np.log(prob / (1 - prob + 1e-10))
        labels = test_info[0][1]
        auc, acc, ce = compute_metrics(scores, labels)
        print(f"Fold {i}\nAcc: {acc:.2f}\nAuc: {auc:.2f}\nCE: {ce:.2f}")

        val_auc_all.append(valid_auc)
        val_acc_all.append(valid_acc)
        val_ce_all.append(valid_ce)
        auc_all.append(auc)
        acc_all.append(acc)
        ce_all.append(ce)
        # print (train_loss, valid_loss)
        # auc_all.append(auc)
        sys.stdout.flush()
        if args.use_wandb:
            writer.add_scalar("val_auc", valid_auc, i)
            writer.add_scalar("test_auc", auc, i)
            writer.add_scalar("val_acc", valid_acc, i)
            writer.add_scalar("test_acc", acc, i)
            writer.add_scalar("val_ce", valid_ce, i)
            writer.add_scalar("test_ce", ce, i)

    print(f"Validation AUCs   {np.array(val_auc_all)}")
    print(f"Test       AUCs   {np.array(auc_all)}")
    print(f"Test       CEs   {np.array(ce_all)}")

    print(
        "Validation AUC is {:.3f} with std {:.3f}".format(
            np.mean(val_auc_all), np.std(val_auc_all)
        )
    )
    print(
        "Validation ACC is {:.3f} with std {:.3f}".format(
            np.mean(val_acc_all), np.std(val_acc_all)
        )
    )
    print(
        "Validation ce is {:.3f} with std {:.3f}".format(
            np.mean(val_ce_all), np.std(val_ce_all)
        )
    )
    print(
        "Test       AUC is {:.3f} with std {:.3f}".format(
            np.mean(auc_all), np.std(auc_all)
        )
    )
    print(
        "Test       ACC is {:.3f} with std {:.3f}".format(
            np.mean(acc_all), np.std(acc_all)
        )
    )
    print(
        "Test       ce is {:.3f} with std {:.3f}".format(
            np.mean(ce_all), np.std(ce_all)
        )
    )

    if args.use_wandb:
        wandb.run.summary["final_test_auc"] = np.mean(auc_all)
        wandb.run.summary["final_test_auc_std"] = np.std(auc_all)
        wandb.run.summary["final_test_acc"] = np.mean(acc_all)
        wandb.run.summary["final_test_acc_std"] = np.std(acc_all)
        wandb.run.summary["final_test_ce"] = np.mean(ce_all)
        wandb.run.summary["final_test_ce_std"] = np.std(ce_all)

        wandb.run.summary["final_val_auc"] = np.mean(val_auc_all)
        wandb.run.summary["final_val_auc_std"] = np.std(val_auc_all)
        wandb.run.summary["final_val_acc"] = np.mean(val_acc_all)
        wandb.run.summary["final_val_acc_std"] = np.std(val_acc_all)
        wandb.run.summary["final_val_ce"] = np.mean(val_ce_all)
        wandb.run.summary["final_val_ce_std"] = np.std(val_ce_all)

    if args.dump_output:
        sys.stdout.close()

