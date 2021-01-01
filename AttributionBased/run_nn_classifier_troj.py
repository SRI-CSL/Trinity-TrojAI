# Classification with a MLP

import os
import pickle
import pandas as pd
import numpy as np
import scipy.interpolate as interpolate
from sklearn import metrics
import random
import ipdb
from sklearn import metrics
import tempfile
import pprint
import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append("utils_temporal_nn")
from utils_nn import (
    load_data,
    neuron_ds,
    TrojNet,
    get_members_list,
    train_model,
    get_spits,
    get_split_data,
)
from wrapper_nn import check_for_cuda, learner
import argparse


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


if __name__ == "__main__":
    # Initialize paths
    model_dir = "/data/SRI-Trinity/projects_2020/trojAI/round2-dataset-train"
    meta_data = pd.read_csv(
        "/data/SRI-Trinity/projects_2020/trojAI/round2-dataset-train/METADATA.csv"
    )
    feature_dir = "/data/SRI-Trinity/mount/rebel/SRI-Trinity/k8s/trinityTrojAI/users/SRI-Trinity/cvpr_exp/troj_round2/curve_features"

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_split", type=int, default=20)
    parser.add_argument("--val_split", type=int, default=20)
    parser.add_argument(
        "--arch",
        type=str,
        default="conv",
        choices=["conv", "conv_posenc", "transformer"],
    )
    parser.add_argument("--lr", type=float, default="1e-3")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max_class", type=int, default=24)
    parser.add_argument("--diff", action="store_true")
    parser.add_argument("--p", type=int, default=0.5)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--ckt_path", type=str)
    parser.add_argument("--stocastic", action="store_true")
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--nfolds", type=int, default=5)
    parser.add_argument("--dump_output", action="store_true")
    parser.add_argument("--output_file", type=str, default="test.txt")
    args = parser.parse_args()

    nfolds = args.nfolds
    # if args.ckt_path is None:
    # Overwriting the checkpoint name since it is not required
    _, args.ckt_path = tempfile.mkstemp()

    if args.dump_output:
        sys.stdout = open(args.output_file, "w")
        now = datetime.datetime.now()
        print("Current date and time : ")
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        pprint.pprint(args)
        sys.stdout.flush()

    #############################################################################
    # trigger_type = ['None', 'polygon']
    trigger_type = ["None", "polygon", "instagram"]
    # all_archs = ["resnet34", "resnet101", "resnet152", "resnet18", "resnet50"]
    all_archs = [
        "googlenet",
        "inceptionv3",
        "mobilenetv2",
        "resnet101",
        "resnet152",
        "resnet18",
        "resnet34",
        "resnet50",
        "vgg11bn",
        "vgg13bn",
        "vgg16bn",
        "vgg19bn",
        "wideresnet101",
        "wideresnet50",
        "squeezenetv1_1",
        "squeezenetv1_0",
        "shufflenet1_0",
        "shufflenet1_5",
        "shufflenet2_0",
        "densenet169",
        "densenet121",
        "densenet169",
    ]
    # models_filter = meta_data[meta_data.model_architecture.isin(all_archs)]
    models_filter = meta_data[meta_data.trigger_type.isin(trigger_type)]
    models_lst = models_filter.model_name.to_list()
    print("Running experiments on", all_archs)

    partitions = []
    for _ in range(nfolds):
        partitions.append(
            get_spits(models_filter, all_archs, args.test_split, args.val_split)
        )

    #############################################################################
    # Load data from all the models
    all_data = load_data(feature_dir, meta_data, models_lst)

    #############################################################################
    # Setup Cross-validation and model
    np.random.seed(0)
    auc_all = []
    scores_all = {}

    # Trainer and model
    # specify loss function
    """
    device = check_for_cuda()
    model = TrojNet()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # specify optimizer
    optimizer = torch.optim.Adam(model.parameters())
    patience = 20
    n_epochs = args.epochs
    ckt_path = "checkpoint.pt"
    """

    model = learner(args)

    auc_all = []
    acc_all = []
    val_auc_all = []
    val_acc_all = []
    for i in range(nfolds):
        train_split, val_split, test_split = partitions[i]
        print(
            "Partition sizes: Train-{}, Val-{}, Test-{}".format(
                len(train_split), len(val_split), len(test_split)
            )
        )

        labels_train, feats_train, archs_train, names_train = get_split_data(
            all_data, train_split
        )
        labels_val, feats_val, archs_val, names_val = get_split_data(
            all_data, val_split
        )
        labels_test, feats_test, archs_test, names_test = get_split_data(
            all_data, test_split
        )

        #     clf = LogisticRegression(random_state=0, class_weight='balanced', C=1)
        #     feats_train = scaler.fit(feats_train).transform(feats_train)
        #     clf.fit(feats_train, labels_train)
        #     scores = clf.predict_proba(scaler.transform(feats_test))[:, 1]

        # Dataloaders
        ds_train = neuron_ds(
            feats_train,
            labels_train,
            archs_train,
            names_train,
            max_class=args.max_class,
            diff=args.diff,
        )
        ds_test = neuron_ds(
            feats_test,
            labels_test,
            archs_test,
            names_test,
            max_class=args.max_class,
            diff=args.diff,
        )
        ds_val = neuron_ds(
            feats_val,
            labels_val,
            archs_val,
            names_val,
            max_class=args.max_class,
            diff=args.diff,
        )

        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
        dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

        train_loader = dl_train
        valid_loader = dl_test

        # if i == 0:
        # args.lr = model.cross_validation(dl_train)
        valid_acc, valid_auc = model.train(dl_train, dl_val)
        acc, auc, scores = model.test(dl_test)
        print(f"Fold {i}\nAcc: {acc:.2f}\nAuc: {auc:.2f}")

        # auc = train_model(
        #     model,
        #     patience,
        #     n_epochs,
        #     train_loader,
        #     valid_loader,
        #     optimizer,
        #     criterion,
        #     device,
        #     ckt_path=ckt_path,
        # )
        # acc = 0

        val_auc_all.append(valid_auc)
        val_acc_all.append(valid_acc)
        auc_all.append(auc)
        acc_all.append(acc)
        # print (train_loss, valid_loss)
        # auc_all.append(auc)
        sys.stdout.flush()

    print(f"Validation AUCs   {np.array(val_auc_all)}")
    print(f"Test       AUCs   {np.array(auc_all)}")

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
        "Test       AUC is {:.3f} with std {:.3f}".format(
            np.mean(auc_all), np.std(auc_all)
        )
    )
    print(
        "Test       ACC is {:.3f} with std {:.3f}".format(
            np.mean(acc_all), np.std(acc_all)
        )
    )

    if args.dump_output:
        sys.stdout.close()

