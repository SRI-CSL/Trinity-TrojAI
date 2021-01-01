"""
Loader, model, utility fns related to the classifier
Set correct paths in init_data_path
"""
import pickle
import os
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import random
from sklearn import metrics
import pandas as pd


#################################################
def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20
    epochs"""
    for param_group in optimizer.param_groups:
        param_group["lr"] = decay * param_group["lr"]


#################################################


def init_data_path(dataset_name, use_holdout, use_kubernetes=False):
    holdout_dir = None
    holdout_metadata = None
    if not use_kubernetes:
        troj_root = "/data/SRI/projects_2020/trojAI"
        cvpr_root = (
            "/data/SRI/mount/rebel/SRI/k8s/trinityTrojAI/users/SRI/cvpr_exp"
        )
    else:
        troj_root = "/data"
        cvpr_root = "/data/k8s/trinityTrojAI/users/SRI/cvpr_exp/"

    if dataset_name == "round2":
        model_dir = os.path.join(troj_root, "round2-dataset-train")
        meta_data = pd.read_csv(os.path.join(model_dir, "METADATA.csv"))
        feature_dir = os.path.join(cvpr_root, "troj_round2/curve_features_gradxact")
    if dataset_name == "round3":
        model_dir = os.path.join(troj_root, "round3-dataset-train")
        meta_data = pd.read_csv(os.path.join(model_dir, "METADATA.csv"))
        feature_dir = os.path.join(cvpr_root, "troj_round3/curve_features")
    if dataset_name == "round1":
        model_dir = os.path.join(troj_root, "round1-dataset-train", "models")
        meta_data = pd.read_csv(
            os.path.join(troj_root, "round1-dataset-train", "METADATA.csv")
        )
        feature_dir = os.path.join(cvpr_root, "troj_round1/curve_features_gradxact")
        if use_holdout:
            holdout_dir = os.path.join(
                cvpr_root, "troj_round1", "/holdout/curve_features_gradxact"
            )
        else:
            holdout_dir = None
        holdout_metadata = pd.read_csv(
            os.path.join(troj_root, "round1-holdout-dataset/METADATA.csv")
        )
        meta_data = meta_data.rename(columns={"ground_truth": "poisoned"})
        holdout_metadata = holdout_metadata.rename(columns={"ground_truth": "poisoned"})
    elif dataset_name == "mnist":
        model_dir = os.path.join(troj_root, "mnist-dataset")
        meta_data = pd.read_csv(os.path.join(model_dir, "METADATA.csv"))
        feature_dir = os.path.join(cvpr_root, "mnist/curve_features_gradxact")

    return model_dir, meta_data, feature_dir, holdout_dir, holdout_metadata


def get_data_info(args, meta_data):
    if args.dataset == "round3":

        # all_archs = [
        #     "densenet169",
        #     "densenet121",
        #     "densenet169",
        # ]
        # meta_data = meta_data[meta_data.model_architecture.isin(all_archs)]

        trigger_type = ["None", "polygon", "instagram"]
        all_archs = meta_data.model_architecture.unique()
        models_filter = meta_data[meta_data.model_architecture.isin(all_archs)]
        models_filter = meta_data[meta_data.trigger_type.isin(trigger_type)]
        models_lst = models_filter.model_name.to_list()
    if args.dataset == "round2":
        trigger_type = ["None", "polygon", "instagram"]

        # For ablation, ignore
        # filter_fn = filters_ablation()
        # out = meta_data["triggered_classes"].apply(filter_fn.triggered_classes)
        # meta_data = meta_data[out]
        # print("Number of triggered classes are 1")
        # print(len(meta_data))
        # print(meta_data[meta_data["poisoned"]]["triggered_classes"])

        # trigger_type = ["None", "polygon"]
        # models_lst = meta_data[
        #     meta_data["poisoned"] == False
        # ].model_name.values.tolist()

        # _meta_data = meta_data[meta_data["poisoned"]]
        # models_lst += _meta_data[
        #     _meta_data.triggered_fraction >= 0.29
        # ].model_name.values.tolist()[:135]
        # models_filter = meta_data[meta_data.model_name.isin(models_lst)]
        # all_archs = meta_data.model_architecture.unique()
        # print(models_filter.triggered_fraction.values)

        # all_archs = ["resnet34", "resnet101", "resnet152", "resnet18", "resnet50"]
        # all_archs = [
        #     "googlenet",
        #     "inceptionv3",
        #     "mobilenetv2",
        #     "resnet101",
        #     "resnet152",
        #     "resnet18",
        #     "resnet34",
        #     "resnet50",
        #     "vgg11bn",
        #     "vgg13bn",
        #     "vgg16bn",
        #     "vgg19bn",
        #     "wideresnet101",
        #     "wideresnet50",
        #     "squeezenetv1_1",
        #     "squeezenetv1_0",
        #     "shufflenet1_0",
        #     "shufflenet1_5",
        #     "shufflenet2_0",
        #     "densenet169",
        #     "densenet121",
        #     "densenet169",
        # ]
        all_archs = meta_data.model_architecture.unique()
        models_filter = meta_data[meta_data.model_architecture.isin(all_archs)]
        models_filter = meta_data[meta_data.trigger_type.isin(trigger_type)]
        models_lst = models_filter.model_name.to_list()
    if args.dataset == "round1":
        all_archs = meta_data.model_architecture.unique()
        # all_archs = np.asarray(["resnet50"])
        models_filter = meta_data[meta_data.model_architecture.isin(all_archs)]
        models_lst = models_filter.model_name.to_list()

        # Done for ablation, ignore
        # all_archs = meta_data.model_architecture.unique()
        # # meta_data = meta_data[meta_data.model_architecture == "inceptionv3"]
        # models_lst = meta_data[meta_data["poisoned"]].model_name.values.tolist()[:125]
        # models_lst += meta_data[
        #     meta_data["poisoned"] == False
        # ].model_name.values.tolist()[:125]
        # models_filter = meta_data[meta_data.model_name.isin(models_lst)]

    elif args.dataset == "mnist":
        # all_archs = ["BadNetExample", "ModdedBadNetExample", "ModdedLeNet5Net"]
        # all_archs = np.asarray(["BadNetExample"])
        all_archs = meta_data.model_architecture.unique()
        models_filter = meta_data[meta_data.model_architecture.isin(all_archs)]
        models_lst = models_filter.model_name.to_list()
        try:
            models_lst.remove("id-00000359")
        except:
            pass
        try:
            models_lst.remove("id-00000587")
        except:
            pass

    return all_archs, models_filter, models_lst


def _get_split(df, _a, val_perc=10, test_perc=10):
    if isinstance(_a, str):
        _lst = df[df.model_architecture == _a].model_name.to_list()
    elif isinstance(_a, list):
        _lst = df[df.model_architecture.isin(_a)].model_name.to_list()
    random.shuffle(_lst)
    _val_split = len(_lst) * val_perc // 100
    _test_split = len(_lst) * test_perc // 100
    return (
        _lst[(_val_split + _test_split) :],
        _lst[_val_split : (_val_split + _test_split)],
        _lst[:_val_split],
    )


def get_splits(models_filter, all_archs, val_split, test_split):
    models_poison = models_filter[models_filter.poisoned == True]
    models_clean = models_filter[models_filter.poisoned == False]

    model_idx_train = []
    model_idx_val = []
    model_idx_test = []
    for _a in all_archs:
        train_lst, val_lst, test_lst = _get_split(
            models_clean, _a, val_perc=val_split, test_perc=test_split
        )
        model_idx_train += train_lst
        model_idx_test += test_lst
        model_idx_val += val_lst
        train_lst, val_lst, test_lst = _get_split(
            models_poison, _a, val_perc=val_split, test_perc=test_split
        )
        model_idx_train += train_lst
        model_idx_test += test_lst
        model_idx_val += val_lst

    return model_idx_train, model_idx_val, model_idx_test


def get_splits_random(models_filter, all_archs, val_split, test_split):
    model_idx_train, model_idx_val, model_idx_test = _get_split(
        models_filter, all_archs.tolist(), val_perc=val_split, test_perc=test_split
    )
    # models_poison = models_filter[models_filter.poisoned == True]
    # models_clean = models_filter[models_filter.poisoned == False]

    # model_idx_train = []
    # model_idx_val = []
    # model_idx_test = []
    # for _a in all_archs:
    #     train_lst, val_lst, test_lst = _get_split(
    #         models_clean, _a, val_perc=val_split, test_perc=test_split
    #     )
    #     model_idx_train += train_lst
    #     model_idx_test += test_lst
    #     model_idx_val += val_lst
    #     train_lst, val_lst, test_lst = _get_split(
    #         models_poison, _a, val_perc=val_split, test_perc=test_split
    #     )
    #     model_idx_train += train_lst
    #     model_idx_test += test_lst
    #     model_idx_val += val_lst

    return model_idx_train, model_idx_val, model_idx_test


def get_split_data(all_data, model_names):
    labels, feats, archs, names = all_data

    labels_ret = [f for f, n in zip(labels, names) if n in model_names]
    feats_ret = [f for f, n in zip(feats, names) if n in model_names]
    archs_ret = [f for f, n in zip(archs, names) if n in model_names]
    names_ret = [f for f, n in zip(names, names) if n in model_names]

    return labels_ret, feats_ret, archs_ret, names_ret


def load_data(feature_dir, meta_data, models_lst):
    labels = []
    feats = []
    archs = []
    model_name_all = []
    for model_name in models_lst:
        file_name = os.path.join(feature_dir, "{}.pkl".format(model_name))
        if os.path.exists(file_name):
            archs.append(
                meta_data[meta_data["model_name"] == model_name][
                    "model_architecture"
                ].item()
            )
            feats.append(pickle.load(open(file_name, "rb")))
            labels.append(
                int(meta_data[meta_data["model_name"] == model_name]["poisoned"].item())
            )
            model_name_all.append(model_name)

    print(f"Loaded {len(feats)} feats")
    return labels, feats, archs, model_name_all


class neuron_ds(Dataset):
    def __init__(
        self,
        feats,
        _labels,
        _archs,
        _names,
        max_class=24,
        diff=False,
        normalize=True,
        feat_sz=2048,
        collapse_graph=True,
    ):
        self.labels = _labels
        self.archs = _archs
        self.names = _names

        div = 100 if normalize else 1
        _curves = [np.array(en[0])[:max_class, :] / div for en in feats]

        #         import ipdb; ipdb.set_trace()
        if collapse_graph:
            feat_sz = 100
            _curves_new = []
            for _c, _f in zip(_curves, feats):
                _idx = np.array(_f[1])
                _c_new = np.zeros((_c.shape[0], 100))
                for _i in range(100):
                    # _c_new[:, _i] = _c[:, _idx == _i].mean(axis=-1)
                    if (_idx == _i).sum() > 0:
                        _c_new[:, _i] = _c[:, _idx == _i].mean(axis=-1)
                    else:
                        # Changing to find the nearest pt if above fails
                        x = np.abs((_idx - _i)).argmin()
                        _c_new[:, _i] = _c[:, x]
                _curves_new.append(_c_new)
            _curves = _curves_new

        if diff:
            _curves = [x[:, :-1] - x[:, 1:] for x in _curves]

        self.feats = np.zeros((len(_curves), max_class, feat_sz))
        self.valid = np.zeros((len(_curves), max_class))
        for idx, en in enumerate(_curves):
            val = en.shape[0]
            self.feats[idx][:val, : en.shape[1]] = en
            self.valid[idx][:val] = 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.feats[idx],
            self.valid[idx],
            self.labels[idx],
            self.archs[idx],
            self.names[idx],
        )


class TrojNet(nn.Module):
    def __init__(self, max_class=24, p=0.5, embedding_dim=16):
        super(TrojNet, self).__init__()
        self.max_class = max_class
        #         self.all_classes = all_classes

        #         self.embed = nn.Embedding(len(all_classes), embedding_dim)

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=13, stride=1),
            nn.BatchNorm1d(4),
            nn.MaxPool1d(9, stride=2),
            nn.ReLU(),
            nn.Conv1d(4, embedding_dim, kernel_size=13, stride=1),
            nn.BatchNorm1d(embedding_dim),
            nn.MaxPool1d(9, stride=2),
            nn.ReLU(),
        )

        self.drop = nn.Dropout(p=p)

        self.fc = nn.Sequential(nn.Linear(160, 20), nn.ReLU(), nn.Linear(20, 2))

    def forward(self, data):
        _input, _valid, _arch = data
        _valid = _valid.float()
        _input = _input.float().unsqueeze(2)
        #         _input = _input.permute(1,0,2).float()
        #         _valid = _valid.permute(1,0)

        #         _arch_idx = torch.Tensor([self.all_classes.index(en) for en in _arch]).long()
        #         emb = self.embed(_arch_idx)
        #         print (_input.shape)
        _enc = []
        for idx in range(self.max_class):
            _enc.append(self.encoder(_input[:, idx, ...]))
        feature = torch.stack(_enc, dim=1)
        #         print (feature.shape)

        feature = _valid.unsqueeze(-1).unsqueeze(-1) * feature

        pooled, _ = feature.max(dim=1)
        #         import ipdb; ipdb.set_trace()

        #         pooled = emb.unsqueeze(-1)*pooled

        flatten = pooled.view(pooled.shape[0], -1)
        flatten = self.drop(flatten)

        #         import ipdb; ipdb.set_trace()
        out = self.fc(flatten)

        return out


def get_members_list(lst, idxs):
    # Return specific elements from a list
    return [lst[it] for it in idxs]


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation measure update ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_model(
    model,
    patience,
    n_epochs,
    train_loader,
    valid_loader,
    optimizer,
    criterion,
    device,
    ckt_path="checkpoint.pt",
):

    #     if os.path.exists(ckt_path):
    #         model.load_state_dict(torch.load(ckt_path))

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    train_acc = []
    valid_acc = []
    avg_train_acc = []
    avg_valid_acc = []

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=ckt_path)

    for epoch in range(1, n_epochs + 1):
        model.train()  # prep model for training
        for batch, data in enumerate(train_loader, 1):
            _input, _valid, _label, _arch, _name = data
            _input = _input.to(device)
            _label = _label.to(device)
            _valid = _valid.to(device)
            optimizer.zero_grad()
            out = model((_input, _valid, _arch))

            #             import ipdb; ipdb.set_trace()
            loss = criterion(out, _label)
            _, pred = out.max(dim=-1)
            #             pred = (torch.sigmoid(out)<0.5).float().squeeze()
            acc = (_label == pred).sum().item() / len(_label)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_acc.append(acc)

        model.eval()  # prep model for evaluation
        predictions = []
        labels = []
        scores = []
        for data in valid_loader:
            _input, _valid, _label, _arch, _name = data
            _input = _input.to(device)
            _label = _label.to(device)
            _valid = _valid.to(device)
            out = model((_input, _valid, _arch))

            loss = criterion(out, _label)
            _, pred = out.max(dim=-1)
            #             pred = (torch.sigmoid(out)<0.5).float().squeeze()
            acc = (_label == pred).sum().item() / len(_label)

            valid_losses.append(loss.item())
            valid_acc.append(acc)
            predictions.append(pred.data.cpu().numpy())
            labels.append(_label.data.cpu().numpy())
            scores.append(out.data.cpu().numpy())

        scores = np.vstack(scores)[:, 1]
        predictions = np.hstack(predictions)
        labels = np.hstack(labels)
        acc = (predictions == labels).mean() * 100
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        auc = metrics.auc(fpr, tpr)
        print("acc", acc, "auc", auc)

        #         ipdb.set_trace()
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        train_a = np.average(train_acc)
        valid_a = np.average(valid_acc)
        avg_train_acc.append(train_a)
        avg_valid_acc.append(valid_a)

        epoch_len = len(str(n_epochs))

        print_msg = (
            f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
            + f"train_loss: {train_loss:.5f} "
            + f"valid_loss: {valid_loss:.5f} "
            + f"train_acc: {train_a:.5f} "
            + f"valid_acc: {valid_a:.5f}"
        )

        print(print_msg)

        train_losses = []
        valid_losses = []
        train_acc = []
        valid_acc = []

        early_stopping(valid_loss, model)
        #         early_stopping(-valid_a, model)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    model.load_state_dict(torch.load(ckt_path))

    # return model, avg_train_losses, avg_valid_losses, avg_train_acc, avg_valid_acc
    return auc


def initialize_wandb(args):
    # Initialize wandb
    if args.use_wandb:
        import wandb

        wandb.init(
            project=args.project_name,
            config=vars(args),
            sync_tensorboard=True,
            name=args.env_name,
            entity="SRI",
            tags=args.tags.split(","),
            dir="/home/SRI/troj",
        )
    else:
        wandb = None
    return wandb


class filters_ablation:
    def triggered_classes(self, triggered_classes):
        if triggered_classes == "None":
            return True
        else:
            triggered_classes = [int(k) for k in triggered_classes[1:-1].split(" ")]
            if len(triggered_classes) > 2:
                return True
            else:
                return False
