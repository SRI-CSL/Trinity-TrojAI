import numpy as np
import torch
import os
import sys
from matplotlib import pyplot as plt
import torch.nn as nn
from xplain.attr import LayerIntegratedGradients, LayerGradientXActivation
import skimage.io
import torchvision
import pickle
import pandas as pd
import scipy.interpolate as interpolate
from torch.utils.data import TensorDataset, DataLoader
import helper
import argparse
from tqdm import tqdm
import warnings
from helper import pre_process_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F


warnings.filterwarnings("ignore")

# attribute_to_layer_input=True,


def densenet_feat_preprocess(x):
    out = F.relu(x, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    return out


def shufflenet_feat_preprocess(x):
    x = x.mean([2, 3])  # globalpool
    return x


def inception_feat_preprocess(x):
    x = F.adaptive_avg_pool2d(x, (1, 1))
    # N x 2048 x 1 x 1
    x = F.dropout(x, training=False)
    # N x 2048 x 1 x 1
    x = torch.flatten(x, 1)
    return x


model_to_layer_dict = {
    "DenseNet": ("features.norm5", None, True),
    "ResNet": ("avgpool", None, False),
    "VGG": ("classifier", 4, False),
    "GoogLeNet": ("dropout", None, False),
    # "Inception3": ("dropout", None, False),
    "Inception3": ("Mixed_7c", None, True),
    "SqueezeNet": ("features", None, False),
    "ShuffleNetV2": ("conv5", 2, True),
    "MobileNetV2": ("classifier", 0, False),
}

model_to_feature_dict = {
    "DenseNet": ("classifier", None, densenet_feat_preprocess),
    "ResNet": ("fc", None, None),
    "VGG": ("classifier", 6, None),
    "GoogLeNet": ("fc", None, None),
    # "Inception3": ("fc", None, None),
    "Inception3": ("fc", None, inception_feat_preprocess),
    "SqueezeNet": ("classifier", None, None),
    "ShuffleNetV2": ("fc", None, shufflenet_feat_preprocess),
    "MobileNetV2": ("classifier", 1, None),
}


def get_layer(model, layer_name, layer_index):
    _layer_names = layer_name.split(".")

    en = model
    for lm in _layer_names:
        en = getattr(en, lm)

    if layer_index is not None:
        en = en[layer_index]

    return en


def get_dl(file_list, round):
    labels = torch.tensor([int(en.split("/")[-1].split("_")[1]) for en in file_list])
    img_batch_load = []
    for img in file_list:
        img = pre_process_image(img, round=round)
        batch_data = torch.FloatTensor(img)
        img_batch_load.append(batch_data)
    img_batch = torch.stack(img_batch_load, 0).squeeze()
    dataset = TensorDataset(torch.Tensor(img_batch), labels)
    dl = DataLoader(dataset, batch_size=2)

    return dl


def identify_bad_neurons(target, attribution, logits_per_class):
    tmp = []

    for cls in range(num_cls):
        if cls == target:
            continue
        _idx = logits_per_class[cls].argsort()[::-1][1:3]
        # if _idx != target:
        # if not target in _idx:
        # continue
        _idx = (labels == cls).nonzero()[0]
        # import ipdb; ipdb.set_trace()
        # attribution_mean = attribution[_idx].mean(0)[..., target]
        # attribution_mean = attribution_mean.flatten()
        if attribution.ndim > 3:
            attribution = attribution.mean(-2).mean(-2)
        assert attribution.ndim == 3, "Check size of attribution"
        attribution_mean = attribution[_idx].mean(0)[:, target]
        _idx = attribution_mean > 0
        try:
            thresh = np.percentile(attribution_mean[_idx], 20)
        except:
            # If all attributions are < 0
            thresh = 0
        attribution_mean[attribution_mean < thresh] = 0
        tmp.append(attribution_mean)

    assert np.mean(tmp, 0).ndim == 1, "bad neurons have ndim > 1"
    bad_neurons = np.mean(tmp, 0).argsort()[::-1].tolist()
    assert bad_neurons
    return bad_neurons


def ablation_plot(dataloader, bad_neurons, target, activation_value=25):
    acc_all = []
    nn_all = []
    N = int(NUM_NEURONS)

    for nn in range(0, N, 2):
        pred = []
        gnd = []
        logits_clean = []

        for data in feat_dl:
            _feat, label = data

            _feat = _feat.to(device)
            # if _feat.ndim != 2:
            #     _feat_shape = _feat.shape
            #     _feat_f = _feat.view(_feat_shape[0], -1)
            #     _feat_f[:, bad_neurons[:nn]] = activation_value
            #     _feat = _feat_f.view(*_feat_shape)
            # else:
            #     _feat[:, bad_neurons[:nn]] = activation_value
            if feat_preprocess is not None:
                _feat = feat_preprocess(_feat)
            if _feat.ndim > 2:
                _feat[:, bad_neurons[:nn], ...] = activation_value
            else:
                _feat[:, bad_neurons[:nn]] = activation_value
            logits = _feat_layer(_feat).squeeze()
            logits_clean.append(logits.data.cpu().numpy())
            pred.append(logits.argmax(1).data.cpu().numpy())
            gnd.append(label.numpy())

        logits_clean = np.vstack(logits_clean)
        acc = np.mean(np.hstack(gnd) == np.hstack(pred)) * 100
        acc_all.append(acc)
        nn_all.append(int(nn / NUM_NEURONS * 100))
        kk = 0

    # % neurons where perf = P
    f = interpolate.interp1d(acc_all, nn_all)

    try:
        P = 20
        position_0 = f(P)
    except:
        position_0 = 0

    try:
        P = 40
        position_1 = f(P)
    except:
        position_1 = 0

    if target < 12:
        plt.plot(nn_all, acc_all)
        plt.plot(nn_all, 20 * np.ones((len(nn_all))))
        plt.plot(nn_all, 40 * np.ones((len(nn_all))), color="red")
        plt.ylabel("Accuracy")
        plt.xlabel("Percentage of neurons triggered in the layer")
        plt.title(f"Ablation for class {target}, Position={position_1}")

    print(target, ":", position_0, position_1)
    return acc_all, nn_all, position_1


def forward_fn(
    model, dataloader, compute_attribution=True, use_internal_batch_size=True
):
    pred = []
    gnd = []
    logits = []
    attribution = []
    labels = []
    feat = []
    for data in tqdm(dataloader):
        img, label = data
        labels.append(label)
        model(img.to(device))
        _feat = hook_fn_feat_layer.outputs
        if isinstance(_feat, list):
            _feat = _feat[0]
        feat.append(_feat.data.cpu().numpy())
        # import ipdb; ipdb.set_trace()
        if feat_preprocess is not None:
            _feat = feat_preprocess(_feat)
        _logits = _feat_layer(_feat).squeeze()
        logits.append(_logits.data.cpu().numpy())
        pred.append(_logits.argmax(1).data.cpu().numpy())
        gnd.append(label.numpy())

        if compute_attribution:
            _attrib = []
            for c in range(num_cls):
                if use_internal_batch_size:
                    _atr = attrib_fn.attribute(
                        img.to(device),
                        target=torch.Tensor([c, c]).to(device).long(),
                        internal_batch_size=4,
                        attribute_to_layer_input=attribute_to_layer_input,
                    )
                else:
                    _atr = attrib_fn.attribute(
                        img.to(device),
                        target=torch.Tensor([c, c]).to(device).long(),
                        attribute_to_layer_input=attribute_to_layer_input,
                    )
                if isinstance(_atr, tuple):
                    _atr = _atr[0]
                _attrib.append(_atr.unsqueeze(-1).cpu().data.numpy())
            attribution.append(np.concatenate(_attrib, axis=-1))

    logits = np.vstack(logits)
    labels = np.hstack(labels)
    attribution = np.vstack(attribution)
    attribution = np.squeeze(attribution)

    feat = np.vstack(feat)
    acc = np.mean(np.hstack(gnd) == np.hstack(pred)) * 100
    print("Accuracy is ", acc)
    print("feat_shape: ", feat.shape)
    print("attr_shape: ", attribution.shape)

    return logits, labels, attribution, feat, acc


# def get_feature(meta_idx, model_dir, meta_data):

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get attribution")
    # parser.add_argument("--meta_idx", type=int, default=439)
    parser.add_argument("--model_name", type=str, default="id-00000823")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/data/SRI/projects_2020/trojAI/round2-dataset-train",
    )
    parser.add_argument(
        "--meta_data",
        type=str,
        default="/data/SRI/projects_2020/trojAI/round2-dataset-train/METADATA.csv",
    )
    parser.add_argument("--results_dir", type=str, default="curve_features")
    parser.add_argument("--attributions_dir", type=str, default="attribution_features")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mult_factor", type=int, default=2)
    parser.add_argument("--round", type=int, default=2)
    parser.add_argument(
        "--attribution_fn", type=str, default="IG", choices=["IG", "GradxAct"]
    )
    args = parser.parse_args()

    device = args.device
    # meta_idx = args.meta_idx
    model_dir = args.model_dir
    meta_data = args.meta_data
    MULT_FACTOR = args.mult_factor

    # meta_idx = 64       #inceptionv3

    # meta_idx = 371      #densenet121
    # meta_idx = 269      #densenet161
    # meta_idx = 342      #densenet169
    # meta_idx = 205      #densenet201
    # meta_idx = 489  # shufflenet1_0
    # meta_idx = 463      #shufflenet1_5
    # meta_idx = 152      #shufflenet2_0

    # meta_idx = 272      #squeezenetv1_0 .. acc not coming
    # meta_idx = 18       #squeezenetv1_1

    meta_data = pd.read_csv(meta_data)

    # model_info = meta_data.loc[meta_idx]
    model_info = meta_data[meta_data["model_name"] == args.model_name]
    model_name = model_info.model_name.item()
    model_curr_dir = os.path.join(model_dir, model_name)

    model_filepath = os.path.join(model_curr_dir, "model.pt")
    model = torch.load(model_filepath, map_location=device)
    model = model.eval()

    # print (model)
    num_cls = model_info.number_classes.item()

    tt = type(model).__name__
    # print(model_info.model_architecture)
    # print(tt)
    info = model_to_layer_dict.get(tt)
    layer_name = info[0]
    layer_index = info[1]
    attribute_to_layer_input = info[2]

    _layer = get_layer(model, layer_name, layer_index)
    # print (layer_name)
    # print (_layer)

    hook_fn_feat_layer = helper.hook_fn_nn()
    _layer.register_forward_hook(hook_fn_feat_layer)

    info = model_to_feature_dict.get(tt)
    layer_name = info[0]
    layer_index = info[1]
    feat_preprocess = info[2]
    _feat_layer = get_layer(model, layer_name, layer_index)

    if args.attribution_fn == "IG":
        attribution_fn = LayerIntegratedGradients
        use_internal_batch_size = True
    elif args.attribution_fn == "GradxAct":
        attribution_fn = LayerGradientXActivation
        use_internal_batch_size = False

    if attribute_to_layer_input:
        attrib_fn = attribution_fn(model, _feat_layer)
    else:
        attrib_fn = attribution_fn(model, _layer)

    # print (model)
    # print (_layer)
    # print (_feat_layer)
    # x = torch.rand((2, 3, 224, 224)).to(device)
    # l = model(x)
    # print (l.shape)
    # f = hook_fn_feat_layer.outputs
    # print (f.shape)
    # print (attribute_to_layer_input)
    # a = attrib_fn.attribute(x,target=torch.Tensor([0,0]).to(device).long(),
    #            internal_batch_size=4, attribute_to_layer_input=attribute_to_layer_input)
    ##import ipdb; ipdb.set_trace()
    # print (a.shape)

    # if feat_preprocess is not None:
    #    f = feat_preprocess(f)
    # o = _feat_layer(f)
    # print (o.shape)

    # exit()

    if args.round == 3:
        clean_image_dir = os.path.join(model_curr_dir, "clean_example_data")
    else:
        clean_image_dir = os.path.join(model_curr_dir, "example_data")
    clean_images = [
        os.path.join(clean_image_dir, en)
        for en in os.listdir(clean_image_dir)
        if en.endswith(".png")
    ]
    # dataloader = get_dl(clean_images[:10])
    dataloader = get_dl(clean_images, round=args.round)

    attribution_path = os.path.join(args.attributions_dir, "{}.npz".format(model_name))
    if os.path.exists(attribution_path):
        # if False:
        data = np.load(attribution_path, allow_pickle=True)
        logits = data["logits"]
        labels = data["labels"]
        attribution = data["attribution"]
        feat = data["feat"]
        acc = data["acc"]
        print("Accuracy is ", acc)
        print("feat_shape: ", feat.shape)
        print("attr_shape: ", attribution.shape)
    else:
        logits, labels, attribution, feat, acc = forward_fn(
            model, dataloader, use_internal_batch_size=use_internal_batch_size
        )
        np.savez(
            attribution_path,
            logits=logits,
            labels=labels,
            attribution=attribution,
            feat=feat,
            acc=acc,
        )
        sys.exit()

    feat_ds = TensorDataset(torch.from_numpy(feat), torch.from_numpy(labels))
    feat_dl = DataLoader(feat_ds, batch_size=8)

    logits_per_class = []
    for i in range(num_cls):
        idx = (labels == i).nonzero()[0]
        logits_per_class.append(logits[idx].mean(0))
    logits_per_class = np.asarray(logits_per_class)

    NUM_NEURONS = feat.shape[1]

    res_file = os.path.join(args.results_dir, "{}.pkl".format(model_name))
    print(res_file)
    # if not os.path.exists(res_file):
    if True:
        fig = plt.figure(figsize=[20, 20])
        acc_ablation = []
        position = []
        # ipdb.set_trace()
        M = feat.mean(0).max() * MULT_FACTOR
        print("Using activation value", M)

        for target in range(num_cls):
            print(f"Running ablation for class {target}/{num_cls} ")
            if target < 12:
                ax = plt.subplot(4, 3, target + 1)
            bad_neurons = identify_bad_neurons(target, attribution, logits_per_class)
            _acc, nn_all, pos = ablation_plot(
                dataloader, bad_neurons, target, activation_value=M
            )
            position.append(pos)
            acc_ablation.append(_acc)

        pickle.dump((acc_ablation, nn_all, position), open(res_file, "wb"))

        position = np.asarray(position)
        print(f"Poisoned class is {position.argmin()} with {position.min()}")
        plt.savefig(
            os.path.join(
                args.results_dir,
                "{}_{}.jpg".format(model_name, model_info.model_architecture.item()),
            )
        )
        print(f"Finished model {model_name}")

