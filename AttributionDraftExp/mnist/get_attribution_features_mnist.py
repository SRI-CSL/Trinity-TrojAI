import numpy as np
import torch
import os
import sys
from matplotlib import pyplot as plt
import torch.nn as nn
from captum.attr import LayerIntegratedGradients, LayerGradientXActivation
import skimage.io
import torchvision
import pickle
import scipy.interpolate as interpolate
import helper
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
from torch.utils.data import TensorDataset, DataLoader

# Later push to config/fn.
src = "/data/mnist-dataset"
# src = "/data/ksikka/mount/rebel/ksikka/mnist-dataset"
device = torch.device("cuda")
num_cls = 10


########################################################################
def create_dataloader(data_src):
    filenames, labels = list(
        zip(
            *[
                it.strip().split(",")
                for it in open(os.path.join(data_src, "labels.txt"))
            ]
        )
    )
    labels = [int(it) for it in labels]
    vecs = torch.stack(
        [
            torch.from_numpy(skimage.io.imread(os.path.join(data_src, img_loc))).float()
            for img_loc in filenames
        ]
    )
    labels = torch.Tensor(labels)
    dataset = TensorDataset(vecs, labels)
    dataloader = DataLoader(dataset, batch_size=2)

    return dataloader


########################################################################


if __name__ == "__main__":
    model_name = sys.argv[1]
    attribution_fn = "GradxAct"
    # attribution_fn = "IG"
    attribution_dir = "attribution_features_gradxact"
    curves_dir = "curve_features_gradxact"

    model = torch.load(os.path.join(src, f"{model_name}", "model.pt.1")).to(device)
    model_type = type(model).__name__
    print(model_type)
    model.eval()

    # Create hook since last layer is softmax
    hook_fn_logit_layer = helper.hook_fn_nn()
    hook_fn_feat_layer = helper.hook_fn_nn()

    if model_type == "ModdedLeNet5Net":
        NUM_NEURONS = 84
        MULT_FACTOR = 2
    elif model_type == "BadNetExample":
        NUM_NEURONS = 512
        MULT_FACTOR = 2
    elif model_type == "ModdedBadNetExample":
        NUM_NEURONS = 512
        MULT_FACTOR = 1.5

    if model_type != "ModdedBadNetExample":
        model.fc[2].register_forward_hook(hook_fn_logit_layer)
        model.fc[0].register_forward_hook(hook_fn_feat_layer)
        layer_pointer = model.fc[0]
    else:
        model.fc[0].register_forward_hook(hook_fn_logit_layer)
        model.convnet[5].register_forward_hook(hook_fn_feat_layer)
        layer_pointer = model.convnet[5]

    print("\n\n\nModel-name", model_name)
    print("Model-type", model_type)
    print("Num_neurons", NUM_NEURONS)

    ###########################################################################
    def forward_fn(
        model, dataloader, model_type, attribution_type, compute_attribution=True
    ):
        print(attribution_type)
        if attribution_type == "IG":
            attribution_fn = LayerIntegratedGradients
            use_internal_batch_size = True
        elif attribution_type == "GradxAct":
            attribution_fn = LayerGradientXActivation
            use_internal_batch_size = False
        attrib_fn = attribution_fn(model, layer_pointer)
        pred = []
        gnd = []
        logits = []
        attribution = []
        labels = []
        feat = []
        for data in tqdm(dataloader):
            img, label = data
            labels.append(label)
            model(img.unsqueeze(1).to(device))
            _logits = hook_fn_logit_layer.outputs
            logits.append(_logits.data.cpu().numpy())
            _feat = hook_fn_feat_layer.outputs
            if model_type == "ModdedBadNetExample":
                feat.append(_feat.view(_feat.shape[0], -1).data.cpu().numpy())
            else:
                feat.append(_feat.data.cpu().numpy())
            pred.append(_logits.argmax(1).data.cpu().numpy())
            gnd.append(label.numpy())
            # Compute attribution over all the classes
            if compute_attribution:
                _attrib = []
                for c in range(num_cls):
                    if use_internal_batch_size:
                        __attrib = attrib_fn.attribute(
                            img.unsqueeze(1).to(device),
                            target=torch.Tensor([c, c]).to(device).long(),
                            internal_batch_size=4,
                        )
                    else:
                        __attrib = attrib_fn.attribute(
                            img.unsqueeze(1).to(device),
                            target=torch.Tensor([c, c]).to(device).long(),
                        )
                    #     __attrib = attrib_fn.attribute(
                    #         img.unsqueeze(1).to(device),
                    #         target=torch.Tensor([c, c]).to(device).long(),
                    #     )
                    if __attrib.ndim > 2:
                        __attrib = __attrib.view(__attrib.shape[0], -1)
                    _attrib.append(__attrib.unsqueeze(-1).data.cpu().numpy())
                attribution.append(np.concatenate(_attrib, axis=-1))

        logits = np.vstack(logits)
        labels = np.hstack(labels)
        feat = np.vstack(feat)
        if compute_attribution:
            attribution = np.concatenate(attribution, 0)
        acc = np.mean(np.hstack(gnd) == np.hstack(pred)) * 100
        print("Accuracy is ", acc)
        return logits, labels, attribution, feat, acc

    ###########################################################################
    attribution_path = os.path.join(attribution_dir, model_name + ".npy")
    if not os.path.exists(attribution_path):
        dataloader = create_dataloader(os.path.join(src, "example"))
        print("Extracting features")
        logits, labels, attribution, feat, acc = forward_fn(
            model, dataloader, model_type, attribution_fn
        )

        # Save these
        np.save(
            attribution_path,
            {
                "logits": logits,
                "labels": labels,
                "attribution": attribution,
                "feat": feat,
            },
        )
    else:
        tmp = np.load(attribution_path, allow_pickle=True).item()
        logits = tmp["logits"]
        labels = tmp["labels"]
        attribution = tmp["attribution"]
        feat = tmp["feat"]

    # ipdb.set_trace()
    feat_ds = TensorDataset(torch.from_numpy(feat), torch.from_numpy(labels))
    feat_dl = DataLoader(feat_ds, batch_size=2)

    ###########################################################################
    def identify_bad_neurons(target, attribution):
        tmp = []

        for cls in range(num_cls):
            if cls == target:
                continue
            _idx = (labels == cls).nonzero()[0]
            attribution_mean = attribution[_idx].mean(0)[:, target]
            _idx = attribution_mean > 0
            thresh = np.percentile(attribution_mean[_idx], 20)
            attribution_mean[attribution_mean < thresh] = 0
            tmp.append(attribution_mean)

        bad_neurons = np.mean(tmp, 0).argsort()[::-1].tolist()
        return bad_neurons

    ###########################################################################
    def ablation_plot(feat_dl, bad_neurons, activation_value=25):
        acc_all = []
        nn_all = []
        N = int(NUM_NEURONS)

        for nn in range(0, N, 2):
            pred = []
            gnd = []
            logits_clean = []

            #         for data in dataloader:
            for data in feat_dl:
                feat, label = data
                feat = feat.to(device)
                #             model(img.unsqueeze(1).to(device))
                #             feat = hook_fn_feat_layer.outputs
                # Following needs to be replaced
                if model_type != "ModdedBadNetExample":
                    feat[:, bad_neurons[:nn]] = activation_value
                    logits = model.fc[2](torch.relu(feat))
                else:
                    feat = feat.view(feat.shape[0], -1)
                    feat[:, bad_neurons[:nn]] = activation_value
                    logits = model.fc[0](feat)
                logits_clean.append(logits.data.cpu().numpy())
                pred.append(logits.argmax(1).data.cpu().numpy())
                gnd.append(label.numpy())

            logits_clean = np.vstack(logits_clean)
            acc = np.mean(np.hstack(gnd) == np.hstack(pred)) * 100
            acc_all.append(acc)
            nn_all.append(int(nn / NUM_NEURONS * 100))
            kk = 0

        # % neurons where perf = P
        position = {}
        f = interpolate.interp1d(acc_all, nn_all)

        try:
            P = 20
            position[P] = f(P).item()
        except:
            position[P] = 0
        try:
            P = 40
            position[P] = f(P).item()
        except:
            position[P] = 0
        try:
            P = 60
            position[P] = f(P).item()
        except:
            position[P] = 0

        plt.plot(nn_all, acc_all)
        plt.plot(nn_all, 20 * np.ones((len(nn_all))))
        plt.plot(nn_all, 40 * np.ones((len(nn_all))), color="red")
        plt.ylabel("Accuracy")
        plt.xlabel("Percentage of neurons triggered in the layer")
        plt.title(f"Ablation for class {target}, Position={position[40]}")
        print(target, ":", position[20], position[40])
        return acc_all, nn_all, position

    ###########################################################################

    fig = plt.figure(figsize=[20, 20])
    acc_ablation = []
    position = []
    # ipdb.set_trace()
    M = feat.mean(0).max() * MULT_FACTOR
    print("Using activation value", M)

    for target in range(num_cls):
        ax = plt.subplot(4, 3, target + 1)
        bad_neurons = identify_bad_neurons(target, attribution)
        _acc, nn_all, pos = ablation_plot(feat_dl, bad_neurons, activation_value=M)
        position.append(pos)
        acc_ablation.append(_acc)

    pickle.dump(
        (acc_ablation, nn_all, position),
        open(os.path.join(curves_dir, model_name + ".pkl"), "wb"),
    )

    position = np.asarray(position)
    # print(f"Poisoned class is {position.argmin()} with {position.min()}")
    plt.savefig(os.path.join(curves_dir, model_name + ".jpg"))

