import sys

sys.path.append("../troj_round2")
sys.path.append("../troj_round2/utils_temporal_nn")
from wrapper_nn import check_for_cuda
from utils_nn import EarlyStopping
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from helper import compute_metrics, hook_fn_nn


def noise_param(args):
    if args.dataset == "mnist":
        W, H = 28, 28
        L = 1
        is_normalized = False
        num_classes = 10
    elif args.dataset == "round1":
        W, H = 224, 224
        L = 3
        is_normalized = True
        num_classes = 5
    elif args.dataset in ["round2", "round3"]:
        W, H = 224, 224
        L = 3
        is_normalized = True
        num_classes = int(20 / args.M)
    return W, H, L, num_classes, is_normalized


def get_model_src(args):
    if not args.use_kubernetes:
        root = "/home/ksikka/data/mount/rebel/ksikka/"
    else:
        root = "/data/"

    if args.dataset == "mnist":
        model_src = os.path.join(root, "mnist-dataset")
    elif args.dataset == "round1":
        model_src = os.path.join(root, "round1-dataset-train", "models")
    elif args.dataset == "round2":
        model_src = os.path.join(root, "round2-dataset-train")
    elif args.dataset == "round3":
        model_src = os.path.join(root, "round3-dataset-train", "models")
    else:
        model_src = os.path.join(root, f"{args.dataset}-dataset-train", "models")
    return model_src


class troj_ulp_model(torch.nn.Module):
    def __init__(self, args):
        super(troj_ulp_model, self).__init__()
        self.args = args

        # Need this info
        W, H, L, num_classes, is_normalized = noise_param(args)
        M = args.M

        self.X = torch.nn.Parameter(
            torch.rand((M, L, W, H), requires_grad=True, device=args.device)
        )
        if not is_normalized:
            self.X.data *= 255.0
        self.W = torch.nn.Parameter(
            torch.randn((M * num_classes, 2), requires_grad=True, device=args.device)
        )
        torch.nn.init.xavier_normal_(self.W)

        self.b = torch.nn.Parameter(
            torch.zeros((2,), requires_grad=True, device=args.device)
        )
        self.is_normalized = is_normalized


class ulp_learner:
    def __init__(self, args):
        self.args = args
        # self.args.lr = 1e-5
        # self.args.epochs = 10
        self.args.device = check_for_cuda()

    def load_model(self, model_name):
        model_src = get_model_src(self.args)
        if self.args.dataset != "mnist":
            cnn = torch.load(
                os.path.join(model_src, model_name, "model.pt"),
                map_location=self.args.device,
            )
        else:
            cnn = torch.load(
                os.path.join(model_src, model_name, "model.pt.1"),
                map_location=self.args.device,
            )
        return cnn

    def compute_logit(self, cnn, X, W, b):
        model_type = type(cnn).__name__
        if self.args.dataset == "round1":
            output = cnn(X).view(1, -1)
            logit = torch.matmul(output, W) + b
        if self.args.dataset == "round2" or self.args.dataset == "round3":
            output = cnn(X).view(1, -1)
            if output.ndim > 2:
                output = output.mean(1).mean(1)
            output = torch.nn.functional.adaptive_avg_pool1d(
                output.unsqueeze(0), 20
            ).squeeze(0)
            logit = torch.matmul(output, W) + b
        if self.args.dataset == "mnist":
            hook_fn_logit_layer = hook_fn_nn()
            if model_type != "ModdedBadNetExample":
                cnn.fc[2].register_forward_hook(hook_fn_logit_layer)
            else:
                cnn.fc[0].register_forward_hook(hook_fn_logit_layer)
            cnn(X)
            output = hook_fn_logit_layer.outputs.view(1, -1)
            logit = torch.matmul(output, W) + b
        assert logit.shape[-1] == 2, "Check shape of logit"
        return logit

    def train(self, train_models, train_labels, val_models, val_labels, verbose=True):
        args = self.args
        device = args.device
        self.early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, path=self.args.ckt_path
        )
        self.classifier = troj_ulp_model(args)
        X, W, b = self.classifier.X, self.classifier.W, self.classifier.b

        optimizerX = optim.SGD(params=[X], lr=args.lr)
        # optimizerX = optim.Adam(params=[X], lr=args.lr)
        optimizerWb = optim.Adam(params=[W, b], lr=args.lr_W)
        batchsize = args.batch_size
        REGULARIZATION = args.reg

        cross_entropy = torch.nn.CrossEntropyLoss()

        Xgrad = list()
        Wgrad = list()
        bgrad = list()

        # training loop
        for epoch in range(args.epochs):
            train_loss = 0
            train_N = 0
            randind = np.random.permutation(len(train_models))
            train_models = np.asarray(train_models)[randind]
            train_labels = np.asarray(train_labels)[randind]
            for i, model in enumerate(tqdm(train_models)):
                cnn = self.load_model(model)
                # cnn.to(args.device)
                cnn.eval()
                label = np.array([train_labels[i]])
                logit = self.compute_logit(cnn, X, W, b)
                y = torch.from_numpy(label).type(torch.LongTensor).to(device)
                reg_loss = REGULARIZATION * (
                    torch.sum(torch.abs(X[:, :, :, :-1] - X[:, :, :, 1:]))
                    + torch.sum(torch.abs(X[:, :, :-1, :] - X[:, :, 1:, :]))
                )

                loss = cross_entropy(logit, y) + reg_loss

                optimizerWb.zero_grad()
                optimizerX.zero_grad()

                loss.backward()

                if np.mod(i, batchsize) == 0 and i != 0:
                    Xgrad = torch.stack(Xgrad, 0)

                    X.grad.data = Xgrad.mean(0)

                    optimizerX.step()

                    if self.classifier.is_normalized:
                        X.data[X.data < 0.0] = 0.0
                        X.data[X.data > 1] = 1.0
                    else:
                        X.data[X.data < 0.0] = 0.0
                        X.data[X.data > 255.0] = 255.0

                    Xgrad = list()
                    Wgrad = list()
                    bgrad = list()

                Xgrad.append(X.grad.data)
                optimizerWb.step()
                train_loss += loss.item()
                train_N += 1

            # Validation
            valid_acc, valid_auc, valid_ce, valid_info = self.test(
                val_models, val_labels
            )
            self.early_stopping(valid_auc, self.classifier)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            if verbose:
                print(
                    "Epoch {:d} | Train Loss {:.4f}".format(epoch, train_loss / train_N)
                )

        self.classifier.load_state_dict(torch.load(self.args.ckt_path))
        valid_acc, valid_auc, valid_ce, valid_scores = self.test(val_models, val_labels)

        return valid_acc, valid_auc, valid_ce, valid_info

    def test(self, val_models, val_labels):
        with torch.no_grad():
            pred = list()
            device = self.args.device
            for i, model in enumerate(tqdm(val_models)):
                cnn = self.load_model(model)
                # cnn.fc[2].register_forward_hook(hook_fn_logit_layer)
                cnn.eval()
                cnn.to(device)
                logit = self.compute_logit(
                    cnn, self.classifier.X, self.classifier.W, self.classifier.b
                )
                pred.append(logit.data.cpu().numpy())
                # pred.append(torch.argmax(logit, 1))
            scores_all = np.vstack(pred)[:, 1]
            auc, acc, ce = compute_metrics(scores_all, val_labels)
            return acc, auc, ce, (scores_all, val_labels)