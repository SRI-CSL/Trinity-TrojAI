# Each learner function will have a train and test function


import numpy as np
import torch
import math
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.utils.data.sampler as sampler
import torch.nn as nn
from sklearn import metrics
from utils_nn import EarlyStopping, adjust_learning_rate
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from helper import cross_entropy, compute_metrics

# import ipdb


def check_for_cuda():
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"Device is {device}")
    return device


class mlp(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(mlp, self).__init__()
        self.output_size = output_size
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.fc(x)
        return out

    def initialize(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def predict_proba(self, x):
        # To get probablities for numpy array x
        x = torch.Tensor(x)
        x = x.to(self.fc.weight.device)
        out = self.forward(x)
        prob = nn.functional.softmax(out, dim=-1)
        prob = prob.data.cpu().numpy()
        return prob


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TrojNet(nn.Module):
    def __init__(self, args):
        super(TrojNet, self).__init__()
        self.args = args
        self.max_class = self.args.max_class
        self.embedding_dim = self.args.embedding_dim
        self.p = self.args.p
        self.stocastic = self.args.stocastic

        if self.args.arch == "conv" or self.args.arch == "conv_posenc":
            ninp = 100
            self.ninp = ninp
            dropout = 0.1
            self.pos_encoder = PositionalEncoding(ninp, dropout)

            self.encoder = nn.Sequential(
                nn.Conv1d(1, 4, kernel_size=13, stride=1),
                # nn.BatchNorm1d(4),
                nn.MaxPool1d(9, stride=2),
                nn.ReLU(),
                nn.Conv1d(4, self.embedding_dim, kernel_size=13, stride=1),
                # nn.BatchNorm1d(self.embedding_dim),
                nn.MaxPool1d(9, stride=2),
                nn.ReLU(),
            )
            feat_sz = 160
        elif self.args.arch == "transformer":
            ninp = 100
            nhid = self.args.nhid
            nhead = self.args.nhead
            nlayers = self.args.nlayers_tx
            dropout = self.args.p_tx
            self.ninp = ninp
            self.src_mask = None
            self.pos_encoder = PositionalEncoding(ninp, dropout)
            encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
            feat_sz = 100
        else:
            ValueError("Arch {} not supported".format(self.args.arch))

        self.drop = nn.Dropout(p=self.p)

        self.fc = nn.Sequential(nn.Linear(feat_sz, 20), nn.ReLU(), nn.Linear(20, 2))

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, data):
        _input, _valid, _arch = data
        _valid = _valid.float()
        _input = _input.float().unsqueeze(2)

        _enc = []
        for idx in range(self.max_class):
            if self.args.arch == "conv":
                src = _input[:, idx, ...]
                _enc_idx = self.encoder(src)
            elif self.args.arch == "conv_posenc":
                src = _input[:, idx, ...] * math.sqrt(self.ninp)
                # src = _input[:, idx, ...]
                src = self.pos_encoder(src)
                _enc_idx = self.encoder(src)
            elif self.args.arch == "transformer":
                if self.src_mask is None or self.src_mask.size(0) != _input.size(0):
                    device = _input.device
                    mask = self._generate_square_subsequent_mask(_input.size(0)).to(
                        device
                    )
                    self.src_mask = mask
                src = _input[:, idx, ...] * math.sqrt(self.ninp)
                # src = _input[:, idx, ...]
                src = self.pos_encoder(src)
                _enc_idx = self.transformer_encoder(src, self.src_mask)
            _enc.append(_enc_idx)
        feature = torch.stack(_enc, dim=1)

        feature = _valid.unsqueeze(-1).unsqueeze(-1) * feature

        pooled, _ = feature.max(dim=1)

        # pooled = emb.unsqueeze(-1)*pooled

        flatten = pooled.view(pooled.shape[0], -1)

        if self.stocastic:
            flatten = F.dropout(flatten, p=self.p)
        else:
            flatten = self.drop(flatten)

        out = self.fc(flatten)

        return out


class learner:
    def __init__(self, args):
        self.args = args
        # self.args.lr = 1e-5
        # self.args.epochs = 10
        self.args.device = check_for_cuda()

    def cross_validation(self, dataloader):
        print("Running internal cross-validation")

        np.random.seed(0)
        dataset = dataloader.dataset
        N = len(dataset)
        lr_values = [1e-2, 1e-3, 1e-5]
        nfolds = 3
        auc_all = np.zeros((nfolds, len(lr_values)))

        for fold in range(nfolds):
            all_idx = np.random.permutation(N)
            train_idx = all_idx[: int(N * 0.9)]
            test_idx = all_idx[len(train_idx) :]
            dl_train = DataLoader(
                dataset, batch_size=32, sampler=sampler.SubsetRandomSampler(train_idx)
            )
            dl_test = DataLoader(
                dataset, batch_size=32, sampler=sampler.SubsetRandomSampler(test_idx)
            )

            for ii, l in enumerate(lr_values):
                self.args.lr = l
                self.train(dl_train, num_classes=2)
                _, auc, _ = self.test(dl_test)
                auc_all[fold, ii] = auc
                print(f"AUC at fold {fold} at lr {l} is {auc}")

        auc_all = auc_all.mean(0)
        print("Done with internal CV")
        for i in range(len(lr_values)):
            print(f"AUC at lr {lr_values[i]} is {auc_all[i]}")
        lr = lr_values[auc_all.argmax()]
        print("Choosing lr", lr)

        return lr

    def train(
        self, dataloader, dataloader_val,
    ):
        self.classifier = TrojNet(self.args)
        self.classifier.to(self.args.device)
        self.early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, path=self.args.ckt_path
        )
        return self.trainer_sgd(dataloader, dataloader_val)

    def trainer_sgd(self, dataloader, dataloader_val, verbose=True):
        args = self.args
        pred_fn = self.classifier
        optimizer = torch.optim.Adam((pred_fn.parameters()), lr=args.lr)
        print(f"lr for training is {optimizer.param_groups[0]['lr']}")
        loss_fn = nn.CrossEntropyLoss()
        # loss_fn = nn.BCEWithLogitsLoss()
        device = args.device

        # training loop
        best_metric = {"acc": 0}
        # pred_fn.initialize()
        pred_fn.train()
        for epoch in range(args.epochs):
            train_loss = 0
            train_N = 0
            for step, data in enumerate(dataloader):
                # get the inputs; data is a list of [inputs, labels]
                _input, _valid, _label, _arch, _name = data
                _input = _input.to(device)
                _label = _label.to(device)
                _valid = _valid.to(device)
                optimizer.zero_grad()
                outputs = pred_fn((_input, _valid, _arch))

                # forward + backward + optimize
                # one_hot = torch.nn.functional.one_hot(labels, pred_fn.output_size)
                loss = loss_fn(outputs, _label)
                # loss = loss_fn(outputs, one_hot.float())
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(_input)
                train_N += len(_input)

                # if best_loss > loss.item():
                #     best_loss = loss.item()
                #     best_model = Model

            # if epoch % 50 == 0:
            #     adjust_learning_rate(optimizer)

            valid_acc, valid_auc, valid_ce, valid_info = self.test(dataloader_val)
            self.early_stopping(valid_auc, pred_fn)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            if verbose:
                print(
                    "Epoch {:d} | Train Loss {:.4f}".format(epoch, train_loss / train_N)
                )

        self.classifier.load_state_dict(torch.load(self.args.ckt_path))
        valid_acc, valid_auc, valid_ce, valid_scores = self.test(dataloader_val)

        return valid_acc, valid_auc, valid_ce, valid_info

    def test(self, dataloader):
        device = self.args.device
        pred = []
        self.classifier.eval()
        labels = []
        scores_all = []
        with torch.no_grad():
            for data in dataloader:
                _input, _valid, _label, _arch, _name = data
                _input = _input.to(device)
                _valid = _valid.to(device)
                if self.args.stocastic:
                    out_lst = []
                    pred_lst = []
                    for _ in range(self.args.T):
                        _out = self.classifier((_input, _valid, _arch))
                        out_lst.append(_out)
                        pred_lst.append(_out.argmax(1))
                    if self.args.hard:
                        out = torch.stack(pred_lst).float().mean(0)
                        pred.append((out > 0.5).long().data.cpu().numpy())
                    else:
                        out = torch.stack(out_lst).mean(0)
                        pred.append(out.argmax(1).data.cpu().numpy())
                else:
                    out = self.classifier((_input, _valid, _arch))
                    pred.append(out.argmax(1).data.cpu().numpy())
                labels.append(_label.data.numpy())
                scores_all.append(out.data.cpu().numpy())

        pred_labels = np.hstack(pred)
        labels = np.hstack(labels)
        if scores_all[0].ndim == 1:
            scores_all = np.hstack(scores_all)
        else:
            scores_all = np.vstack(scores_all)[:, 1]
        # acc = np.mean(pred_labels == labels) * 100
        # fpr, tpr, thresholds = metrics.roc_curve(labels, scores_all)
        # auc = metrics.auc(fpr, tpr)
        # ce = cross_entropy(scores_all, labels)
        auc, acc, ce = compute_metrics(scores_all, labels)
        return acc, auc, ce, (scores_all, labels)

