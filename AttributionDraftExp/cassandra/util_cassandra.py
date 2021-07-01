import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision.models import MobileNetV2, mobilenet_v2

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from wrapper_nn import EarlyStopping
from helper import cross_entropy, compute_metrics


def initialize_wandb(args):
    # Initialize wandb
    if args.use_wandb:
        import wandb

        wandb.init(
            project="cassendra",
            config=vars(args),
            sync_tensorboard=True,
            name=args.env_name,
            # entity="ksikka",
            tags=args.tags.split(","),
            dir="/home/isur/troj",
        )
    else:
        wandb = None
    return wandb

def init_data_path_cassendra(dataset_name, use_holdout, use_kubernetes=False):
    holdout_dir = None
    holdout_metadata = None
    if not use_kubernetes:
        troj_root = "/data/ksikka/projects_2020/trojAI"
        cvpr_root = (
            "/data/isur/0.Work/trojAI/trinityTrojAI/users/ksikka/checkpoints_path_bkp/unv_pert_feat"
        )
    else:
        Exception("Set paths")
        # troj_root = "/data"5
        # cvpr_root = "/data/k8s/trinityTrojAI/users/ksikka/cvpr_exp/"

    if dataset_name == "round2":
        model_dir = os.path.join(troj_root, "round2-dataset-train")
        meta_data = pd.read_csv(os.path.join(model_dir, "METADATA.csv"))
        feature_dir = os.path.join(cvpr_root, "round2")
    elif dataset_name == "round3":
        model_dir = os.path.join(troj_root, "round3-dataset-train")
        meta_data = pd.read_csv(os.path.join(model_dir, "METADATA.csv"))
        feature_dir = os.path.join(cvpr_root, "round3")
    elif dataset_name == "round1":
        model_dir = os.path.join(troj_root, "round1-dataset-train", "models")
        meta_data = pd.read_csv(
            os.path.join(troj_root, "round1-dataset-train", "METADATA.csv")
        )
        feature_dir = os.path.join(cvpr_root, "round1")
        if use_holdout:
            holdout_dir = os.path.join(cvpr_root, "round1_holdout")
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
        feature_dir = os.path.join(cvpr_root, "mnist")

    return model_dir, meta_data, feature_dir, holdout_dir, holdout_metadata


class cassandra_ds(Dataset):
    def __init__(
        self,
        feature_dir,
        meta_data,
        models_list,
        args
    ):
        self.feature_dir = feature_dir
        self.meta_data = meta_data
        self.models_list = models_list
        self.args = args

        self.feat_types = args.feat_types
        self.crop_stride = args.crop_stride
        self.crop_size = args.crop_size

        if args.dataset == "mnist" and args.scale_mnist == False:
            self.crop_stride = 1
            self.crop_size = 10
        else:
            self.crop_size = 50

        if not args.use_blank_noise:
            for feat_type in self.feat_types:
                self.models_list = [en for en in self.models_list 
                                    if os.path.exists(os.path.join(self.feature_dir, feat_type, en, 'noise.png'))]

        self.all_feats = []
        self.all_labels = []
        for model_name in tqdm(self.models_list, desc='Loading data'):
            feats = []
            poisoned = self.meta_data[self.meta_data.model_name == model_name].iloc[0].poisoned
            for feat_type in self.feat_types:
                feats.append(self.get_feat(model_name, feat_type))
            self.all_labels.append(poisoned)
            self.all_feats.append(feats)

    def __len__(self):
        return len(self.models_list)
    
    def get_max_energy_crop(self, noise):
        img = np.array(Image.fromarray(noise).convert('L'))
        row,col = img.shape
        stride = self.crop_stride
        crop = self.crop_size

        max_energy = 0
        max_r,max_c = 0,0
        for _r in range(0, row-crop, stride):
            for _c in range(0, row-crop, stride):
                _e = img[_r:_r+crop, _c:_c+crop].sum()
                if _e > max_energy:
                    max_energy = _e
                    max_r,max_c = _r,_c
        return img[max_r:max_r+crop, max_c:max_c+crop]

    def get_feat(self, model_name, env):
        model_dir = os.path.join(self.feature_dir, env, model_name)
        noise_file = os.path.join(model_dir, 'noise.png')
        if os.path.exists(noise_file):
            noise_img = Image.open(noise_file)
            if self.args.dataset == "mnist" and self.args.scale_mnist:
                noise_img = noise_img.resize((224,224)).convert('RGB')
            noise = np.array(noise_img)
            f = open(os.path.join(model_dir, 'stats.txt'), "r")
            fl =f.readlines()
            fr = float(fl[0].strip().split(' ')[-1])
            f.close()
        else:
            noise = np.zeros((224,224,3), dtype="uint8")
            if self.args.dataset == "mnist" and not self.args.scale_mnist:
                noise = np.zeros((28,28), dtype="uint8")
            fr = 0.

        max_eng_crop = self.get_max_energy_crop(noise)
        # noise = np.transpose(noise, (2, 0, 1))

        if noise.ndim == 2:
            noise = noise[..., None]
        
        return noise/255., max_eng_crop/255., fr
    
    def __getitem__(self, idx):
        # ret = []
        # model_name = self.models_list[idx]
        # poisoned = self.meta_data[self.meta_data.model_name == model_name].iloc[0].poisoned
        # for feat_type in self.feat_types:
        #     ret.append(self.get_feat(model_name, feat_type))
        # return ret, poisoned, model_name
        return self.all_feats[idx], self.all_labels[idx], self.models_list[idx]


class ModdedLeNet5Net(nn.Module):
    def __init__(self, channels=1):
        super(ModdedLeNet5Net, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 6, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU()
        )
        self.fc = nn.Linear(120, 84)

    def features(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

class CassendraFeatNet(nn.Module):
    def __init__(self, args):
        super(CassendraFeatNet, self).__init__()
        self.args = args
        self.use_fr = args.use_fr
        if args.dataset == "mnist" and args.scale_mnist == False:
            self.cnn_encoder = ModdedLeNet5Net()
            self.energy_encoder = nn.Sequential(
                                        nn.Linear(100,120),
                                        nn.ReLU(),
                                        nn.Linear(120,50),
                                        nn.ReLU(),
                                        nn.Linear(50,16),
                                    )
            self.feat_sz = ((84+16+1) if args.use_fr else (84+16))
        else:
            self.cnn_encoder = mobilenet_v2(pretrained=True)
            self.energy_encoder = nn.Sequential(
                                        nn.Linear(2500,3126),
                                        nn.ReLU(),
                                        nn.Linear(3126,1024),
                                        nn.ReLU(),
                                        nn.Linear(1024,512),
                                        nn.ReLU(),
                                        nn.Linear(512,256)
                                    )
            self.feat_sz = ((1280+256+1) if args.use_fr else (1280+256))
        
    def forward(self, data):
        noise, energy, fr = data
        # fr = fr.to(self.args.device)
        if self.use_fr:
            # import ipdb; ipdb.set_trace()
            # l1_norm = np.linalg.norm(np.linalg.norm(noise, axis=(3), ord=1), axis=(1,2), ord=1)
            # l1_norm = np.linalg.norm(np.linalg.norm(noise, axis=(1,2), ord=1), axis=(-1), ord=1)
            l1_norm = np.linalg.norm(noise.view(noise.shape[0], -1), axis=-1, ord=1)
            # y = l1_norm/fr
            # y = l1_norm/(fr+1e-7)
            y = l1_norm/(fr+.1)
            y = y.float().to(self.args.device)

        noise = noise.to(self.args.device)
        energy = energy.to(self.args.device)
        noise = noise.float()
        energy = energy.float()
        # fr = fr.float()
        
        x = self.cnn_encoder.features(noise.permute(0, 3, 1, 2))
        if self.args.dataset == "mnist" and self.args.scale_mnist == False:
            emb_cnn = x
        else:
            emb_cnn = F.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        # print (emb_cnn.shape)
        emb_energy = self.energy_encoder(energy.reshape(energy.shape[0], -1))
        # print (emb_energy.shape)

        if self.use_fr:
            ret = torch.cat((emb_cnn, emb_energy, y.unsqueeze(-1)), -1)
        else:
            ret = torch.cat((emb_cnn, emb_energy), -1)

        return ret
    
class CassendraNet(nn.Module):
    def __init__(self, args):
        super(CassendraNet, self).__init__()
        self.args = args
        self.feat_types = args.feat_types
        self.use_fr = args.use_fr
        self.feat_extract = []
        for _ in self.feat_types:
            self.feat_extract.append(CassendraFeatNet(args))
        self.feat_extract = nn.ModuleList(self.feat_extract)
        
        # self.feat_sz = len(self.feat_types)*(1537 if args.use_fr else 1536)
        self.feat_sz = len(self.feat_types)*self.feat_extract[0].feat_sz
        self.fc = nn.Linear(self.feat_sz,2)
        
    def forward(self, data):
        feats = torch.cat([e(f) for e, f in zip(self.feat_extract, data)], -1)
        # print (feats.shape)
        out = self.fc(feats)

        return out


class learner_cassendra:
    def __init__(self, args):
        self.args = args

    def train(
        self, dataloader, dataloader_val,
    ):
        self.classifier = CassendraNet(self.args)
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
                _input, _label, _name = data
                # _input = _input.to(device)
                _label = _label.long().to(device)
                optimizer.zero_grad()
                outputs = pred_fn(_input)

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
                _input, _label, _name = data
                # _input = _input.to(device)
                _label = _label.long()
                out = self.classifier(_input)
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

