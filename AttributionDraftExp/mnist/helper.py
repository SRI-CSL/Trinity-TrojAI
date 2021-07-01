import skimage.io
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import torch
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class hook_fn_nn:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs = module_out.squeeze()
        
    def clear(self):
        self.outputs = []



def pre_process_image(img_path, noise=0):
    img = skimage.io.imread(img_path)
    # perform center crop to what the CNN is expecting 224x224
    # Convert to 3d if a 
    h, w, c = img.shape
    dx = int((w - 224) / 2)
    dy = int((w - 224) / 2)
    img = img[dy:dy+224, dx:dx+224, :]
    img = img.astype('float32') + np.random.randn(224, 224, 3) * noise

    # If needed: convert to BGR
#     img_orig = np.copy(img)
#     r = img[:, :, 0]
#     g = img[:, :, 1]
#     b = img[:, :, 2]
#     img = np.stack((b, g, r), axis=2)

    # perform tensor formatting and normalization explicitly
    # convert to CHW dimension ordering
    img = np.transpose(img, (2, 0, 1))
    # convert to NCHW dimension ordering
    img = np.expand_dims(img, 0)
    # normalize the image
    img = img - np.min(img)
    img = img / np.max(img)
    # convert image to a gpu tensor
    
    return img


def cross_validation_features(X, Y, verbose=True):
    # Cross validate on some features and binarized labels
    kfold = KFold(5, True, 1)
    # enumerate splits
    fold = 0
    auc_all = []
    ce_all = []
    for train, test in kfold.split(X):
        clf = RandomForestClassifier(max_depth=4, random_state=0)
        clf.fit(X[train], Y[train])
        scores = clf.predict_proba(X[test])[:, 1]
#         clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, C=32.0))
#         clf.fit(X[train], Y[train])
#         scores = clf.predict_proba(X[test])[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(Y[test], scores)
        auc = metrics.auc(fpr, tpr)
        ce,_ = cross_entropy(scores, Y[test])
        if verbose:
            print(f"AUC for fold {fold} is {auc}")
        auc_all.append(auc)
        ce_all.append(ce)
        fold += 1

    auc_all = np.nanmean(auc_all)
    ce_all = np.nanmean(ce_all)
    if verbose:
        print(f'Overall AUC is {auc_all}')
        print(f'Overall CE is {ce_all}')
    
    clf.fit(X, Y)
    return clf, auc_all


def cross_entropy(prob, labels):
    """
    Code to compute cross-entropy
    prob: probabilities from the model (numpy: Nx1)
    labels: ground-truth labels (numpy: Nx1) 
    """
    prob = torch.Tensor(prob).squeeze()
    labels = torch.Tensor(labels).squeeze()
    assert (
        prob.shape == labels.shape
    ), "Check size of labels and probabilities in computing cross-entropy"
    ce = F.binary_cross_entropy(prob, labels, reduction='none')
    return ce.mean(), ce


def load_model(src_dir, dataloader, model_id):
    model_filepath = os.path.join(src_dir, model_id, "model.pt")
    mlmodel = torch.load(model_filepath)
    mlmodel.eval()
    example_dir = os.path.join(src_dir, model_id, "example_data")
    filenames = os.listdir(example_dir)
    dataset = dataloader.TrojAIDatasetRound2(example_dir, filenames)
    return mlmodel, dataset