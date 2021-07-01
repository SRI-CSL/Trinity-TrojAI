import numpy as np
import torch
import utils
import pandas as pd
import dataloader
import os
import warnings
import shutil
import attacks
import skimage.io
import torch.nn as nn
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=ImportWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="SourceChangeWarning")

class MNISTdataset(Dataset):
    def __init__(self, root_dir, filenames, csv_file):
        self.root_dir = root_dir

        df = pd.read_csv(csv_file, header=None)

        self.filenames = []
        self.truelabels = []
        for f in filenames:
            label = df[df[0]==f][1].to_list()
            if len(label):
                self.filenames.append(f)
                self.truelabels.append(label[0])
        
    def __len__(self):
        return len(self.truelabels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        label = self.truelabels[idx]
        image = self.pre_process_image(img_path)
        return image, label, img_path

    @staticmethod
    def pre_process_image(img_path):
        img = skimage.io.imread(img_path)
        img = img[None, ...]/255.
        return img.astype("float32")

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        out = x*255.
        out[out>255] = 255.
        return self.model(out)

def load_function(src_dir, model_name, device, meta_data):
    model_info = {}
    model_dir = os.path.join(src_dir)
    model_info["model_dir"] = os.path.join(model_dir, model_name)
    model_info["image_dir"] = os.path.join(src_dir, "example")
    model_info["csv_file"] = os.path.join(model_info["image_dir"], "labels.txt")

    meta_data_row=(meta_data.loc[ meta_data["model_name"] == model_name]).iloc[0]

    model_info["model_img_size"] = 28
    model_info["num_classes"] = meta_data_row["number_classes"]
    print(f'Number of classes:{model_info["num_classes"]}')

    fs = []
    for (dirpath, dirnames, filenames) in os.walk(model_info["image_dir"]):
        fs.extend(filenames)
        break


    model_info["file_list"] = fs

    #model_info["file_list"] = [
    #            it.strip().split(",")[0:2]
    #            for it in open(os.path.join(model_info["image_dir"], "data.csv"))
    #        ][1:]
    model_info["model_filepath"] = os.path.join(model_info["model_dir"], "model.pt.1")
    model_info["label"] = meta_data_row["poisoned"]

    dataset = MNISTdataset(model_info["image_dir"], model_info["file_list"], model_info["csv_file"])

    # print(f"Number of benign images in {model_name} are {len(model_info["file_list"])}")
    model = torch.load(model_info["model_filepath"], map_location=device)
    model.eval()
    print(f"Loaded model {model_name} with ground-truth {model_info['label']}")

    return model_info, model, dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="univ_pert_ut_l1")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint_dir", type=str, default="/data/isur/0.Work/trojAI/trinityTrojAI/users/ksikka/checkpoints_path_bkp/unv_pert_feat/mnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="id-00000021")
    # For attacked targets
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument(
        "--attack_type",
        type=str,
        default="univ_pert_ut_1",
        choices=["pgd_t", "univ_pert_ut_1", "univ_pert_ut_2", "univ_pert_ut_n", "fgsm_ut"],
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default="/data/ksikka/projects_2020/trojAI/mnist-dataset/",
    )

    args = parser.parse_args()
    args.device = utils.check_for_cuda()

    meta_data = pd.read_csv(os.path.join(args.src_dir, "METADATA.csv"))

    env_name = args.env_name
    model_name = args.model_name
    target_class = args.target_class
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # checkpoint_dir will save the images and numpy files
    checkpoint_dir = os.path.join(args.checkpoint_dir, env_name, model_name)
    if os.path.exists(os.path.join(checkpoint_dir, "noise.png")):
        print ('{} done'.format(model_name)); exit()
    
    #this will erase the model directory when you are adding targets to it - MKY
    #if os.path.isdir(checkpoint_dir):
    #    print("\t%s already exists. Deleting..." % checkpoint_dir)
    # try:
    #     shutil.rmtree(checkpoint_dir)
    #     # shutil.rmtree(pth.join(args.logdir, args.env_name))
    # except:
    #     pass
    os.makedirs(checkpoint_dir, exist_ok=True)

    # load_function_name = "load_model_round" + str(args.round)
    # load_function = getattr(utils, load_function_name)

    
    model_info, model_mnist, dataset = load_function(
        args.src_dir, model_name, args.device, meta_data
    )

    model = ModelWrapper(model_mnist)

    # Confirm accuracy of the model (works with reduced batch-size)
    # print("Confirming pre-liminary accuracy on the given data")
    # utils.compute_accuracy(model, trainLoader, args.device)

    # Using full batch-size since batching is also done inside ART
    trainLoader = torch.utils.data.DataLoader(
        dataset,
        batch_size=len(model_info["file_list"]),
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )


    # TODO: Pass parameters for different attack for logging
    if args.attack_type == "pgd_t":
        dest_images = attacks.attack_pgd_targeted(
            trainLoader, model, model_info, args, checkpoint_dir
        )
    elif args.attack_type == "univ_pert_ut_1":
        eps = 25
        dest_images = attacks.attack_universal_perturbations_nontargeted(
            trainLoader, model, model_info, args, checkpoint_dir, 1, eps
        )
    elif args.attack_type == "univ_pert_ut_2":
        dest_images = attacks.attack_universal_perturbations_nontargeted(
            trainLoader, model, model_info, args, checkpoint_dir, 2, eps
        )
    elif args.attack_type == "univ_pert_ut_n":
        dest_images = attacks.attack_universal_perturbations_nontargeted(
            trainLoader, model, model_info, args, checkpoint_dir, np.inf, eps
        )
    elif args.attack_type == "fgsm_ut":
        dest_images = attacks.attack_FGSM_nontargeted(
            trainLoader, model, model_info, args, checkpoint_dir
        )

    # Generate new dataset after the attack as well as features for post-analysis
    
    #dataset = dataloader.TrojAIDatasetNumpy(dest_images, model_info["file_list"])
    #trainLoader = torch.utils.data.DataLoader(
    #    dataset,
    #    batch_size=args.batch_size,
    #    num_workers=args.num_workers,
    #    pin_memory=True,
    #    shuffle=False,
    #)

    #img_path, predicted_labels, attacked_acc = utils.compute_accuracy(
    #    model, trainLoader, args.device
    #)
    #with open(os.path.join(dest_images, "predicted_labels.txt"), "w") as f:
    #    for i in range(len(img_path)):
    #        f.write(f"{img_path[i]},{predicted_labels[i]}\n")

    #with open(os.path.join(dest_images, "stats.txt"), "a") as f:
    #    f.write(f"Accuracy after attack is {attacked_acc}\n")
    #    hist = np.histogram(predicted_labels, bins=model_info["num_classes"])
    #    for i in range(model_info["num_classes"]):
    #        f.write(
    #            f"Percentage of example classified in class {i} is {100 * (np.asarray(predicted_labels) == i).astype('float32').mean()}\n"
    #        )


    #feat = utils.get_last_layer(trainLoader, model, args).cpu().numpy()
    #np.save(os.path.join(dest_images, "feat_attacked"), feat)

