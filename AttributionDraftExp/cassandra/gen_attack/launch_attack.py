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

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=ImportWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="SourceChangeWarning")


if __name__ == "__main__":
    args = utils.argparser()
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

    load_function_name = "load_model_round" + str(args.round)
    load_function = getattr(utils, load_function_name)

    
    model_info, model, dataset = load_function(
        args.src_dir, model_name, args.device, meta_data
    )

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
        eps = 5000
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

