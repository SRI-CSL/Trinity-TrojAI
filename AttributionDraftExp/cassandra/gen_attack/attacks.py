from tqdm import tqdm
from art.attacks.evasion import (
    ProjectedGradientDescentPyTorch,
    UniversalPerturbation,
    FastGradientMethod,
)
from art.estimators.classification import PyTorchClassifier
import torch.optim as optim
import torch
import utils
import torch.nn as nn
import numpy as np
import skimage.io
import os
from skimage import img_as_ubyte

######################################################
def attack_pgd_targeted(dataloader, model, model_info, args, checkpoint_dir):
    """
    PGD attack
    Need to change a few things
    """
    device = args.device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    img_size = model_info["model_img_size"]
    n_classes = model_info["num_classes"]

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        clip_values=(0.0, 1.0),
        optimizer=optimizer,
        input_shape=(img_size, img_size),
        nb_classes=n_classes,
        device_type=device,
    )
    attack = ProjectedGradientDescentPyTorch(
        estimator=classifier,
        norm=1,
        eps=500,
        eps_step=5,
        max_iter=100,
        targeted=True,
        num_random_init=0,
        batch_size=args.batch_size,
        random_eps=False,
    )

    # Launching a targeted attack
    t = args.target_class
    print(f"Launching attack for target {t}")
    dest_images = os.path.join(checkpoint_dir, f"target_{t}")
    os.makedirs(dest_images, exist_ok=True)

    for data in tqdm(dataloader):
        sample, label, img_path = data

        # Launch attack
        target = np.full_like(label, t)
        target_labels = (np.arange(n_classes) == target[:, None]).astype("float32")

        sample_adv = attack.generate(x=sample, y=target_labels)
        prediction = np.argmax(classifier.predict(sample_adv), axis=1)
        
        # Code to save these images
        img_path = [it.split("/")[-1] for it in img_path]

        #s_indexes = sorted(range(len(img_path)), key=lambda k: img_path[k])
        #[print(img_path[i]) for i in s_indexes]
        #print(label[s_indexes])
        #print(prediction[s_indexes])
        
        for i in range(len(sample_adv)):
            np.save(os.path.join(dest_images, img_path[i].replace("png", "npy")), sample_adv[i])

            _img = sample_adv[i].transpose(1, 2, 0)
            skimage.io.imsave(
                os.path.join(dest_images, img_path[i]), img_as_ubyte(_img)
            )


        #with open(os.path.join(dest_images, "stats.txt"), "w") as f:
        #    f.write(f"Fooling-rate was nan\n")

    return dest_images


######################################################
def attack_universal_perturbations_nontargeted(
    dataloader, model, model_info, args, checkpoint_dir, norm, eps
):
    """
    UAP nontargeted attack 
    """
    device = args.device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    img_size = model_info["model_img_size"]
    n_classes = model_info["num_classes"]

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        clip_values=(0.0, 1.0),
        optimizer=optimizer,
        input_shape=(img_size, img_size),
        nb_classes=n_classes,
        device_type=device,
    )

    attack = UniversalPerturbation(
        classifier=classifier,
        attacker="fgsm",
        attacker_params={"eps": eps, "batch_size": 32, "norm": norm},
        delta=0.25,
        max_iter=20,
        #max_iter=3,
        eps=eps,
        norm=norm,
    )


    # Launching a non-targeted attack
    # t = args.target_class
    print(f"Launching univ-pert nontargeted attack")
    #dest_images = os.path.join(checkpoint_dir, args.model_name)
    dest_images = checkpoint_dir
    os.makedirs(dest_images, exist_ok=True)

    # Running over the entire-batch to compute a universal perturbation
    for data in tqdm(dataloader):
        sample, label, img_path = data
        sample = sample.float()
        # Launch attack
        sample_adv = attack.generate(x=sample.cpu())

        # Code to save these images
        img_path = [it.split("/")[-1] for it in img_path]
        for i in range(len(sample_adv)):
            _img = sample_adv[i].transpose(1, 2, 0)
            skimage.io.imsave(
                os.path.join(dest_images, img_path[i]), img_as_ubyte(_img)
            )

        # Also save noise image for universal attack
        _img = attack.noise.squeeze(0).transpose(1, 2, 0)
        #import ipdb; ipdb.set_trace()
        skimage.io.imsave(os.path.join(dest_images, "noise.png"), img_as_ubyte(_img))

    with open(os.path.join(dest_images, "stats.txt"), "w") as f:
        f.write(f"Fooling-rate was {attack.fooling_rate}\n")

    return dest_images


######################################################
def attack_FGSM_nontargeted(dataloader, model, model_info, args, checkpoint_dir):
    """
    FGSM attack
    """
    device = args.device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    img_size = model_info["model_img_size"]
    n_classes = model_info["num_classes"]

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        clip_values=(0.0, 1.0),
        optimizer=optimizer,
        input_shape=(img_size, img_size),
        nb_classes=n_classes,
        device_type=device,
    )

    # attack = FastGradientMethod(estimator=classifier, batch_size=args.batch_size)
    attack = FastGradientMethod(estimator=classifier, batch_size=args.batch_size)

    # Launching a non-targeted attack
    # t = args.target_class
    print(f"Launching FGSM nontargeted attack")
    dest_images = os.path.join(checkpoint_dir, args.model_name)
    os.makedirs(dest_images, exist_ok=True)

    # Running over the entire-batch to compute a universal perturbation
    for data in tqdm(dataloader):
        sample, label, img_path = data
        sample = sample.to(device)
        # Launch attack
        sample_adv = attack.generate(x=sample.cpu())

        # Code to save these images
        img_path = [it.split("/")[-1] for it in img_path]

        for i in range(len(sample_adv)):
            _img = sample_adv[i].transpose(1, 2, 0)
            skimage.io.imsave(
                os.path.join(dest_images, img_path[i]), img_as_ubyte(_img)
            )

    with open(os.path.join(dest_images, "stats.txt"), "w") as f:
        f.write(f"Fooling-rate was nan\n")

    return dest_images
