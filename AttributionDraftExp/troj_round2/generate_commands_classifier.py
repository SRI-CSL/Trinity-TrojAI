# Generate runs for testing

import os
import pickle
import pandas as pd

USE_KUBERNETES = True
NUM_ENSEMBLE = 5
PROJECT_NAME = "troj_gradxact"

###################################################
# CONV
####################################################
DATASET = "mnist"
ARCH = "conv"

# Dictionary of parameters sweep
LR = [1e-2, 1e-3, 1e-4]
EPOCHS = [100, 200, 300]
P_TX = [0.8]
NLAYERS_TX = [2]
NHEADS = [2]
N_HID = [16]
###################################################
"""

###################################################
# Tx
####################################################
DATASET = "mnist"
ARCH = "transformer"

# Dictionary of parameters sweep
# LR = [1e-2, 1e-3, 1e-4]
LR = [1e-3]
EPOCHS = [100, 200, 300]
# P_TX = [0.1, 0.5, 0.8]
P_TX = [0.8]
NLAYERS_TX = [2, 4]
NHEADS = [2, 4]
# N_HID = [16, 32, 64]
N_HID = [16]
###################################################
"""

output_file = f"commands/commands_{DATASET}_{ARCH}_E{NUM_ENSEMBLE}"
if USE_KUBERNETES:
    output_file += f"_kubernetes.txt"
else:
    output_file += f".txt"


N = 0
f = open(output_file, "w")

# for DATASET in ["mnist", "round1", "round2"]:
for DATASET in ["mnist"]:
    TAG = f"{DATASET}_{ARCH}_GradxAct"
    for lr in LR:
        for epoch in EPOCHS:
            for nlayers in NLAYERS_TX:
                for nhead in NHEADS:
                    for n_hid in N_HID:
                        for p_tx in P_TX:
                            N += 1
                            template = f"python run_trojan_detector.py --test_split 10 --val_split 10 --nfolds 5 --arch {ARCH} --epochs {epoch} --lr {lr} --p_tx {p_tx} --nhead {nhead} --nlayers_tx {nlayers} --use_wandb --tags {TAG} --dataset {DATASET} --patience 20 --num_ensemble {NUM_ENSEMBLE} --nhid {n_hid} --project_name {PROJECT_NAME}"
                            if USE_KUBERNETES:
                                template += f" --use_kubernetes"
                            f.write(template + "\n")

f.close()
print(f"Written {N} commands to {output_file}")
