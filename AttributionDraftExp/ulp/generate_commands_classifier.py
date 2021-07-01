# Generate runs for testing

import os
import pickle
import pandas as pd

USE_KUBERNETES = True
PROJECT_NAME = "troj_ulp"

###################################################
# CONV
####################################################
DATASETS = ["mnist"]

# Dictionary of parameters sweep
LR = [1e2, 1e4]
MM = [10]
EPOCHS = [100]
###################################################

output_file = f"commands/commands_{DATASETS[0]}"
if USE_KUBERNETES:
    output_file += f"_kubernetes.txt"
else:
    output_file += f".txt"


N = 0
f = open(output_file, "w")

for DATASET in DATASETS:
    TAG = f"{DATASET}"
    for lr in LR:
        for epoch in EPOCHS:
            for m in MM:
                N += 1
                template = f"python run_trojan_detector.py --test_split 10 --val_split 10 --nfolds 5 --M {m} --epochs {epoch} --lr {lr} --use_wandb --tags {TAG} --dataset {DATASET} --patience 10 --project_name {PROJECT_NAME}"
                if USE_KUBERNETES:
                    template += f" --use_kubernetes"
                f.write(template + "\n")

f.close()
print(f"Written {N} commands to {output_file}")
