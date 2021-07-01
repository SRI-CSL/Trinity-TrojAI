# Generate runs for testing

import os
import pickle
import pandas as pd

ROUND = 3
# ATTRIB_FN = "IG"
ATTRIB_FN = "GradxAct"


model_src = f"/data/ksikka/mount/rebel/ksikka/round{ROUND}-dataset-train/models"
dest = f"/data/ksikka/mount/rebel/ksikka/k8s/trinityTrojAI/users/ksikka/cvpr_exp/troj_round2/round3"

# model_src = f"/data/round{ROUND}-dataset-train/models"
# dest = f"round{ROUND}"


squeezenet_list = ["squeezenetv1_1", "squeezenetv1_0"]
densenet_list = ["densenet121", "densenet169", "densenet201"]
shufflenet_list = ["shufflenet1_0", "shufflenet1_5", "shufflenet2_0"]

if ROUND == 2:
    meta_data_path = os.path.join(model_src, "METADATA.csv")
else:
    meta_data_path = os.path.join(os.path.dirname(model_src), "METADATA.csv")
meta_data = pd.read_csv(meta_data_path)

# meta_data = meta_data[meta_data.model_architecture.isin(densenet_list)]
path_attribution_features = os.path.join(dest, "attribution_features_gradxact")
path_curve_features = os.path.join(dest, "curve_features_gradxact")

f = open(f"commands_round{ROUND}_1.sh", "w")
model_names = meta_data.model_name.tolist()
model_names = list(
    filter(
        lambda x: os.path.exists(os.path.join(path_attribution_features, x + ".npz")),
        model_names,
    )
)
model_names = list(
    filter(
        lambda x: not os.path.exists(os.path.join(path_curve_features, x + ".pkl")),
        model_names,
    )
)
"model_names = model_names[260:]
print("len of model_names", len(model_names))

for it in model_names:
    template = f"python get_attribution_features_trojai.py --model_name {it} --model_dir {model_src} --meta_data {meta_data_path} --attributions_dir {path_attribution_features} --results_dir {path_curve_features} --attribution_fn {ATTRIB_FN} --round {ROUND}"
    f.write(template + "\n")

f.close()
