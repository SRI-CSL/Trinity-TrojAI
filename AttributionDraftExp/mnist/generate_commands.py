# Generate runs for testing

import os
import pickle
import pandas as pd

# src = "/data/ksikka/mount/rebel/ksikka/mnist-dataset"
src = "/data/mnist-dataset"


f = open("commands_1.sh", "w")
meta_data = pd.read_csv(os.path.join(src, "METADATA.csv"))
# meta_data = meta_data[meta_data.model_architecture == "ModdedLeNet5Net"]
model_names = meta_data.model_name.tolist()

# model_names = list(
#     filter(
#         lambda x: not os.path.exists(
#             os.path.join("curve_features_gradxact", x + ".pkl")
#         ),
#         model_names,
#     )
# )
model_names = model_names[430:]
print("len of model_names", len(model_names))

for it in model_names:
    template = f"python get_attribution_features_mnist.py {it}"
    f.write(template + "\n")

f.close()
