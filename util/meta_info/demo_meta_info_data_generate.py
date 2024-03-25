# Author: ray
# Date: 3/25/24
# Description:
import json
import os.path

import numpy as np

from util.common import get_proje_root_path

proje_root = get_proje_root_path()
meta_info_folder = os.path.join(proje_root, "statistic/meta_info")

if not os.path.exists(meta_info_folder):
    os.makedirs(meta_info_folder)
    print(f"Directory '{meta_info_folder}' was created.")
else:
    print(f"Directory '{meta_info_folder}' already exists.")

input_list = list(range(96, 5000, 96))
label_list = list(range(48, 5000, 48))
pred_list = list(range(96, 5000, 96))

counter = 0
meta_info_list = []

for i in input_list:
    for j in label_list:
        for k in pred_list:
            mse = np.random.uniform(0.001, 5)
            mae = np.random.uniform(0.001, 5)
            meta_info_list.append(
                {"input_length": i,
                 "label_length": j,
                 "predict_length": k,
                 "mse": mse,
                 "mae": mae,
                 "model_id": f"testing_model_{counter}"}
            )
            counter += 1

with open(os.path.join(meta_info_folder, "test_meta_info.json"), "w") as f:
    json.dump(meta_info_list, f, indent=4)
