# Author: ray
# Date: 3/24/24
# Description: If later on we developed an interface for generating meta info while training, we add interface here
import json
import os.path
from collections import defaultdict

import pandas as pd

from util.common import get_proje_root_path


def get_model_list_from_folder_reading():
    proje_root_path = get_proje_root_path()
    data_path = os.path.join(proje_root_path, "statistic", "model_result_list")
    entries = os.listdir(data_path)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(data_path, entry))]

    return folders

def reading_test_meta_data():
    proje_root = get_proje_root_path()
    meta_data_path = os.path.join(proje_root, "statistic/meta_info/test_meta_info.json")
    with open(meta_data_path, "r") as f:
        meta_info = json.load(f)

    df_dict = defaultdict(list)

    for k, v in meta_info[0].items():
        for i in meta_info:
            df_dict[k].append(i[k])

    model_mate_df = pd.DataFrame(df_dict)
    return model_mate_df
