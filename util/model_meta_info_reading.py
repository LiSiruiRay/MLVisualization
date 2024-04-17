# Author: ray
# Date: 3/24/24
# Description: If later on we developed an interface for generating meta info while training, we add interface here
import json
import os.path
from collections import defaultdict

import numpy as np
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


def read_meta_data():
    proje_root = get_proje_root_path()
    meta_data_path = os.path.join(proje_root, "hpc_sync_files/meta_info/model_meta_info")
    meta_info_json_list = os.listdir(meta_data_path)
    df_dict = defaultdict(list)
    meta_info_list = list()

    for m in meta_info_json_list:
        if os.path.isfile(os.path.join(meta_data_path, m)):
            with open(os.path.join(meta_data_path, m)) as f:
                meta_info = json.load(f)
            meta_info_list.append(meta_info)

    for m in meta_info_list:
        model_path = os.path.join(proje_root, "hpc_sync_files/results", m["model_name"], "metrics.npy")
        # TODO: check this
        if not os.path.isfile(model_path):
            continue
        metric_array = read_metric_result(meta_info=m)
        df_dict["input_length"].append(m["seq_len"])
        df_dict["label_length"].append(m["label_len"])
        df_dict["predict_length"].append(m["pred_len"])
        df_dict["mae"].append(metric_array[0])
        df_dict["mse"].append(metric_array[1])
        df_dict["rmse"].append(metric_array[2])
        df_dict["mape"].append(metric_array[3])
        df_dict["mspe"].append(metric_array[4])

    model_mate_df = pd.DataFrame(df_dict)
    return model_mate_df


def read_metric_result(meta_info: dict):
    proje_root = get_proje_root_path()
    model_name = meta_info["model_name"]
    model_path = os.path.join(proje_root, "hpc_sync_files/results", model_name, "metrics.npy")
    metric_array = np.load(model_path)
    return metric_array
