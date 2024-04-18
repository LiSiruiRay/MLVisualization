# Author: ray
# Date: 4/18/24
# Description:
import os

import pandas as pd

from util.common import get_proje_root_path


def read_training_ds(ds_name: str, root_path: str = None):
    if root_path is None:
        root_path = os.path.join(get_proje_root_path(), "hpc_sync_files/meta_info/datasets_info")

    file_path = os.path.join(root_path, f"{ds_name}.csv")
    return pd.read_csv(file_path)


def read_training_ds_by_meta(meta_info: dict, root_path: str = None):
    return read_training_ds(meta_info["dataset_id"], root_path)