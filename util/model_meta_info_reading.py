# Author: ray
# Date: 3/24/24
# Description: If later on we developed an interface for generating meta info while training, we add interface here
import os.path

from util.common import get_proje_root_path


def get_model_list_from_folder_reading():
    proje_root_path = get_proje_root_path()
    data_path = os.path.join(proje_root_path, "statistic", "model_result_list")
    entries = os.listdir(data_path)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(data_path, entry))]

    return folders
