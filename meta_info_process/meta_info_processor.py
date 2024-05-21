# Author: ray
# Date: 3/31/24
# Description:
import copy
import json
import os.path
from typing import List

from util.common import get_proje_root_path


class MetaInfoProcessor:
    all_model_info_folder_path: str
    check_points_path: str
    datasets_info_path: str
    model_meta_info_path: str
    model_meta_info_file_name_list: List[str]
    model_mate_info_list: list

    def __init__(self, file_path: str = None):
        if file_path is None:
            # path might be abs or relative, thus give default as None
            proje_root = get_proje_root_path()
            self.all_model_info_folder_path = os.path.join(proje_root, "hpc_sync_files")
        else:
            self.all_model_info_folder_path = file_path

        self.check_points_path = os.path.join(self.all_model_info_folder_path, "checkpoints")
        self.datasets_info_path = os.path.join(self.all_model_info_folder_path, "meta_info/datasets_info")
        self.model_meta_info_path = os.path.join(self.all_model_info_folder_path, "meta_info/model_meta_info")
        self.model_mate_info_list = list()
        self._load_all_model_info()

    @classmethod
    def _load_one_model_info(cls, single_model_mata_info_path: str):
        with open(single_model_mata_info_path, 'r') as f:
            single_model_meta_info = json.load(f)
        return single_model_meta_info

    def model_meta_info_valid(self, model_meta_info_path):
        single_model_mata_info_path = os.path.join(self.model_meta_info_path, model_meta_info_path)
        with open(single_model_mata_info_path, 'r') as f:
            meta_info = json.load(f)
        return os.path.exists(os.path.join(self.check_points_path, meta_info["model_name"], "checkpoint.pth"))

    def _load_all_model_info(self):
        # os.listdir(data_path)
        meta_info_path_list = os.listdir(self.model_meta_info_path)
        meta_info_path_list = [m for m in meta_info_path_list if self.model_meta_info_valid(m)]

        self.model_meta_info_file_name_list = copy.deepcopy(meta_info_path_list)
        self.model_meta_info_file_name_list = [i.replace('.json', '') for i in self.model_meta_info_file_name_list]
        for each_model_meta_info in meta_info_path_list:
            single_model_mata_info_path = os.path.join(self.model_meta_info_path, each_model_meta_info)
            self.model_mate_info_list.append(MetaInfoProcessor._load_one_model_info(single_model_mata_info_path))

    def get_model_list_from_folder_reading(self) -> List[str]:
        model_list = []
        for each_model_info in self.model_mate_info_list:
            model_list.append(each_model_info["model_name"])
        return model_list
