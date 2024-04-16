# Author: ray
# Date: 4/16/24
# Description:
import copy
import json
import os

from torch import nn
from torch.utils.data import Dataset, DataLoader

from FEDformer.data_provider.data_factory import data_provider
from util.common import get_proje_root_path
from util.model_until.model_loader import ModelLoader


class DataProviderLoader:
    loaded_model: nn.Module
    proje_root_path: str
    sync_file_path: str
    model_id: str
    meta_info: dict
    model_meta_info_path: str
    config: dict
    data_set: Dataset
    data_loader: DataLoader

    def __init__(self, model_id: str, sync_file_path: str = "hpc_sync_files"):
        self.model_id = model_id
        self.proje_root_path = get_proje_root_path()
        self.sync_file_path = os.path.join(self.proje_root_path, sync_file_path)
        self.model_meta_info_path = os.path.join(self.sync_file_path, f"meta_info/model_meta_info/{self.model_id}.json")
        with open(self.model_meta_info_path, "r") as f:
            self.meta_info = json.load(f)

        self.config = ModelLoader.read_json_and_create_namespace(json_file_path=self.model_meta_info_path)

    def load_load_data_provider(self, flag: str = "test"):
        config_updated_dataset = copy.deepcopy(self.config)
        dataset_id = config_updated_dataset.dataset_id
        new_root_path = os.path.join(self.sync_file_path, f"meta_info/datasets_info/")
        config_updated_dataset.root_path = new_root_path
        config_updated_dataset.data_path = f"{dataset_id}.csv"
        self.data_set, self.data_loader = data_provider(config_updated_dataset, flag=flag)
