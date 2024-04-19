# Author: ray
# Date: 4/16/24
# Description:
import json
import argparse
import os

import torch
import torch.nn as nn

from FEDformer.models import Informer, FEDformer, Autoformer, Transformer
from util.common import get_proje_root_path

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model_dict = {
    'FEDformer': FEDformer,
    'Autoformer': Autoformer,
    'Transformer': Transformer,
    'Informer': Informer,
}


class ModelLoader:
    loaded_model: nn.Module
    proje_root_path: str
    sync_file_path: str
    model_id: str
    meta_info: dict
    model_meta_info_path: str
    device: torch.device

    def __init__(self, model_id: str, sync_file_path: str = "hpc_sync_files"):
        self.model_id = model_id
        self.proje_root_path = get_proje_root_path()
        self.sync_file_path = os.path.join(self.proje_root_path, sync_file_path)
        self.model_meta_info_path = os.path.join(self.sync_file_path, f"meta_info/model_meta_info/{self.model_id}.json")
        with open(self.model_meta_info_path, "r") as f:
            self.meta_info = json.load(f)

    # TODO: change the device to cuda
    def load_model(self, model_name: str = "Autoformer", device: torch.device = torch.device('cpu')):
        model_check_point_path = os.path.join(self.sync_file_path,
                                              f"checkpoints/{self.meta_info['model_name']}/checkpoint.pth")

        configs = ModelLoader.read_json_and_create_namespace(json_file_path=self.model_meta_info_path)
        model = model_dict[model_name]
        logger.debug("check if called Autocorrelation used 11111 ! ---")
        self.loaded_model = model.Model(configs)
        logger.debug("check if called Autocorrelation used ! ---")
        self.loaded_model.load_state_dict(torch.load(model_check_point_path, map_location=device))
        self.device = device

    @classmethod
    def read_json_and_create_namespace(cls, json_file_path: str):
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Create a Namespace from the JSON data
        namespace = argparse.Namespace(
            is_training=data['is_training'],
            task_id=data['task_id'],
            model=data['model'],
            version=data['version'],
            mode_select=data['mode_select'],
            modes=data['modes'],
            L=data['L'],
            base=data['base'],
            cross_activation=data['cross_activation'],
            data=data['data'],
            root_path=data['root_path'],
            data_path=data['data_path'],
            features=data['features'],
            target=data['target'],
            freq=data['freq'],
            detail_freq=data['detail_freq'],
            checkpoints='./checkpoints/',
            seq_len=data['seq_len'],
            label_len=data['label_len'],
            pred_len=data['pred_len'],
            enc_in=data['enc_in'],
            dec_in=data['dec_in'],
            c_out=data['c_out'],
            d_model=data['d_model'],
            n_heads=data['n_heads'],
            e_layers=data['e_layers'],
            d_layers=data['d_layers'],
            d_ff=data['d_ff'],
            moving_avg=data['moving_avg'],
            factor=data['factor'],
            distil=data['distil'],
            dropout=data['dropout'],
            embed=data['embed'],
            activation=data['activation'],
            output_attention=data['output_attention'],
            do_predict=data['do_predict'],
            num_workers=10,
            itr=data['itr'],
            train_epochs=data['train_epochs'],
            batch_size=32,
            patience=data['patience'],
            learning_rate=0.0001,
            des=data['des'],
            loss='mse',
            lradj='type1',
            use_amp=data['use_amp'],
            use_gpu=True,
            gpu=0,
            use_multi_gpu=data['use_multi_gpu'],
            devices='0,1',
            dataset_id=data['dataset_id']
        )

        # Optionally, you can print the namespace to verify it
        print(namespace)
        return namespace

    def predict(self, input_data):
        pass
