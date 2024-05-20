# Author: ray
# Date: 4/16/24
# Description:
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas import DataFrame

from FEDformer.models import Informer, FEDformer, Autoformer, Transformer
from FEDformer.utils.timefeatures import time_features
from util.common import get_proje_root_path

import logging

from util.data_set import read_json_and_create_namespace, concat_time_series
from util.model_until.data_provider_loader import DataProviderLoader

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

        configs = read_json_and_create_namespace(json_file_path=self.model_meta_info_path)
        model = model_dict[model_name]
        logger.debug("check if called Autocorrelation used")
        self.loaded_model = model.Model(configs)
        logger.debug("check if called Autocorrelation used 2")
        self.loaded_model.load_state_dict(torch.load(model_check_point_path, map_location=device))
        self.device = device

    def get_input_dates(self, input_data: DataFrame, data_stamp):
        seq_len = self.meta_info['seq_len']
        label_len = self.meta_info['label_len']
        pred_len = self.meta_info['pred_len']

        s_begin = 0
        s_end = s_begin + seq_len
        r_begin = s_end - label_len
        r_end = r_begin + label_len + pred_len

        seq_x = input_data[s_begin:s_end]
        seq_y = input_data[r_begin:r_end]
        seq_y_expected_length = r_end - r_begin
        seq_y_actual_length = len(seq_y)

        array_zeros = np.zeros((seq_y_expected_length - seq_y_actual_length, 1))
        seq_y = np.concatenate((seq_y, array_zeros), axis=0)

        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    @classmethod
    def get_batch(cls, raw_data):
        """
        simulate the dataloader process but when use only 1 vector, batch should have length 1
        """
        batched = torch.tensor(raw_data, dtype=torch.float32)
        batched = batched.unsqueeze(0)
        return batched

    @classmethod
    def general_predict(cls,
                        batch_x,
                        batch_y,
                        batch_x_mark,
                        batch_y_mark,
                        model,
                        device,
                        pred_len: int,
                        label_len: int,
                        output_attention: bool = False,
                        use_amp: bool = False,
                        features: str = "S",
                        ):
        """
        extracted from source code
        """
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()  # mask
        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)

        # encoder - decoder

        # TODO: use_amp default is false, but might be true
        # TODO: output_attention default is false, but might be true

        def _run_model():
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if output_attention:
                outputs = outputs[0]
            return outputs

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        pred = outputs.detach().cpu().numpy()

        return pred

    def predict(self, input_data: DataFrame, dpl: DataProviderLoader):
        scaler = dpl.data_set.scaler
        # TODO: might not be scaled, do this later
        normalized_input = scaler.transform(input_data["OT"].values.reshape(-1, 1))

        df_raw = input_data

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        # TODO: data_stamp have different cases, default 1
        df_stamp = concat_time_series(df_stamp=df_stamp, n=self.meta_info["pred_len"])

        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq='h')
        data_stamp = data_stamp.transpose(1, 0)

        # TODO: better name for input_data in the definition of the get_input_dates
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.get_input_dates(normalized_input, data_stamp)

        batch_x = self.get_batch(seq_x)
        batch_y = self.get_batch(seq_y)
        batch_x_mark = self.get_batch(seq_x_mark)
        batch_y_mark = self.get_batch(seq_y_mark)

        pred = ModelLoader.general_predict(batch_x=batch_x,
                                           batch_y=batch_y,
                                           batch_x_mark=batch_x_mark,
                                           batch_y_mark=batch_y_mark,
                                           model=self.loaded_model,
                                           device=self.device,
                                           pred_len=self.meta_info['seq_len'],
                                           label_len=self.meta_info['label_len'], )

        pred = pred.reshape((self.meta_info["pred_len"], 1))
        return scaler.inverse_transform(pred)
