# Author: ray
# Date: 4/16/24
# Description:
import json
import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from pandas import Series, DataFrame

from FEDformer.models import Informer, FEDformer, Autoformer, Transformer
from util.common import get_proje_root_path

import logging

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
                        pred_len: int = 720,
                        label_len: int = 96,
                        output_attention: bool = False,
                        use_amp: bool = False,
                        features: str = "S",
                        ):
        """
        extracted from source code
        """
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float()
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
        normalized_input = scaler.transform(input_data["OT"].values)

        df_raw = input_data

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        # TODO: data_stamp have different cases, default 0
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
        data_stamp = df_stamp.drop(['date'], 1).values

        # TODO: better name for input_data in the definition of the get_input_dates
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.get_input_dates(normalized_input, df_stamp)

        batch_x = self.get_batch(seq_x)
        batch_y = self.get_batch(seq_y)
        batch_x_mark = self.get_batch(seq_x_mark)
        batch_y_mark = self.get_batch(seq_y_mark)

        pred = ModelLoader.general_predict(batch_x=batch_x,
                                           batch_y=batch_y,
                                           batch_x_mark=batch_x_mark,
                                           batch_y_mark=batch_y_mark,
                                           model=self.loaded_model,
                                           device=self.device, )

        # scaler.inverse_transform(pred.values)
        return scaler.inverse_transform(pred.values)
