# Author: ray
# Date: 4/18/24
# Description:
import argparse
import json
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


def sub_frame(df, start_date: str, end_date: str): # df must be sorted
    df['date'] = pd.to_datetime(df['date'])

    # Use searchsorted to find the start and end indices
    start_idx = df['date'].searchsorted(pd.to_datetime(start_date), side='left')
    end_idx = df['date'].searchsorted(pd.to_datetime(end_date), side='right')

    # Use the indices to slice the DataFrame
    filtered_df = df.iloc[start_idx:end_idx]
    return filtered_df


def read_json_and_create_namespace(json_file_path: str):
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
