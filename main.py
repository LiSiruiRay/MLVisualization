import os

import numpy as np
import streamlit as st

from front_end.live_calc import live_calc_output
from front_end.model_selection import initialize_state, display_model_selection
from front_end.model_test_data_visual import show_test_data
from meta_info_process.meta_info_processor import MetaInfoProcessor
from util.common import get_proje_root_path
from util.model_meta_info_reading import get_model_list_from_folder_reading

import plotly.graph_objects as go

from util.model_until.data_provider_loader import DataProviderLoader
from util.model_until.model_loader import ModelLoader


def main():
    mip = MetaInfoProcessor()

    model_names = mip.model_meta_info_file_name_list
    proje_root = get_proje_root_path()

    initialize_state({
        'selected_model': None,
        # 'expanded': False
    })

    selected_model = display_model_selection(model_names)

    fig = go.Figure()

    fig.update_layout(
        title='Time Series Plot with Selection Window',
    )

    if selected_model:
        ml_list = []
        for model_id in selected_model:
            m = ModelLoader(model_id=model_id)
            m.load_model()
            ml_list.append(m)
        dpl_list = []
        for model_id in selected_model:
            d = DataProviderLoader(model_id=model_id)
            dpl_list.append(d)
        dpl_list = [d.load_load_data_provider() for d in dpl_list]

        live_calc_output(ml_list=ml_list, dpl_list=dpl_list, fig=fig)
        # print(selected_model)
        # refer to http://127.0.0.1:8889/lab/tree/visual_data_set.ipynb
        # preds = np.load(os.path.join(proje_root, "hpc_sync_files/results", selected_model, 'pred.npy'))
        # trues = np.load(os.path.join(proje_root, "hpc_sync_files/results", selected_model, 'true.npy'))
        # show_test_data(preds, trues)


if __name__ == '__main__':
    main()
