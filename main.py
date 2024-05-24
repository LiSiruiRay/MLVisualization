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

def display_selected_model_info(ml_list):
    info_list = []
    for i, m in enumerate(ml_list):
        meta_info = m.meta_info
        single_info_dict = {"model name": meta_info["model"],
                            "seq_len": meta_info["seq_len"],
                            "label_len": meta_info["label_len"],
                            "pred_len": meta_info["pred_len"],
                            "data_path": meta_info["data_path"],}
        info_list.append(single_info_dict)

    return info_list

def main():
    st.set_page_config(layout="wide")
    mip = MetaInfoProcessor()

    model_names = mip.model_meta_info_file_name_list

    initialize_state({
        'selected_model': None,
        # 'expanded': False
    })

    selected_model = display_model_selection(model_names)


    fig = go.Figure()

    fig.update_layout(
        title='Time Series Plot with Selection Window',
        width=1700, height=600
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
            d.load_load_data_provider()
            dpl_list.append(d)
        st.write("selected", display_selected_model_info(ml_list=ml_list))
        live_calc_output(ml_list=ml_list, dpl_list=dpl_list, fig=fig)


if __name__ == '__main__':
    main()
