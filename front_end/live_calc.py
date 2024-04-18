# Author: ray
# Date: 4/18/24
# Description:
import json
import sys
import os.path
from datetime import date

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from util.common import get_proje_root_path

import plotly.graph_objects as go
import numpy as np
import streamlit as st

from util.data_set import read_training_ds_by_meta

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def live_calc_output(meta_info: dict, model):
    training_data = read_training_ds_by_meta(meta_info=meta_info)
    fig = go.Figure(data=go.Scatter(x=training_data["date"], y=training_data["OT"], mode='lines'))
    start_time = st.date_input("Enter start time", date(2016, 7, 1))
    end_time = st.date_input("Enter end time", date(2018, 7, 1))

    if 'first_time' not in st.session_state:
        st.session_state.first_time = True

    if st.button("Submit") or st.session_state.first_time:
        st.session_state.start_time = start_time
        st.session_state.end_time = end_time
        st.session_state.first_time = False

    # TODO: fix the end point, calculation based on input data needed
    start_index = 0
    end_index = 100
    new_data_variance = st.slider('Test data index', start_index, end_index)
    fig.update_layout(
        title='Time Series Plot with Selection Window',
        # yaxis=dict(range=[-500, 500]),
        xaxis=dict(range=[f"{st.session_state.start_time}", f"{st.session_state.end_time}"]),
    )
    # fig.show()
    st.plotly_chart(fig)


if __name__ == '__main__':
    proje_root = get_proje_root_path()
    meta_info_path = os.path.join(proje_root,
                                  "hpc_sync_files/meta_info/model_meta_info/pure_sin_first_with_meta_script_20240330@03h10m11s_20240330@03h10m11s.json")
    with open(meta_info_path, 'r') as f:
        meta_info = json.load(f)
    live_calc_output(meta_info, None)
