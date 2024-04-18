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

from util.data_set import read_training_ds_by_meta, sub_frame

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def live_calc_output(meta_info: dict, model):
    training_data = read_training_ds_by_meta(meta_info=meta_info)
    start_time = st.date_input("Enter start time", date(2016, 7, 1))
    end_time = st.date_input("Enter end time", date(2018, 7, 1))

    input_length = meta_info['seq_len']
    pred_length = meta_info['pred_len']

    if 'first_time' not in st.session_state:
        st.session_state.first_time = True
        selected_sub_frame = sub_frame(df=training_data,
                                       start_date=start_time,
                                       end_date=end_time)

    if st.button("Submit") or st.session_state.first_time:
        if ("start_time" not in st.session_state or
                "end_time" not in st.session_state or
                st.session_state.start_time != start_time or
                st.session_state.end_time != end_time):
            st.session_state.time_range_changed = True
        st.session_state.start_time = start_time
        st.session_state.end_time = end_time
        st.session_state.first_time = False

    if st.session_state.time_range_changed:
        selected_sub_frame = sub_frame(df=training_data,
                                       start_date=st.session_state.start_time,
                                       end_date=st.session_state.end_time)
        st.session_state.time_range_changed = True
    fig = go.Figure(data=go.Scatter(x=selected_sub_frame["date"], y=selected_sub_frame["OT"], mode='lines'))

    start_index = 0
    end_index = len(selected_sub_frame) - (input_length + pred_length)
    window_start_point = st.slider('Test data index', start_index, end_index)
    window_end_point = window_start_point + input_length
    update_fig_to_show_test(fig=fig,
                            selected_sub_frame=selected_sub_frame,
                            window_start_point=window_start_point,
                            window_end_point=window_end_point)
    st.plotly_chart(fig)


def update_fig_to_show_test(fig, selected_sub_frame, window_start_point, window_end_point):
    fig.add_shape(type="line",
                  x0=selected_sub_frame.iloc[window_start_point]["date"],
                  y0=min(selected_sub_frame["OT"]),
                  x1=selected_sub_frame.iloc[window_start_point]["date"],
                  y1=max(selected_sub_frame["OT"]),
                  line=dict(color="RoyalBlue", width=3))
    fig.add_shape(type="line",
                  x0=selected_sub_frame.iloc[window_end_point]["date"],
                  y0=min(selected_sub_frame["OT"]),
                  x1=selected_sub_frame.iloc[window_end_point]["date"],
                  y1=max(selected_sub_frame["OT"]),
                  line=dict(color="RoyalBlue", width=3))

    # Add area between lines
    fig.add_shape(type="rect",
                  x0=selected_sub_frame.iloc[window_start_point]["date"],
                  y0=min(selected_sub_frame["OT"]),
                  x1=selected_sub_frame.iloc[window_end_point]["date"],
                  y1=max(selected_sub_frame["OT"]),
                  line=dict(color="RoyalBlue", width=0),
                  fillcolor="LightSkyBlue", opacity=0.5)

    fig.update_layout(
        title='Time Series Plot with Selection Window',
        # yaxis=dict(range=[-500, 500]),
        xaxis=dict(range=[f"{st.session_state.start_time}", f"{st.session_state.end_time}"]),
    )


if __name__ == '__main__':
    proje_root = get_proje_root_path()
    meta_info_path = os.path.join(proje_root,
                                  "hpc_sync_files/meta_info/model_meta_info/pure_sin_first_with_meta_script_20240330@03h10m11s_20240330@03h10m11s.json")
    with open(meta_info_path, 'r') as f:
        meta_info = json.load(f)
    live_calc_output(meta_info, None)
