# Author: ray
# Date: 4/18/24
# Description:
import json
import sys
import os.path
from datetime import date
from typing import List

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from util.model_until.data_provider_loader import DataProviderLoader
from util.model_until.model_loader import ModelLoader

import plotly.graph_objects as go
import numpy as np
import streamlit as st

from util.data_set import read_training_ds_by_meta, sub_frame, sub_frame_by_index

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def live_calc_output(ml_list: List[ModelLoader], dpl_list: List[DataProviderLoader], fig):

    meta_info = ml_list[0].meta_info
    training_data = read_training_ds_by_meta(meta_info=meta_info)

    input_length = meta_info['seq_len']
    pred_length = meta_info['pred_len']

    start_index = 0
    end_index = len(training_data) - (input_length + pred_length)
    window_start_point = st.slider('Select input data: ', start_index, end_index)
    selected_sub_frame = sub_frame_by_index(df=training_data, start_idx=window_start_point,
                                            end_idx=window_start_point + int((input_length + pred_length) * 1.5))
    # fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected_sub_frame["date"],
                             y=selected_sub_frame["OT"],
                             mode='lines',
                             name='Ground Truth',
                             showlegend=True,
                             line=dict(color='black')))

    fig.add_trace(go.Scatter(x=selected_sub_frame["date"][:input_length],
                             y=selected_sub_frame["OT"],
                             mode='lines',
                             name='Input Data',
                             showlegend=True,
                             line=dict(color='blue')))

    input_for_test = selected_sub_frame.iloc[0:input_length + pred_length]  # data_x and data_y

    if st.button("calculate"):
        for i, ml in enumerate(ml_list):
            dpl = dpl_list[i]
            update_fig_to_show_pred_meta_info(window_start_point=0,
                                              meta_info=meta_info,
                                              selected_sub_frame=selected_sub_frame,
                                              fig=fig,
                                              input_data=input_for_test,
                                              ml=ml,
                                              dpl=dpl,
                                              model_number=i)

    st.plotly_chart(fig)


def update_fig_to_show_pred_meta_info(window_start_point: int,
                                      meta_info: dict,
                                      selected_sub_frame: pd.DataFrame,
                                      fig,
                                      input_data: pd.DataFrame,
                                      ml: ModelLoader, dpl: DataProviderLoader,
                                      model_number):
    update_fig_to_show_pred_detailed(window_start_point=window_start_point,
                                     seq_len=meta_info["seq_len"],
                                     pred_len=meta_info["pred_len"],
                                     selected_sub_frame=selected_sub_frame,
                                     fig=fig, input_data=input_data,
                                     ml=ml, dpl=dpl,
                                     model_number=model_number)


def update_fig_to_show_pred_detailed(window_start_point: int,
                                     seq_len: int, pred_len: int,
                                     selected_sub_frame: pd.DataFrame, fig,
                                     input_data: pd.DataFrame,
                                     ml: ModelLoader, dpl: DataProviderLoader,
                                     model_number):
    pred = ml.predict(input_data=input_data, dpl=dpl)
    pred_series = pd.Series(pred.flatten())
    prediction_dates = selected_sub_frame['date'][window_start_point + seq_len: window_start_point + seq_len + pred_len]
    prediction_values = pred_series.values

    # Trace for the prediction data
    fig.add_trace(go.Scatter(
        x=prediction_dates,
        y=prediction_values,
        mode='lines',
        name=f'Predicted Data Model: {model_number}',
        line=dict(color='orange')
    ))
    fig.update_layout(
        title='Time Series Plot with Selection Window',
        # yaxis=dict(range=[-500, 500]),
        # xaxis=dict(range=[f"{st.session_state.start_time}", f"{st.session_state.end_time}"]),
    )


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
                  fillcolor="LightSkyBlue", opacity=0.5,
                  name="selected part")


if __name__ == '__main__':
    print(f"hello")
    model_id = "pure_sin_first_with_meta_script_20240330@03h10m11s_20240330@03h10m11s"
    model_id_2 = "pure_sin_first_with_meta_script_20240401@03h16m24s_20240401@03h16m24s"

    fig = go.Figure()

    fig.update_layout(
        title='Time Series Plot with Selection Window',
    )
    ml = ModelLoader(model_id=model_id)
    ml_2 = ModelLoader(model_id=model_id_2)

    ml.load_model()
    ml_2.load_model()

    dpl = DataProviderLoader(model_id=model_id)
    dpl_2 = DataProviderLoader(model_id=model_id_2)

    dpl.load_load_data_provider()
    dpl_2.load_load_data_provider()

    ml_list = [ml, ml_2]
    dpl_list = [dpl, dpl_2]
    live_calc_output(ml_list=ml_list, dpl_list=dpl_list, fig=fig)
    # live_calc_output(ml_2.meta_info, ml=ml_2, dpl=dpl_2, fig=fig)
