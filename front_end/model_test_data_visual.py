# Author: ray
# Date: 3/24/24
# Description: given data, showing the test result with a slide bar

import plotly.graph_objects as go
import numpy as np
import streamlit as st


def show_test_data(preds: np.ndarray, trues: np.ndarray):
    new_data_variance = st.slider('Test data index', 0, preds.shape[0] - 1)
    initial_a = 1
    fig = go.Figure()

    # Add trace for initial data
    fig.add_trace(go.Scatter(y=np.squeeze(trues[new_data_variance]), mode='lines', name='Ground Truth'))
    fig.add_trace(go.Scatter(y=np.squeeze(preds[new_data_variance]), mode='lines', name='Prediction'))

    sliders = []

    # Slider for coefficient 'a'
    # slider_a = dict(
    #     active=0,
    #     currentvalue={"prefix": "Coefficient a: "},
    #     pad={"t": 50},  # Adjust the top padding
    #     steps=[{
    #         'method': 'update',
    #         'label': str(a),
    #         'args': [{'y': [np.squeeze(trues[a]), np.squeeze(preds[a])]}]
    #     } for a in range(1, 120)]  # Slider values for 'a' from 1 to 5
    # )
    # sliders.append(slider_a)

    fig.update_layout(
        # sliders=sliders,
        title="test result drawing",
        xaxis_title="X",
        yaxis_title="Y",
        # width=600,
        # height=500
    )
    st.plotly_chart(fig)


if __name__ == "__main__":
    folder_path = "front_end/test/test_data/seq96_label48_p4000_pati7_epoch10_des-non_stat_sin_sin_Autoformer_ETTm2_ftS_sl96_ll48_pl4000_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0/"
    preds = np.load(folder_path + 'pred.npy')
    trues = np.load(folder_path + 'true.npy')
    show_test_data(preds, trues)
