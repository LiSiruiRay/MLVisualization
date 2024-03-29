# Author: ray
# Date: 3/25/24
# Description:

import streamlit as st

import plotly.express as px

from util.model_meta_info_reading import reading_test_meta_data

# df = px.data.iris()
df = reading_test_meta_data()
fig = px.parallel_coordinates(df, color="mae", labels={"input_length": "input_length",
                                                              "label_length": "label_length",
                                                              "predict_length": "predict_length",},
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=2)
st.plotly_chart(fig)

fig = px.scatter(df, x='input_length', y='predict_length', color='mae',
                 color_continuous_scale='Viridis',  # Color scale
                 labels={'color_col': 'Color Intensity'})  # Label for color scale
fig.update_layout(title='Scatter Plot with Color Encoding',
                  xaxis_title='x_col',
                  yaxis_title='y_col')
st.plotly_chart(fig)
