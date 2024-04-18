# Author: ray
# Date: 4/18/24
# Description:

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

# Sample time series data
np.random.seed(0)
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
values = np.random.randn(100).cumsum()
df = pd.DataFrame({'Date': dates, 'Value': values})

# Create a Plotly figure
fig = px.line(df, x='Date', y='Value', title='Time Series Data')

# Slider for selecting the window
window_size = st.slider('Select Window Size', min_value=5, max_value=30, value=10)
start_position = st.slider('Select Start Position', min_value=0, max_value=len(df)-window_size, value=0)

# Update the display of the windowed data
windowed_data = df.iloc[start_position:start_position+window_size]
st.write('Selected Window Data:', windowed_data)

# Display the plot
st.plotly_chart(fig)

# Highlight the selected window on the plot
fig.add_vrect(x0=windowed_data['Date'].iloc[0], x1=windowed_data['Date'].iloc[-1], fillcolor="green", opacity=0.25, line_width=0)
st.plotly_chart(fig)
