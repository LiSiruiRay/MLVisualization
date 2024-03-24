# Author: ray
# Date: 3/23/24
# Description:

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Define your list of models
model_list = ['Linear Regression', 'Random Forest', 'SVM', 'Neural Network']

# Simulate some data
X = np.random.rand(100, 1) * 10  # Features
y = 2.5 * X + np.random.randn(100, 1) * 2  # Targets with some noise

# Initialize a linear regression model
model = LinearRegression()

# Sidebar for user inputs
st.sidebar.title('Model Controls')
retrain_model = st.sidebar.button('Retrain Model')
new_data_variance = st.sidebar.slider('Data Variance', 1, 10, 2)

if retrain_model:
    # Simulate new data with the specified variance
    y = 2.5 * X + np.random.randn(100, 1) * new_data_variance
    # Retrain the model on the new data
    model.fit(X, y)

# Always display the model's predictions
model.fit(X, y)
predictions = model.predict(X)

# fig = go.Figure()
fig = px.scatter(x=X.ravel(), y=y.ravel(), labels={'x': 'Features', 'y': 'Targets'})
fig.add_scatter(x=X.ravel(), y=predictions.ravel(), mode='lines', name='Prediction')
fig.update_layout(
    title="Interactive Plot: Changing Coefficients 'a' and 'b' in a*x^2 + b*x",
    xaxis_title="X",
    yaxis_title="Y",
    xaxis=dict(range=[-10, 10]),
    yaxis=dict(range=[0, 250]),
    width=600,
    height=500
)

st.plotly_chart(fig)
