# Author: ray
# Date: 3/25/24
# Description:

import streamlit as st

import plotly.express as px

df = px.data.iris()
fig = px.parallel_coordinates(df, color="species_id", labels={"species_id": "Species",
                                                              "sepal_width": "Sepal Width",
                                                              "sepal_length": "Sepal Length",
                                                              "petal_width": "Petal Width",
                                                              "petal_length": "Petal Length", },
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=2)
st.plotly_chart(fig)
