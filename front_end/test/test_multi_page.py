# Author: ray
# Date: 3/25/24
# Description:

import streamlit as st

# Define your page functions
def home_page():
    st.title("Home Page")
    if st.button("Go to Model Page"):
        # Note: Assuming Streamlit introduces a replacement for setting query params
        st.experimental_set_query_params(page="model")

def model_page():
    st.title("Model Page")
    if st.button("Go to Analysis Page"):
        # Note: Assuming Streamlit introduces a replacement for setting query params
        st.experimental_set_query_params(page="analysis")

def analysis_page():
    st.title("Analysis Page")
    if st.button("Go to Home"):
        # Note: Assuming Streamlit introduces a replacement for setting query params
        st.experimental_set_query_params(page="home")

# Main app logic
# Corrected usage: access st.query_params as a property, not a function
query_params = st.query_params
page = query_params.get("page", ["home"])[0]  # Default to home page

if page == "home":
    home_page()
elif page == "model":
    model_page()
elif page == "analysis":
    analysis_page()
else:
    st.write("Page not found!")
