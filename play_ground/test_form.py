# Author: ray
# Date: 3/23/24
# Description:

import streamlit as st
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model_names = ['Model A', 'Model B', 'Model C']  # Your actual model names

# Initialize session state for selected model
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = None

# Check if the 'expanded' key exists in session_state, if not initialize it as False
if 'expanded' not in st.session_state:
    st.session_state['expanded'] = False

# Button to show/hide selections
expand_button_text = 'Show/Hide Selections'
if st.button(expand_button_text):
    # print(f"text: {expand_button_text}")
    logger.debug(f"text-: {expand_button_text}")
    st.session_state['expanded'] = not st.session_state['expanded']


# Conditional rendering of the selection buttons
if st.session_state['expanded']:
    # Display buttons for each model
    for name in model_names:
        if st.button(name):
            st.session_state['selected_model'] = name

    # Display the currently selected model
    if st.session_state['selected_model']:
        st.write(f"You have selected: {st.session_state['selected_model']}")
