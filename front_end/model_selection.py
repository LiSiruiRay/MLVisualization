# Author: ray
# Date: 3/24/24
# Description:


import streamlit as st
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_state(keys_with_defaults):
    """Initialize Streamlit session state with default values for given keys"""
    for key, default_value in keys_with_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def display_model_selection(model_names: list):
    # Display buttons for each model
    for name in model_names:
        if st.button(name):
            st.session_state['selected_model'] = name
        st.write(f"Info about {name}: This model is known for its accuracy in specific conditions.")

    # Display the currently selected model
    if st.session_state['selected_model']:
        st.write(f"You have selected: {st.session_state['selected_model']}")


if __name__ == "__main__":
    logger.debug("running")
    model_names = ['Model A', 'Model B', 'Model C']  # actual model names

    initialize_state({
        'selected_model': None,
        'expanded': False
    })

    expand_button_text = 'Show/Hide Selections'
    if st.button(expand_button_text):
        st.session_state['expanded'] = not st.session_state['expanded']

    # Conditional rendering of the selection buttons
    if st.session_state['expanded']:
        display_model_selection(model_names)