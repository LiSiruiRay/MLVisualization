import streamlit as st
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_state(keys_with_defaults):
    """Initialize Streamlit session state with default values for given keys"""
    for key, default_value in keys_with_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def display_model_selection(model_names: list) -> list:
    # Use a multi-select box for model selection
    selected_models = st.multiselect('Select Model(s):', model_names, default=None)

    # Update the selected models in the session state
    st.session_state['selected_models'] = selected_models

    # Display the currently selected models
    st.write("You have selected: ", selected_models)
    logger.debug(f"Selected models: {type(selected_models)}")

    return selected_models


if __name__ == "__main__":
    logger.debug("running")
    model_names = ['Model A', 'Model B', 'Model C']  # actual model names

    initialize_state({
        'selected_models': [],  # Default to an empty list
        'expanded': False
    })

    expand_button_text = 'Show/Hide Selections'
    if st.button(expand_button_text):
        st.session_state['expanded'] = not st.session_state['expanded']

    # Conditional rendering of the selection buttons
    if st.session_state['expanded']:
        display_model_selection(model_names)
