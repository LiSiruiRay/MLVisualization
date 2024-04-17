# Author: ray
# Date: 3/24/24
# Description:

import sys
import os
import streamlit as st

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '..')))

from front_end.model_selection import initialize_state, display_model_selection
from util.model_meta_info_reading import get_model_list_from_folder_reading


def main():
    model_names = get_model_list_from_folder_reading()

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


if __name__ == '__main__':
    main()
