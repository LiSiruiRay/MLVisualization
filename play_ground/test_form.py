# Author: ray
# Date: 3/23/24
# Description:

# import streamlit as st
#
# # Check if the 'expanded' key exists in session_state, if not initialize it as False
# if 'expanded' not in st.session_state:
#     st.session_state['expanded'] = False
#
# # Create the button
# if st.button('Show Selections' if not st.session_state['expanded'] else 'Hide Selections'):
#     # Toggle the 'expanded' boolean each time the button is clicked
#     st.session_state['expanded'] = not st.session_state['expanded']
#
# # Use the value of 'expanded' to conditionally show the rest of the form
# if st.session_state['expanded']:
#     # Render the form here when expanded
#     form = st.form(key='my_form')
#
#     # Let's say you have a list of model names
#     model_names = ['Model A', 'Model B', 'Model C']  # Fill in with your actual model names
#
#     # Dynamically create checkboxes for each model
#     for name in model_names:
#         form.checkbox(name, key=name)
#
#     # A submit button for the form
#     submit_button = form.form_submit_button('Select')
#
#     # You can handle the form submission here
#     if submit_button:
#         selected_models = [name for name in model_names if st.session_state[name]]
#         st.write("You have selected:", selected_models)

# ----- second try
# import streamlit as st
#
# # Initialize session state for each model with a default False (not selected)
# model_names = ['Model A', 'Model B', 'Model C']  # Fill in with your actual model names
# for name in model_names:
#     if name not in st.session_state:
#         st.session_state[name] = False
#
# # Check if the 'expanded' key exists in session_state, if not initialize it as False
# if 'expanded' not in st.session_state:
#     st.session_state['expanded'] = False
#
# # Create the button
# if st.button('Show Selections' if not st.session_state['expanded'] else 'Hide Selections'):
#     # Toggle the 'expanded' boolean each time the button is clicked
#     st.session_state['expanded'] = not st.session_state['expanded']
#
# # Use the value of 'expanded' to conditionally show the rest of the form
# if st.session_state['expanded']:
#     # Render the form here when expanded
#     for name in model_names:
#         # If the button is clicked, toggle the state of the selection
#         if st.button(f"Select {'✓' if st.session_state[name] else ''} {name}"):
#             st.session_state[name] = not st.session_state[name]
#
#     # You can display the selected models dynamically
#     selected_models = [name for name in model_names if st.session_state[name]]
#     if selected_models:
#         st.write("You have selected:", ', '.join(selected_models))

import streamlit as st

model_names = ['Model A', 'Model B', 'Model C']  # Your actual model names

# Initialize session state for selected model
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = None

# Check if the 'expanded' key exists in session_state, if not initialize it as False
if 'expanded' not in st.session_state:
    st.session_state['expanded'] = False

# Button to show/hide selections
expand_button_text = 'Show Selections' if not st.session_state['expanded'] else 'Hide Selections'
if st.button(expand_button_text):
    print(f"text: {expand_button_text}")
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
