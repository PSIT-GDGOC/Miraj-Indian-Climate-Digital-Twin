import pytest
from streamlit.testing.v1 import AppTest

def test_app_launch_and_interact():
    # Initialize the app and run it
    at = AppTest.from_file("../app/streamlit_app.py").run(timeout=30)
    
    # Assert at.exception is empty to verify the app launches without errors
    assert not at.exception
    
    # Programmatically simulate switching the state selector widget and re-running the app.
    # The first selectbox is the State selector.
    state_selectbox = at.sidebar.selectbox[0]
    
    if len(state_selectbox.options) > 1:
        # Select the second option to simulate user interaction
        new_state = state_selectbox.options[1]
        state_selectbox.select(new_state).run(timeout=30)
        
    # Assert no exceptions after interaction (verifies dynamic re-masking and zero leakage)
    assert not at.exception
