import os
import pytest
from unittest.mock import patch
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "streamlit_app.py"))


def test_app_launch_and_interact():
    """Verify filters, visualizations, exports, and reset behavior."""
    st.cache_resource.clear()
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception
    assert len(at.get("plotly_chart")) == 3
    assert any("Climate Risk Screening" in heading.value for heading in at.subheader)
    assert any("Drought Conditions" in heading.value for heading in at.subheader)
    assert any("Screening indicator only" in caption.value for caption in at.caption)
    assert any("Historical observations required for drought index" in info.value for info in at.info)

    download_buttons = at.get("download_button")
    assert {button.label for button in download_buttons} == {
        "Download national summary as CSV",
        "Download Filtered State Data as CSV",
        "Download state time series as CSV",
    }
    assert all(button.proto.url.endswith(".csv") for button in download_buttons)

    # Confirm that the filters drive a complete rerender without errors.
    state_selectbox = at.sidebar.selectbox[0]
    new_state = "Kerala" if "Kerala" in state_selectbox.options else state_selectbox.options[1]
    state_selectbox.select(new_state).run(timeout=30)
    at.sidebar.selectbox[2].select("Max Temperature (°C)").run(timeout=30)

    assert not at.exception
    assert at.sidebar.selectbox[0].value == new_state
    assert at.sidebar.selectbox[2].value == "Max Temperature (°C)"

    # The reset button should restore all widget defaults.
    at.sidebar.button[0].click().run(timeout=30)
    assert not at.exception
    expected_state = "Karnataka" if "Karnataka" in at.sidebar.selectbox[0].options else at.sidebar.selectbox[0].options[0]
    assert at.sidebar.selectbox[0].value == expected_state
    assert at.sidebar.selectbox[1].value == at.sidebar.selectbox[1].options[-1]
    assert at.sidebar.selectbox[2].value == "Rainfall (mm)"


def test_app_missing_dataset_fails_safely():
    """Verify app handles missing NetCDF dataset gracefully with st.error and st.stop."""
    st.cache_resource.clear()
    orig_exists = os.path.exists

    def mock_exists(path):
        if "sample_rainfall" in str(path):
            return False
        return orig_exists(path)

    with patch("os.path.exists", side_effect=mock_exists):
        at = AppTest.from_file(APP_PATH).run(timeout=30)
        # Should stop cleanly with zero unhandled exceptions
        assert not at.exception
        # Error message should instruct user to run the decoder script
        errors = [err.value for err in at.error]
        assert any("scripts/decoder.py" in str(msg) for msg in errors)


def test_app_corrupt_dataset_fails_safely():
    """Verify app handles NetCDF backend/corruption errors gracefully with st.error and st.stop."""
    st.cache_resource.clear()
    with patch("xarray.open_dataset", side_effect=OSError("NetCDF: Unknown file format")):
        at = AppTest.from_file(APP_PATH).run(timeout=30)
        # Should stop cleanly with zero unhandled exceptions
        assert not at.exception
        # Error message should indicate backend engine or file corruption
        errors = [err.value for err in at.error]
        assert any("Unable to parse NetCDF dataset" in str(msg) for msg in errors)


def test_app_drought_conditions_with_extended_dataset():
    """Verify that when >= 30 days are available, SPI metrics and map render cleanly."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    st.cache_resource.clear()
    st.cache_data.clear()

    # Generate 70 daily observations for test (window=30 + min_samples=30 requires >= 59 days)
    dates = pd.date_range("2024-05-01", periods=70)
    lons = np.linspace(66.5, 100.0, 135)
    lats = np.linspace(6.5, 38.5, 129)
    rng = np.random.default_rng(42)

    rainfall_data = rng.gamma(shape=2.0, scale=10.0, size=(70, 129, 135))
    temp_data = np.full((70, 129, 135), 30.0)

    mock_ds = xr.Dataset(
        {
            "rainfall": (["time", "lat", "lon"], rainfall_data),
            "temperature": (["time", "lat", "lon"], temp_data),
        },
        coords={"time": dates, "lat": lats, "lon": lons},
    )

    with patch("xarray.open_dataset", return_value=mock_ds):
        at = AppTest.from_file(APP_PATH).run(timeout=30)
        assert not at.exception
        assert any("Drought Conditions" in heading.value for heading in at.subheader)

        metric_labels = [m.label for m in at.metric]
        assert "State Mean SPI" in metric_labels
        assert "Peak Drought Intensity (Min SPI)" in metric_labels
        assert "Drought Category" in metric_labels

        # 4 plotly charts: baseline map, trend line, histogram, and SPI state map
        assert len(at.get("plotly_chart")) == 4

        download_labels = {btn.label for btn in at.get("download_button")}
        assert any("SPI as CSV" in label for label in download_labels)
