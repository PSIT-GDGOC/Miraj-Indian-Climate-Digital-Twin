"""Deterministic test suite for Standardized Precipitation Index (SPI) calculations.

Verifies mathematical accuracy, standard normal statistical properties,
zero-precipitation probability adjustment, accumulation window rolling logic,
insufficient data handling (including the bundled 7-day dataset), input validation,
non-mutation guarantees, and WMO category classification.
"""

import math
import numpy as np
import pytest
import scipy.special as special
import xarray as xr

from src.climate_indices import (
    INSUFFICIENT_DATA_CATEGORY,
    SPI_CATEGORIES,
    assess_drought_condition,
    calculate_spi,
    classify_spi,
    classify_spi_array,
    get_all_spi_categories,
    get_spi_category_info,
)


@pytest.fixture
def synthetic_long_rainfall_1d():
    """Deterministic synthetic daily precipitation series of 1,200 time steps.

    Uses a fixed random seed to sample from a Gamma distribution (shape=2.5,
    scale=8.0) with an introduced 15% zero-precipitation frequency.
    """
    rng = np.random.default_rng(seed=42)
    n_days = 1200
    times = np.arange(n_days)

    gamma_samples = rng.gamma(shape=2.5, scale=8.0, size=n_days)
    zero_mask = rng.random(size=n_days) < 0.15
    gamma_samples[zero_mask] = 0.0

    return xr.DataArray(
        gamma_samples,
        coords={"time": times},
        dims=["time"],
        name="rainfall",
        attrs={"units": "mm", "long_name": "Daily Rainfall"},
    )


@pytest.fixture
def synthetic_gridded_rainfall():
    """Deterministic 3D synthetic precipitation grid (time=90, lat=3, lon=3)."""
    rng = np.random.default_rng(seed=123)
    times = np.arange(90)
    lats = np.array([12.0, 13.0, 14.0])
    lons = np.array([75.0, 76.0, 77.0])

    grid = rng.gamma(shape=2.0, scale=12.0, size=(90, 3, 3))
    # Introduce NaN mask on one cell to verify spatial masking preservation
    grid[:, 0, 0] = np.nan
    # Introduce all-zero on one cell
    grid[:, 2, 2] = 0.0

    return xr.DataArray(
        grid,
        coords={"time": times, "lat": lats, "lon": lons},
        dims=["time", "lat", "lon"],
        name="rainfall",
        attrs={"units": "mm"},
    )


def test_calculate_spi_statistical_properties(synthetic_long_rainfall_1d):
    """Verify SPI values approximate standard normal distribution N(0, 1)."""
    window = 30
    spi = calculate_spi(synthetic_long_rainfall_1d, window=window, min_samples=30)

    assert isinstance(spi, xr.DataArray)
    assert spi.name == f"spi_{window}"
    assert spi.dims == synthetic_long_rainfall_1d.dims

    # First (window - 1) time points must be NaN
    assert np.isnan(spi.values[: window - 1]).all()

    valid_spi = spi.values[window - 1 :]
    assert np.isfinite(valid_spi).all()

    mean_spi = float(np.mean(valid_spi))
    std_spi = float(np.std(valid_spi))

    # Standard normal mean should be close to 0 and std close to 1
    assert abs(mean_spi) < 0.15, f"Expected mean near 0, got {mean_spi:.3f}"
    assert abs(std_spi - 1.0) < 0.18, f"Expected std near 1, got {std_spi:.3f}"


def test_calculate_spi_zero_precipitation_probability():
    """Verify zero precipitation produces cumulative probability q = n_0 / N.

    When zero rain occurs, H(0) = q, so SPI should equal Phi^-1(q).
    """
    n_days = 200
    # First 40 days zero (20%), remaining 160 days strictly positive
    data = np.zeros(n_days, dtype=float)
    rng = np.random.default_rng(seed=99)
    data[40:] = rng.gamma(shape=3.0, scale=10.0, size=160) + 1.0

    da = xr.DataArray(data, coords={"time": np.arange(n_days)}, dims=["time"])
    # 1-day window directly evaluates daily zero probability
    spi = calculate_spi(da, window=1, min_samples=30)

    # For zero observations, q = 40 / 200 = 0.20
    expected_spi_for_zero = special.ndtri(0.20)
    zero_spi_values = spi.values[:40]

    np.testing.assert_allclose(
        zero_spi_values,
        expected_spi_for_zero,
        rtol=1e-4,
        err_msg="Zero rainfall SPI did not match probit(q)",
    )


def test_calculate_spi_accumulation_windows():
    """Verify 30-day and 90-day rolling accumulation window mechanics."""
    n_days = 150
    rng = np.random.default_rng(seed=77)
    data = rng.gamma(shape=2.0, scale=5.0, size=n_days)
    da = xr.DataArray(data, coords={"time": np.arange(n_days)}, dims=["time"])

    # 30-day window
    spi_30 = calculate_spi(da, window=30, min_samples=30)
    assert np.isnan(spi_30.values[:29]).all()
    assert np.isfinite(spi_30.values[29:]).all()

    # 90-day window
    spi_90 = calculate_spi(da, window=90, min_samples=30)
    assert np.isnan(spi_90.values[:89]).all()
    assert np.isfinite(spi_90.values[89:]).all()


def test_insufficient_data_graceful_handling():
    """Verify that datasets with fewer observations than window return all NaNs without extrapolation."""
    # 7-day dataset (exact match for bundled sample_rainfall.nc)
    da_7days = xr.DataArray(
        np.array([12.0, 5.0, 0.0, 22.0, 18.0, 3.0, 0.0]),
        coords={"time": np.arange(7)},
        dims=["time"],
        name="rainfall",
    )

    # Requesting 30-day SPI on 7-day dataset
    spi_30 = calculate_spi(da_7days, window=30, min_samples=30)
    assert isinstance(spi_30, xr.DataArray)
    assert spi_30.shape == (7,)
    assert np.isnan(spi_30.values).all(), "7-day dataset must yield 100% NaN SPI without fabrication"

    # Requesting 90-day SPI on 7-day dataset
    spi_90 = calculate_spi(da_7days, window=90, min_samples=30)
    assert np.isnan(spi_90.values).all()


def test_min_samples_threshold():
    """Verify that series with fewer valid samples than min_samples return all NaNs."""
    n_days = 40
    data = np.full(n_days, 10.0)
    da = xr.DataArray(data, coords={"time": np.arange(n_days)}, dims=["time"])

    # 35 available samples after window=5, but min_samples=50 required
    spi = calculate_spi(da, window=5, min_samples=50)
    assert np.isnan(spi.values).all()


def test_all_zeros_and_constant_rainfall_handling():
    """Verify edge cases of all zeros or identical constant rainfall return NaN gracefully."""
    times = np.arange(50)

    # All zeros
    da_zeros = xr.DataArray(np.zeros(50), coords={"time": times}, dims=["time"])
    spi_zeros = calculate_spi(da_zeros, window=10, min_samples=30)
    assert np.isnan(spi_zeros.values).all()

    # Constant non-zero rainfall (zero variance -> A = 0 -> cannot fit Gamma)
    da_const = xr.DataArray(np.full(50, 15.0), coords={"time": times}, dims=["time"])
    spi_const = calculate_spi(da_const, window=10, min_samples=30)
    assert np.isnan(spi_const.values).all()


def test_gridded_preservation_and_spatial_masking(synthetic_gridded_rainfall):
    """Verify gridded (time, lat, lon) DataArray coordinates and NaN masking are preserved."""
    spi = calculate_spi(synthetic_gridded_rainfall, window=30, min_samples=30)

    assert spi.dims == ("time", "lat", "lon")
    assert spi.shape == synthetic_gridded_rainfall.shape
    np.testing.assert_array_equal(spi.lat.values, synthetic_gridded_rainfall.lat.values)
    np.testing.assert_array_equal(spi.lon.values, synthetic_gridded_rainfall.lon.values)

    # Cell (0, 0) was masked to NaN in input -> must remain NaN across all times in output
    assert np.isnan(spi.sel(lat=12.0, lon=75.0).values).all()

    # Cell (2, 2) was all zeros -> must remain NaN
    assert np.isnan(spi.sel(lat=14.0, lon=77.0).values).all()

    # Other cells with valid Gamma distributions must have finite values after window
    valid_cell = spi.sel(lat=13.0, lon=76.0).values[29:]
    assert np.isfinite(valid_cell).all()


def test_input_immutability(synthetic_long_rainfall_1d):
    """Verify calculate_spi does not mutate the input DataArray."""
    original_values = synthetic_long_rainfall_1d.values.copy()
    original_attrs = dict(synthetic_long_rainfall_1d.attrs)

    _ = calculate_spi(synthetic_long_rainfall_1d, window=30, min_samples=30)

    np.testing.assert_array_equal(synthetic_long_rainfall_1d.values, original_values)
    assert synthetic_long_rainfall_1d.attrs == original_attrs


def test_input_validation_errors():
    """Verify calculate_spi raises appropriate ValueError and TypeError on invalid inputs."""
    valid_da = xr.DataArray(np.ones(40), coords={"time": np.arange(40)}, dims=["time"])

    # Not a DataArray
    with pytest.raises(TypeError, match="xarray.DataArray"):
        calculate_spi([1.0, 2.0, 3.0], window=30)  # type: ignore

    # Missing 'time' dimension
    no_time_da = xr.DataArray(np.ones((5, 5)), dims=["lat", "lon"])
    with pytest.raises(ValueError, match="'time' dimension"):
        calculate_spi(no_time_da, window=30)

    # Invalid window
    with pytest.raises(ValueError, match="window must be a positive integer"):
        calculate_spi(valid_da, window=0)
    with pytest.raises(ValueError, match="window must be a positive integer"):
        calculate_spi(valid_da, window=-5)

    # Invalid min_samples
    with pytest.raises(ValueError, match="min_samples must be a positive integer"):
        calculate_spi(valid_da, window=10, min_samples=0)

    # Negative precipitation
    neg_da = xr.DataArray(np.array([1.0, -2.0, 3.0]), coords={"time": [0, 1, 2]}, dims=["time"])
    with pytest.raises(ValueError, match="non-negative"):
        calculate_spi(neg_da, window=2)


@pytest.mark.parametrize(
    ("spi_val", "expected_category"),
    [
        (2.5, "Extremely Wet"),
        (2.0, "Extremely Wet"),
        (1.99, "Very Wet"),
        (1.5, "Very Wet"),
        (1.49, "Moderately Wet"),
        (1.0, "Moderately Wet"),
        (0.5, "Near Normal"),
        (0.0, "Near Normal"),
        (-0.99, "Near Normal"),
        (-1.0, "Moderate Drought"),
        (-1.49, "Moderate Drought"),
        (-1.5, "Severe Drought"),
        (-1.99, "Severe Drought"),
        (-2.0, "Extreme Drought"),
        (-3.5, "Extreme Drought"),
        (float("nan"), "Insufficient Data"),
        (None, "Insufficient Data"),
        (float("inf"), "Insufficient Data"),
        (float("-inf"), "Insufficient Data"),
    ],
)
def test_classify_spi_categories(spi_val, expected_category):
    """Verify scalar SPI values map deterministically to WMO categories."""
    assert classify_spi(spi_val) == expected_category


def test_get_spi_category_info():
    """Verify get_spi_category_info returns SPICategory dataclass with proper metadata."""
    info_wet = get_spi_category_info(2.2)
    assert info_wet.name == "Extremely Wet"
    assert info_wet.severity == "extreme_wet"
    assert info_wet.color.startswith("#")

    info_drought = get_spi_category_info(-2.5)
    assert info_drought.name == "Extreme Drought"
    assert info_drought.severity == "extreme_drought"

    info_nan = get_spi_category_info(np.nan)
    assert info_nan == INSUFFICIENT_DATA_CATEGORY

    all_cats = get_all_spi_categories()
    assert len(all_cats) == 7
    assert {c.name for c in all_cats} == {
        "Extremely Wet",
        "Very Wet",
        "Moderately Wet",
        "Near Normal",
        "Moderate Drought",
        "Severe Drought",
        "Extreme Drought",
    }


def test_classify_spi_array():
    """Verify classify_spi_array performs vectorized string classification."""
    spi_arr = np.array([2.5, 1.2, 0.0, -1.2, -2.5, np.nan])
    cats = classify_spi_array(spi_arr)

    expected = np.array([
        "Extremely Wet",
        "Moderately Wet",
        "Near Normal",
        "Moderate Drought",
        "Extreme Drought",
        "Insufficient Data",
    ])
    np.testing.assert_array_equal(cats, expected)


@pytest.mark.parametrize(
    ("spi_val", "expected_severity", "expected_cat"),
    [
        (-2.5, "extreme_drought", "Extreme Drought"),
        (-1.8, "severe_drought", "Severe Drought"),
        (-1.2, "moderate_drought", "Moderate Drought"),
        (0.2, "normal", "Near Normal"),
        (1.2, "moderately_wet", "Moderately Wet"),
        (1.7, "very_wet", "Very Wet"),
        (2.5, "extreme_wet", "Extremely Wet"),
        (float("nan"), "unavailable", "Insufficient Data"),
    ],
)
def test_assess_drought_condition(spi_val, expected_severity, expected_cat):
    """Verify operational drought assessment returns structured screening feedback."""
    assessment = assess_drought_condition(spi_val, window=30)

    assert assessment.category == expected_cat
    assert assessment.severity == expected_severity
    assert assessment.message
    assert assessment.recommendation
