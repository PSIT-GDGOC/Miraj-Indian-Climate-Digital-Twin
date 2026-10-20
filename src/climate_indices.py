"""Standardized Precipitation Index (SPI) calculation engine and classification.

This module provides a mathematically rigorous, fully vectorized implementation
of the Standardized Precipitation Index (SPI) formulated by McKee, Doesken, and
Kleist (1993) and standardized by the World Meteorological Organization (WMO 2012).

Calculations fit a two-parameter Gamma distribution to rolling precipitation
accumulations using Thom's (1958) Maximum Likelihood Estimation with an empirical
probability adjustment for zero-precipitation observations. The cumulative
distribution is transformed to the standard normal distribution via the probit
(inverse normal cumulative distribution) function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union
import warnings

import numpy as np
import scipy.special as special
import xarray as xr


@dataclass(frozen=True)
class SPICategory:
    """WMO drought and wetness category metadata."""

    name: str
    severity: str
    description: str
    color: str
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class DroughtAssessment:
    """Operational drought screening summary for an observation region."""

    spi_value: float
    category: str
    severity: str
    message: str
    recommendation: str


# Standard WMO SPI classification categories (WMO-No. 1090)
SPI_CATEGORIES: tuple[SPICategory, ...] = (
    SPICategory(
        name="Extremely Wet",
        severity="extreme_wet",
        description="Extremely wet conditions; potential flood hazard",
        color="#08306b",
        lower_bound=2.0,
        upper_bound=float("inf"),
    ),
    SPICategory(
        name="Very Wet",
        severity="very_wet",
        description="Very wet conditions; substantially above normal precipitation",
        color="#2171b5",
        lower_bound=1.5,
        upper_bound=2.0,
    ),
    SPICategory(
        name="Moderately Wet",
        severity="moderately_wet",
        description="Moderately wet conditions; moderately above normal precipitation",
        color="#6baed6",
        lower_bound=1.0,
        upper_bound=1.5,
    ),
    SPICategory(
        name="Near Normal",
        severity="normal",
        description="Near normal precipitation",
        color="#c6dbef",
        lower_bound=-1.0,
        upper_bound=1.0,
    ),
    SPICategory(
        name="Moderate Drought",
        severity="moderate_drought",
        description="Moderate drought; reduced soil moisture and streamflow",
        color="#fdd0a2",
        lower_bound=-1.5,
        upper_bound=-1.0,
    ),
    SPICategory(
        name="Severe Drought",
        severity="severe_drought",
        description="Severe drought; crop water stress and water supply deficits",
        color="#f16913",
        lower_bound=-2.0,
        upper_bound=-1.5,
    ),
    SPICategory(
        name="Extreme Drought",
        severity="extreme_drought",
        description="Extreme drought; widespread water shortages and agricultural losses",
        color="#7f0000",
        lower_bound=float("-inf"),
        upper_bound=-2.0,
    ),
)

INSUFFICIENT_DATA_CATEGORY = SPICategory(
    name="Insufficient Data",
    severity="unavailable",
    description="Insufficient historical observations to fit distribution",
    color="#d9d9d9",
    lower_bound=float("nan"),
    upper_bound=float("nan"),
)


def get_all_spi_categories() -> tuple[SPICategory, ...]:
    """Return all standard WMO SPI categories."""
    return SPI_CATEGORIES


def classify_spi(spi_value: Union[float, int, None]) -> str:
    """Classify a scalar SPI value into a standard WMO category name.

    Parameters
    ----------
    spi_value:
        The Standardized Precipitation Index value (typically between -3.0 and +3.0).

    Returns
    -------
    str
        The standard WMO category label (e.g. 'Near Normal', 'Severe Drought').
        Returns 'Insufficient Data' if the value is NaN, None, or infinite.
    """
    if spi_value is None:
        return INSUFFICIENT_DATA_CATEGORY.name

    try:
        val = float(spi_value)
    except (ValueError, TypeError):
        return INSUFFICIENT_DATA_CATEGORY.name

    if np.isnan(val) or np.isinf(val):
        return INSUFFICIENT_DATA_CATEGORY.name

    if val <= -2.0:
        return "Extreme Drought"
    if val <= -1.5:
        return "Severe Drought"
    if val <= -1.0:
        return "Moderate Drought"
    if val < 1.0:
        return "Near Normal"
    if val < 1.5:
        return "Moderately Wet"
    if val < 2.0:
        return "Very Wet"
    return "Extremely Wet"


def get_spi_category_info(spi_value: Union[float, int, None]) -> SPICategory:
    """Return full SPICategory metadata for a scalar SPI value.

    Parameters
    ----------
    spi_value:
        Scalar SPI value.

    Returns
    -------
    SPICategory
        Immutable dataclass instance containing name, severity, description,
        hex color, and threshold bounds.
    """
    cat_name = classify_spi(spi_value)
    if cat_name == INSUFFICIENT_DATA_CATEGORY.name:
        return INSUFFICIENT_DATA_CATEGORY

    for cat in SPI_CATEGORIES:
        if cat.name == cat_name:
            return cat

    return INSUFFICIENT_DATA_CATEGORY


def assess_drought_condition(
    spi_value: Union[float, int, None],
    window: int = 30,
) -> DroughtAssessment:
    """Evaluate operational drought screening status from a regional SPI metric.

    Parameters
    ----------
    spi_value:
        Regional mean or grid SPI value.
    window:
        SPI accumulation window in days (default 30).

    Returns
    -------
    DroughtAssessment
        Structured assessment object with category, severity, message,
        and operational recommendation.
    """
    category = classify_spi(spi_value)

    if category == INSUFFICIENT_DATA_CATEGORY.name:
        return DroughtAssessment(
            spi_value=float("nan"),
            category=INSUFFICIENT_DATA_CATEGORY.name,
            severity="unavailable",
            message=(
                f"Insufficient historical observations to calculate {window}-day SPI. "
                "Reliable Gamma parameter fitting requires at least 30 consecutive observations."
            ),
            recommendation="Load extended historical observations or archive records before evaluating drought risk.",
        )

    val = float(spi_value)

    if val <= -2.0:
        return DroughtAssessment(
            spi_value=val,
            category="Extreme Drought",
            severity="extreme_drought",
            message=f"Extreme drought condition detected with {window}-day SPI of {val:.2f} (<= -2.00).",
            recommendation="Activate regional emergency drought contingency protocols and municipal water rationing.",
        )
    if val <= -1.5:
        return DroughtAssessment(
            spi_value=val,
            category="Severe Drought",
            severity="severe_drought",
            message=f"Severe drought condition detected with {window}-day SPI of {val:.2f} (-2.00 < SPI <= -1.50).",
            recommendation="Implement agricultural irrigation restrictions and conduct frequent reservoir capacity reviews.",
        )
    if val <= -1.0:
        return DroughtAssessment(
            spi_value=val,
            category="Moderate Drought",
            severity="moderate_drought",
            message=f"Moderate drought condition detected with {window}-day SPI of {val:.2f} (-1.50 < SPI <= -1.00).",
            recommendation="Initiate soil moisture conservation measures and issue early advisories for dryland agriculture.",
        )
    if val < 1.0:
        return DroughtAssessment(
            spi_value=val,
            category="Near Normal",
            severity="normal",
            message=f"Near-normal precipitation balance with {window}-day SPI of {val:.2f} (-1.00 < SPI < +1.00).",
            recommendation="Continue routine hydrometeorological surveillance and seasonal water balance tracking.",
        )
    if val < 1.5:
        return DroughtAssessment(
            spi_value=val,
            category="Moderately Wet",
            severity="moderately_wet",
            message=f"Moderately wet condition detected with {window}-day SPI of {val:.2f} (+1.00 <= SPI < +1.50).",
            recommendation="Monitor drainage efficiency in low-lying basins and track secondary crop planting windows.",
        )
    if val < 2.0:
        return DroughtAssessment(
            spi_value=val,
            category="Very Wet",
            severity="very_wet",
            message=f"Very wet condition detected with {window}-day SPI of {val:.2f} (+1.50 <= SPI < +2.00).",
            recommendation="Review flood retention basins, drainage systems, and reservoir discharge thresholds.",
        )
    return DroughtAssessment(
        spi_value=val,
        category="Extremely Wet",
        severity="extreme_wet",
        message=f"Extremely wet condition detected with {window}-day SPI of {val:.2f} (>= +2.00).",
        recommendation="Alert disaster management agencies for flash flood response and infrastructure safety inspections.",
    )


def classify_spi_array(spi_data: Union[np.ndarray, xr.DataArray]) -> np.ndarray:
    """Vectorized classification of an array of SPI values into WMO category strings.

    Parameters
    ----------
    spi_data:
        NumPy array or xarray DataArray of SPI values.

    Returns
    -------
    np.ndarray
        String array of the same shape containing category names.
    """
    arr = np.asarray(spi_data, dtype=float)
    output = np.full(arr.shape, INSUFFICIENT_DATA_CATEGORY.name, dtype=object)

    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return output

    vals = arr[finite_mask]
    cats = np.full(vals.shape, INSUFFICIENT_DATA_CATEGORY.name, dtype=object)

    cats[vals <= -2.0] = "Extreme Drought"
    cats[(vals > -2.0) & (vals <= -1.5)] = "Severe Drought"
    cats[(vals > -1.5) & (vals <= -1.0)] = "Moderate Drought"
    cats[(vals > -1.0) & (vals < 1.0)] = "Near Normal"
    cats[(vals >= 1.0) & (vals < 1.5)] = "Moderately Wet"
    cats[(vals >= 1.5) & (vals < 2.0)] = "Very Wet"
    cats[vals >= 2.0] = "Extremely Wet"

    output[finite_mask] = cats
    return output


def calculate_spi(
    rainfall: xr.DataArray,
    window: int,
    min_samples: int = 30,
) -> xr.DataArray:
    """Calculate a gridded Standardized Precipitation Index.

    Computes the Standardized Precipitation Index (SPI) across a gridded or
    one-dimensional precipitation xarray DataArray for a designated accumulation
    timescale (e.g., 30-day or 90-day window).

    The calculation follows the McKee et al. (1993) standard:
    1. Moving-window rolling summation of precipitation over ``window`` steps.
    2. Zero-precipitation probability estimation:
       :math:`q = n_0 / N`, where :math:`n_0` is zero count, :math:`N` is sample size.
    3. Fitting a two-parameter Gamma distribution to positive values using
       Thom's (1958) Maximum Likelihood Estimation:
       :math:`A = \\ln(\\bar{x}) - \\overline{\\ln(x)}`
       :math:`\\hat{\\alpha} = \\frac{1 + \\sqrt{1 + 4A/3}}{4A}`
       :math:`\\hat{\\beta} = \\frac{\\bar{x}}{\\hat{\\alpha}}`
    4. Evaluating cumulative probability:
       :math:`H(x) = q + (1 - q) G(x)`
    5. Probit transformation to standard normal variate:
       :math:`Z = \\Phi^{-1}(H(x))`

    Parameters
    ----------
    rainfall:
        xarray DataArray containing precipitation accumulations (in mm).
        Must include a ``time`` dimension. May contain arbitrary spatial
        dimensions (such as ``lat`` and ``lon``).
    window:
        Accumulation window in time steps (e.g. 30 for 30-day, 90 for 90-day).
        Must be a positive integer >= 1.
    min_samples:
        Minimum number of valid accumulated samples required along the time
        dimension to fit the Gamma distribution (default 30, per McKee et al.
        recommendations). Must be a positive integer >= 1.

    Returns
    -------
    xr.DataArray
        DataArray of identical shape, coordinates, and dimension ordering as
        ``rainfall``, containing dimensionless SPI values. Times or grid cells
        with insufficient observations, zero variance, or fewer than ``min_samples``
        are assigned NaN.

    Raises
    ------
    TypeError
        If ``rainfall`` is not an ``xarray.DataArray``.
    ValueError
        If ``window`` < 1, ``min_samples`` < 1, ``time`` dimension is missing,
        or ``rainfall`` contains negative values.
    """
    if not isinstance(rainfall, xr.DataArray):
        raise TypeError(
            f"rainfall must be an xarray.DataArray, received {type(rainfall).__name__}"
        )

    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError(f"window must be a positive integer >= 1, got {window}")

    if not isinstance(min_samples, (int, np.integer)) or min_samples < 1:
        raise ValueError(f"min_samples must be a positive integer >= 1, got {min_samples}")

    if "time" not in rainfall.dims:
        raise ValueError("Input DataArray must contain a 'time' dimension.")

    # Validate non-negativity without mutating input DataArray
    rainfall_values = rainfall.values
    finite_mask = np.isfinite(rainfall_values)
    if finite_mask.any() and np.any(rainfall_values[finite_mask] < 0):
        raise ValueError("Rainfall values must be non-negative.")

    total_time_steps = rainfall.sizes["time"]

    # Construct clean template DataArray with identical coordinates and attributes
    attrs = {
        "long_name": f"{window}-day Standardized Precipitation Index",
        "units": "dimensionless",
        "window": int(window),
        "min_samples": int(min_samples),
        "method": "Thom (1958) Gamma distribution fit with zero-precipitation probability adjustment",
    }

    # If dataset has fewer observations than the window or min_samples,
    # return an all-NaN DataArray without fabricating artificial values.
    if total_time_steps < window or total_time_steps < min_samples:
        nan_data = np.full(rainfall.shape, np.nan, dtype=float)
        return xr.DataArray(
            nan_data,
            coords=rainfall.coords,
            dims=rainfall.dims,
            name=f"spi_{window}",
            attrs=attrs,
        )

    # Rolling window accumulation along the time dimension without mutating input
    # First (window - 1) time points will automatically be NaN.
    rolling_acc = rainfall.rolling(time=window, min_periods=window).sum()

    # Move 'time' dimension to the leading axis (axis 0) for uniform 2D processing
    spatial_dims = [dim for dim in rainfall.dims if dim != "time"]
    ordered_dims = ["time"] + spatial_dims
    transposed = rolling_acc.transpose(*ordered_dims)
    acc_values = transposed.values.copy()
    orig_shape = acc_values.shape

    time_len = orig_shape[0]
    spatial_flat_len = int(np.prod(orig_shape[1:])) if len(orig_shape) > 1 else 1

    # Reshape to 2D array: (time, spatial_cells)
    acc_2d = acc_values.reshape(time_len, spatial_flat_len)

    # Initialize SPI output array with NaNs
    spi_2d = np.full_like(acc_2d, np.nan, dtype=float)

    valid_mask = ~np.isnan(acc_2d)
    valid_counts = valid_mask.sum(axis=0)

    # Filter columns that meet the minimum sample threshold
    zero_counts = ((acc_2d == 0) & valid_mask).sum(axis=0)
    q = np.divide(
        zero_counts,
        valid_counts,
        out=np.zeros(spatial_flat_len, dtype=float),
        where=valid_counts > 0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        pos_data = np.where((acc_2d > 0) & valid_mask, acc_2d, np.nan)
        pos_counts = (~np.isnan(pos_data)).sum(axis=0)

        mean_pos = np.nanmean(pos_data, axis=0)
        mean_log_pos = np.nanmean(np.log(pos_data), axis=0)
        a_param = np.log(mean_pos) - mean_log_pos

        # A column is eligible for Gamma fitting if:
        # 1. Total valid accumulated observations >= min_samples
        # 2. Not all observations are zero (q < 1.0)
        # 3. At least 2 positive observations exist (for variance estimation)
        # 4. A > 0 (variance > 0; non-constant values)
        # 5. Parameters are finite
        eligible_cols = (
            (valid_counts >= min_samples)
            & (q < 1.0)
            & (pos_counts >= 2)
            & (a_param > 1e-7)
            & np.isfinite(a_param)
            & np.isfinite(mean_pos)
        )

        if np.any(eligible_cols):
            acc_eligible = acc_2d[:, eligible_cols]
            q_el = q[eligible_cols]
            a_el = a_param[eligible_cols]
            mean_el = mean_pos[eligible_cols]

            # Thom's Maximum Likelihood Estimation of Gamma parameters
            alpha = (1.0 + np.sqrt(1.0 + 4.0 * a_el / 3.0)) / (4.0 * a_el)
            beta = mean_el / alpha

            # Broadcast parameters across time dimension
            alpha_b = alpha[np.newaxis, :]
            beta_b = beta[np.newaxis, :]
            q_b = q_el[np.newaxis, :]

            # Incomplete gamma integral P(alpha, x / beta)
            scaled_x = np.divide(
                acc_eligible,
                beta_b,
                out=np.zeros_like(acc_eligible),
                where=beta_b > 0,
            )
            gamma_cdf = special.gammainc(alpha_b, scaled_x)

            # Cumulative probability adjustment for zero precipitation
            prob_h = np.where(acc_eligible == 0, q_b, q_b + (1.0 - q_b) * gamma_cdf)

            # Clip cumulative probability to prevent +/- infinity from normal quantile
            prob_h = np.clip(prob_h, 1e-7, 1.0 - 1e-7)

            # Probit transform: standard normal quantile
            spi_computed = special.ndtri(prob_h)

            # Preserve NaNs where rolling sum was NaN (e.g. initial window periods)
            spi_computed = np.where(
                ~np.isnan(acc_eligible),
                spi_computed,
                np.nan,
            )

            spi_2d[:, eligible_cols] = spi_computed

    # Reshape back to original dimensional structure
    spi_ordered = spi_2d.reshape(orig_shape)

    # Reconstruct xarray DataArray with transposed coordinate mapping
    transposed_result = xr.DataArray(
        spi_ordered,
        coords=transposed.coords,
        dims=transposed.dims,
        name=f"spi_{window}",
        attrs=attrs,
    )

    # Transpose back to match the exact dimension order of the original input
    return transposed_result.transpose(*rainfall.dims)
