
"""Tests for the IMD temperature decoder."""

import numpy as np
import pytest

from scripts.decode_imd_temp import decode_temperature_grd


def create_grd_file(tmp_path, values):
    """Create a synthetic float32 GRD file."""
    path = tmp_path / "temperature.grd"
    np.asarray(values, dtype=np.float32).tofile(path)
    return path


@pytest.mark.parametrize(
    ("variable", "expected_name"),
    [
        ("max_temp", "Daily Maximum Temperature"),
        ("min_temp", "Daily Minimum Temperature"),
    ],
)
def test_decode_temperature_metadata(
    tmp_path, variable, expected_name
):
    values = np.arange(31 * 31, dtype=np.float32)
    path = create_grd_file(tmp_path, values)

    result = decode_temperature_grd(path, variable)

    assert result.shape == (31, 31)
    assert result.dims == ("lat", "lon")
    assert result.name == variable

    assert result.lat.values[0] == pytest.approx(7.5)
    assert result.lat.values[-1] == pytest.approx(37.5)
    assert result.lon.values[0] == pytest.approx(67.5)
    assert result.lon.values[-1] == pytest.approx(97.5)

    assert result.attrs["units"] == "degC"
    assert result.attrs["long_name"] == expected_name


def test_missing_value_conversion(tmp_path):
    values = np.ones(31 * 31, dtype=np.float32)
    values[0] = -999.0
    path = create_grd_file(tmp_path, values)

    result = decode_temperature_grd(path, "max_temp")

    assert np.isnan(result.values[0, 0])
    assert result.values[0, 1] == pytest.approx(1.0)


def test_invalid_variable(tmp_path):
    path = create_grd_file(
        tmp_path, np.ones(31 * 31, dtype=np.float32)
    )

    with pytest.raises(ValueError, match="Invalid variable"):
        decode_temperature_grd(path, "rainfall")


@pytest.mark.parametrize("size", [0, 100, 960, 962])
def test_incorrect_binary_file_size(tmp_path, size):
    path = create_grd_file(
        tmp_path, np.ones(size, dtype=np.float32)
    )

    with pytest.raises(ValueError, match="contains .* values"):
        decode_temperature_grd(path, "max_temp")


def test_output_opens_with_xarray(tmp_path):
    path = create_grd_file(
        tmp_path, np.ones(31 * 31, dtype=np.float32)
    )
    output = tmp_path / "temperature.nc"

    data = decode_temperature_grd(path, "max_temp")
    data.to_dataset().to_netcdf(output)

    with __import__("xarray").open_dataset(output) as dataset:
        assert dataset["max_temp"].shape == (31, 31)

