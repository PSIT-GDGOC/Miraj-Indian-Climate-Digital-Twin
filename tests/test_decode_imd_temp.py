"""Tests for the IMD binary temperature decoder."""

import numpy as np
import pytest
import xarray as xr

from scripts.decode_imd_temp import (
    GRID_SHAPE,
    decode_temperature_grd,
    stack_daily_files,
)


def test_decode_max_temp_grd_handles_missing_values_and_metadata(tmp_path):
    """Verify max_temp shape, coordinates, metadata, and missing-value conversion."""
    grid = np.full(GRID_SHAPE, 32.5, dtype=np.float32)
    grid[0, 0] = -999.0
    grid[15, 15] = -999.0

    file_path = tmp_path / "temperature_20240701.grd"
    grid.tofile(file_path)

    result = decode_temperature_grd(file_path, variable="max_temp")

    # Shape and dimensions
    assert result.shape == (31, 31)
    assert result.dims == ("lat", "lon")

    # Coordinate boundaries
    assert np.isclose(result.lat.values[0], 7.5)
    assert np.isclose(result.lat.values[-1], 37.5)
    assert np.isclose(result.lon.values[0], 67.5)
    assert np.isclose(result.lon.values[-1], 97.5)

    # Missing value handling
    assert np.isnan(result.values[0, 0])
    assert np.isnan(result.values[15, 15])
    assert np.isclose(result.values[0, 1], 32.5)
    assert not np.any(np.isclose(result.values, -999.0, equal_nan=False))

    # Metadata
    assert result.name == "max_temp"
    assert result.attrs["units"] == "degC"
    assert result.attrs["long_name"] == "Daily Maximum Temperature"

    # Opens successfully with xarray
    nc_path = tmp_path / "decoded_max_temp.nc"
    result.to_dataset().to_netcdf(nc_path)
    with xr.open_dataset(nc_path) as ds:
        assert "max_temp" in ds.data_vars
        assert ds["max_temp"].shape == (31, 31)
        assert ds["max_temp"].attrs["units"] == "degC"
        assert np.isclose(ds["lat"].values[0], 7.5)
        assert np.isclose(ds["lon"].values[-1], 97.5)


def test_decode_min_temp_grd_metadata(tmp_path):
    """Verify min_temp variable name and metadata attributes."""
    grid = np.full(GRID_SHAPE, 21.0, dtype=np.float32)
    file_path = tmp_path / "temperature_min_20240701.grd"
    grid.tofile(file_path)

    result = decode_temperature_grd(file_path, variable="min_temp")

    assert result.shape == (31, 31)
    assert result.name == "min_temp"
    assert result.attrs["units"] == "degC"
    assert result.attrs["long_name"] == "Daily Minimum Temperature"
    assert np.all(np.isclose(result.values, 21.0))


def test_decode_temperature_grd_invalid_variable(tmp_path):
    """Verify that an unsupported variable name raises ValueError."""
    grid = np.zeros(GRID_SHAPE, dtype=np.float32)
    file_path = tmp_path / "test.grd"
    grid.tofile(file_path)

    with pytest.raises(ValueError, match="Unsupported variable 'invalid_var'"):
        decode_temperature_grd(file_path, variable="invalid_var")


def test_decode_temperature_grd_incorrect_file_size(tmp_path):
    """Verify that a file with incorrect element count raises ValueError."""
    wrong_size_grid = np.ones(100, dtype=np.float32)
    file_path = tmp_path / "corrupt_temperature.grd"
    wrong_size_grid.tofile(file_path)

    with pytest.raises(ValueError, match="contains 100 values; expected 961"):
        decode_temperature_grd(file_path, variable="max_temp")


def test_stack_daily_temperature_files(tmp_path):
    """Verify stacking multiple daily temperature files into a Dataset."""
    grid_day1 = np.full(GRID_SHAPE, 30.0, dtype=np.float32)
    grid_day2 = np.full(GRID_SHAPE, 32.0, dtype=np.float32)

    (tmp_path / "temperature_20240701.grd").write_bytes(grid_day1.tobytes())
    (tmp_path / "temperature_20240702.grd").write_bytes(grid_day2.tobytes())

    dataset = stack_daily_files(tmp_path, variable="max_temp")

    assert "max_temp" in dataset.data_vars
    assert dataset.sizes["time"] == 2
    assert dataset.sizes["lat"] == 31
    assert dataset.sizes["lon"] == 31
    assert dataset["max_temp"].attrs["units"] == "degC"
    assert dataset.attrs["Conventions"] == "CF-1.8"

    # Verify export to NetCDF and re-opening with xarray
    nc_path = tmp_path / "stacked.nc"
    dataset.to_netcdf(nc_path)
    with xr.open_dataset(nc_path) as reloaded:
        assert reloaded.sizes["time"] == 2
        assert np.isclose(reloaded["max_temp"].isel(time=0).values[0, 0], 30.0)
        assert np.isclose(reloaded["max_temp"].isel(time=1).values[0, 0], 32.0)


def test_cli_decode_imd_temp(tmp_path, monkeypatch):
    """Verify the CLI entrypoint generates a valid NetCDF file."""
    grid = np.full(GRID_SHAPE, 28.0, dtype=np.float32)
    input_file = tmp_path / "temperature_20240701.grd"
    output_file = tmp_path / "output.nc"
    grid.tofile(input_file)

    from scripts.decode_imd_temp import main

    test_args = [
        "decode_imd_temp.py",
        "--input",
        str(input_file),
        "--output",
        str(output_file),
        "--variable",
        "max_temp",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()

    assert output_file.exists()
    with xr.open_dataset(output_file) as ds:
        assert "max_temp" in ds.data_vars
        assert ds.sizes["time"] == 1
        assert ds.sizes["lat"] == 31
        assert ds.sizes["lon"] == 31
