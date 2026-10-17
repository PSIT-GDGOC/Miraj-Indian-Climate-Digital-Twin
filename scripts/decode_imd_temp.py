"""Decode IMD daily maximum and minimum temperature binary files into NetCDF."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


GRID_SHAPE = (31, 31)
LATITUDES = np.linspace(7.5, 37.5, GRID_SHAPE[0])
LONGITUDES = np.linspace(67.5, 97.5, GRID_SHAPE[1])
MISSING_VALUES = (-999.0,)

SUPPORTED_VARIABLES = {
    "max_temp": {
        "units": "degC",
        "long_name": "Daily Maximum Temperature",
    },
    "min_temp": {
        "units": "degC",
        "long_name": "Daily Minimum Temperature",
    },
}


def decode_temperature_grd(path, variable: str) -> xr.DataArray:
    """Decode one IMD maximum or minimum temperature GRD file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a binary file containing 31 * 31 float32 values.
    variable : str
        Variable name to decode ('max_temp' or 'min_temp').

    Returns
    -------
    xarray.DataArray
        Temperature grid with dimensions lat and lon. Missing-value flag
        -999.0 is represented as NaN.

    Raises
    ------
    ValueError
        If variable is not 'max_temp' or 'min_temp', or if the file does not
        contain exactly 31 * 31 float32 values.
    """
    if variable not in SUPPORTED_VARIABLES:
        raise ValueError(
            f"Unsupported variable '{variable}'. Supported variables are: "
            f"{', '.join(sorted(SUPPORTED_VARIABLES.keys()))}."
        )

    path = Path(path)
    values = np.fromfile(path, dtype=np.float32)

    expected_size = GRID_SHAPE[0] * GRID_SHAPE[1]
    if values.size != expected_size:
        raise ValueError(
            f"{path} contains {values.size} values; expected "
            f"{expected_size} for grid shape {GRID_SHAPE}."
        )

    grid = values.reshape(GRID_SHAPE).astype(np.float32, copy=True)

    missing_mask = np.zeros(grid.shape, dtype=bool)
    for missing_value in MISSING_VALUES:
        missing_mask |= np.isclose(
            grid, missing_value, atol=1e-4, rtol=0
        )
    grid[missing_mask] = np.nan

    var_meta = SUPPORTED_VARIABLES[variable]

    return xr.DataArray(
        grid,
        dims=("lat", "lon"),
        coords={
            "lat": (
                "lat",
                LATITUDES,
                {
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "units": "degrees_north",
                    "axis": "Y",
                },
            ),
            "lon": (
                "lon",
                LONGITUDES,
                {
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "units": "degrees_east",
                    "axis": "X",
                },
            ),
        },
        name=variable,
        attrs={
            "units": var_meta["units"],
            "long_name": var_meta["long_name"],
        },
    )


def stack_daily_files(input_path, variable: str = "max_temp") -> xr.Dataset:
    """Decode and combine daily temperature GRD files into a time-indexed dataset.

    Parameters
    ----------
    input_path : str or pathlib.Path
        A single .grd file or a directory containing daily .grd files.
        Filenames are sorted lexicographically.
    variable : str, default="max_temp"
        Variable name to decode ('max_temp' or 'min_temp').

    Returns
    -------
    xarray.Dataset
        Dataset with {variable}(time, lat, lon).
    """
    if variable not in SUPPORTED_VARIABLES:
        raise ValueError(
            f"Unsupported variable '{variable}'. Supported variables are: "
            f"{', '.join(sorted(SUPPORTED_VARIABLES.keys()))}."
        )

    input_path = Path(input_path)

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.grd"))
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not files:
        raise FileNotFoundError(f"No .grd files found in {input_path}")

    daily_arrays = [decode_temperature_grd(file, variable=variable) for file in files]

    # Use dates from filenames when they contain YYYYMMDD or YYYY-MM-DD.
    dates = []
    for file in files:
        date_text = file.stem.replace("-", "")
        parsed_date = pd.NaT

        for index in range(max(0, len(date_text) - 7)):
            candidate = date_text[index:index + 8]
            try:
                parsed_date = pd.to_datetime(candidate, format="%Y%m%d")
                break
            except ValueError:
                continue

        dates.append(parsed_date)

    # If filenames don't contain usable dates, raise error.
    if any(pd.isna(date) for date in dates):
        raise ValueError(
            "Could not extract a valid YYYYMMDD date from every filename. "
            f"Please use filenames containing dates, such as {variable}_20240701.grd."
        )

    times = pd.DatetimeIndex(dates)

    combined = xr.concat(daily_arrays, dim=pd.Index(times, name="time"))
    combined.name = variable

    combined["time"].attrs.update(
        {
            "standard_name": "time",
            "axis": "T",
        }
    )

    var_meta = SUPPORTED_VARIABLES[variable]
    combined.attrs.update(
        {
            "units": var_meta["units"],
            "long_name": var_meta["long_name"],
        }
    )

    dataset = combined.to_dataset()
    dataset.attrs.update(
        {
            "Conventions": "CF-1.8",
            "title": f"IMD Gridded Daily {var_meta['long_name']}",
            "source": "India Meteorological Department (IMD)",
        }
    )

    return dataset


def main():
    """Parse CLI arguments and save the decoded dataset as NetCDF4."""
    parser = argparse.ArgumentParser(
        description="Decode IMD daily temperature GRD files into NetCDF4."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to one .grd file or a directory of .grd files.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/IMD_Gridded_Temperature.nc",
        help="Output NetCDF file path.",
    )
    parser.add_argument(
        "--variable",
        default="max_temp",
        choices=list(SUPPORTED_VARIABLES.keys()),
        help="Temperature variable to decode: 'max_temp' or 'min_temp' (default: 'max_temp').",
    )
    args = parser.parse_args()

    dataset = stack_daily_files(args.input, variable=args.variable)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_netcdf(output_path, engine="netcdf4", format="NETCDF4")
    print(f"Decoded {dataset.sizes['time']} daily file(s) to {output_path}")


if __name__ == "__main__":
    main()
