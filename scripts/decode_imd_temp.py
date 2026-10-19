
"""Decode IMD daily maximum and minimum temperature GRD files."""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


GRID_SHAPE = (31, 31)

LATITUDES = np.linspace(7.5, 37.5, GRID_SHAPE[0])
LONGITUDES = np.linspace(67.5, 97.5, GRID_SHAPE[1])

MISSING_VALUE = -999.0

VARIABLE_METADATA = {
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
    """Decode one IMD maximum or minimum temperature GRD file."""

    if variable not in VARIABLE_METADATA:
        raise ValueError(
            f"Invalid variable '{variable}'. "
            "Expected 'max_temp' or 'min_temp'."
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

    # Replace the IMD missing-value flag with NaN.
    missing_mask = np.isclose(
        grid, MISSING_VALUE, atol=1e-4, rtol=0
    )
    grid[missing_mask] = np.nan

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
        attrs=VARIABLE_METADATA[variable].copy(),
    )


def main():
    """Decode a temperature GRD file and save it as NetCDF."""

    parser = argparse.ArgumentParser(
        description="Decode an IMD temperature GRD file into NetCDF."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input temperature GRD file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output NetCDF file.",
    )
    parser.add_argument(
        "--variable",
        required=True,
        choices=["max_temp", "min_temp"],
        help="Temperature variable to decode.",
    )

    args = parser.parse_args()

    data = decode_temperature_grd(args.input, args.variable)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data.to_dataset().to_netcdf(output_path)

    print(f"Decoded {args.variable} successfully.")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
