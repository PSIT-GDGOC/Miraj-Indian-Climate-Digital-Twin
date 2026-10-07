import xarray as xr
import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter

# Standard IMD 0.25 degree coordinate ranges
lons = np.linspace(66.5, 100.0, 135)
lats = np.linspace(6.5, 38.5, 129)
times = pd.date_range("2024-07-01", "2024-07-07")
shape = (len(times), len(lats), len(lons))

np.random.seed(42)

def generate_field(base_val, scale, filter_sigma, min_val, max_val):
    # Generate random noise for each time step
    noise = np.random.randn(*shape) * scale + base_val
    # Apply spatial smoothing (sigma=0 for time axis, filter_sigma for lat/lon)
    smoothed = gaussian_filter(noise, sigma=[0, filter_sigma, filter_sigma])
    # Normalize back to intended range roughly
    smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    smoothed = smoothed * (max_val - min_val) + min_val
    return np.clip(smoothed, min_val, max_val)

# Generate variables with continuous fields
rainfall = generate_field(0, 50, 4.0, 0, 80)
temperature = generate_field(30, 15, 3.0, 20, 45)
humidity = generate_field(60, 30, 3.5, 30, 95)
wind_speed = generate_field(15, 20, 2.5, 5, 55)

ds = xr.Dataset(
    {
        "rainfall": (["time", "lat", "lon"], rainfall, {"units": "mm", "long_name": "Daily Rainfall"}),
        "temperature": (["time", "lat", "lon"], temperature, {"units": "°C", "long_name": "Max Daily Temperature"}),
        "humidity": (["time", "lat", "lon"], humidity, {"units": "%", "long_name": "Relative Humidity"}),
        "wind_speed": (["time", "lat", "lon"], wind_speed, {"units": "km/h", "long_name": "Surface Wind Speed"}),
    },
    coords={"lon": lons, "lat": lats, "time": times}
)

os.makedirs("data/sample", exist_ok=True)
output_path = "data/sample/sample_rainfall.nc"
ds.to_netcdf(output_path)
print(f"Synthetic NetCDF created at {output_path}")
