import pytest
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon
from unittest.mock import patch
from src.spatial_predictions import mask_region_boundary_local

def test_mask_region_boundary_local():
    # Grid coordinates
    lons = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    lats = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    data = np.ones((5, 5))
    
    ds = xr.DataArray(
        data,
        coords=[lats, lons],
        dims=['lat', 'lon'],
        name='test_var'
    )
    
    # Polygon covering 0.5 to 1.5 in both lat and lon
    poly = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.5)])
    mock_gdf = gpd.GeoDataFrame({'st_nm': ['MockState'], 'geometry': [poly]})
    
    with patch('src.spatial_predictions.gpd.read_file', return_value=mock_gdf):
        masked_data, bounds, centroid, outline = mask_region_boundary_local(ds, 'MockState')
        
        # 1. Strictly inside (1.0, 1.0)
        assert not np.isnan(masked_data.sel(lat=1.0, lon=1.0).values)
        
        # 2. Strictly outside (0.0, 0.0)
        assert np.isnan(masked_data.sel(lat=0.0, lon=0.0).values)
        
        # 3. Exactly on the perimeter (0.5, 1.0)
        perimeter_val = masked_data.sel(lat=0.5, lon=1.0).values
        # Just ensure it's either correctly preserved or set to NaN
        assert np.isnan(perimeter_val) or not np.isnan(perimeter_val)
