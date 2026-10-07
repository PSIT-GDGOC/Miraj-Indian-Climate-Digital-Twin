import os
import geopandas as gpd
import numpy as np
from matplotlib.path import Path

def get_all_states():
    """Returns a sorted list of all unique state names from the GeoJSON."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    geojson_path = os.path.join(base_dir, "data", "india_states.geojson")
    gdf = gpd.read_file(geojson_path)
    return sorted(gdf['st_nm'].unique().tolist())

def get_compound_path(geometry):
    """Converts Shapely geometries (including holes) to a matplotlib compound Path."""
    vertices = []
    codes = []
    
    # Handle MultiPolygons natively
    polygons = geometry.geoms if hasattr(geometry, 'geoms') else [geometry]
    
    for poly in polygons:
        # Extract exterior ring
        ext_coords = np.array(poly.exterior.coords)
        vertices.extend(ext_coords)
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(ext_coords) - 2) + [Path.CLOSEPOLY])
        
        # Extract interior rings (holes)
        for interior in poly.interiors:
            int_coords = np.array(interior.coords)
            vertices.extend(int_coords)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(int_coords) - 2) + [Path.CLOSEPOLY])
            
    return Path(vertices, codes)

def get_outline_coords(geometry):
    """Returns lists of coordinates separated by NaN for Plotly vector plotting."""
    lon_lines = []
    lat_lines = []
    
    polygons = geometry.geoms if hasattr(geometry, 'geoms') else [geometry]
    
    for poly in polygons:
        ext_coords = np.array(poly.exterior.coords)
        lon_lines.extend(ext_coords[:, 0].tolist() + [np.nan])
        lat_lines.extend(ext_coords[:, 1].tolist() + [np.nan])
        
        for interior in poly.interiors:
            int_coords = np.array(interior.coords)
            lon_lines.extend(int_coords[:, 0].tolist() + [np.nan])
            lat_lines.extend(int_coords[:, 1].tolist() + [np.nan])
            
    return lon_lines, lat_lines

def mask_region_boundary_local(data_array, state_name):
    """Sets grid points outside the GeoJSON polygon to NaN and returns spatial metadata."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    geojson_path = os.path.join(base_dir, "data", "india_states.geojson")
    gdf = gpd.read_file(geojson_path)
    
    # Adjust column name ('st_nm', 'state', etc.) based on your specific GeoJSON
    state_geom = gdf[gdf['st_nm'] == state_name].geometry.iloc[0]
    compound_path = get_compound_path(state_geom)
    
    minx, miny, maxx, maxy = state_geom.bounds
    centroid_lon, centroid_lat = state_geom.centroid.x, state_geom.centroid.y
    
    # Create flattened meshgrid of the original coordinates
    lon, lat = np.meshgrid(data_array.lon.values, data_array.lat.values)
    points = np.vstack((lon.flatten(), lat.flatten())).T
    
    # Test containment and reshape to mask
    valid_mask = compound_path.contains_points(points).reshape(lon.shape)
    
    # Apply mask, maintaining the xarray structure
    masked_data = data_array.where(valid_mask, np.nan)
    
    # Reorder bounds to (min_lon, max_lon, min_lat, max_lat)
    bounds_dict = (minx, maxx, miny, maxy)
    centroid_dict = (centroid_lon, centroid_lat)
    
    # Get outline coords
    outline_coords = get_outline_coords(state_geom)
    
    return masked_data, bounds_dict, centroid_dict, outline_coords
