import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import xarray as xr
import numpy as np
import plotly.express as px
import pandas as pd
import warnings
from src.spatial_predictions import mask_region_boundary_local, get_all_states

st.set_page_config(layout="wide", page_title="GIS Analytics Dashboard")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "sample", "sample_rainfall.nc")
    if not os.path.exists(file_path):
        return None
    return xr.open_dataset(file_path)

@st.cache_data
def load_states():
    return get_all_states()

@st.cache_data
def get_time_series(_full_ds, state_name, var_name):
    masked_full, _, _, _ = mask_region_boundary_local(_full_ds[var_name], state_name)
    return masked_full.mean(dim=['lat', 'lon']).to_dataframe().reset_index()

@st.cache_data
def get_national_data(_full_ds, date_str, var_name):
    states = get_all_states()
    day_data = _full_ds.sel(time=date_str)[var_name]
    records = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for st_name in states:
            masked, _, _, _ = mask_region_boundary_local(day_data, st_name)
            valid_mask = ~np.isnan(masked.values)
            if valid_mask.any():
                mean_val = float(np.nanmean(masked.values))
                max_val = float(np.nanmax(masked.values))
            else:
                mean_val = np.nan
                max_val = np.nan
                
            records.append({
                'State / UT': st_name,
                'Mean': mean_val,
                'Peak': max_val
            })
    return pd.DataFrame(records).dropna(subset=['Mean'])

st.title("Interactive GIS Climate Dashboard")

ds = load_data()

if ds is None:
    st.warning("Missing .nc file. Please check if data/sample/sample_rainfall.nc exists.")
else:
    states_list = load_states()
    
    # Variable mappings
    VAR_MAP = {
        "Rainfall (mm)": {"key": "rainfall", "colorscale": "Blues", "units": "mm"},
        "Max Temperature (°C)": {"key": "temperature", "colorscale": "Turbo", "units": "°C"},
        "Relative Humidity (%)": {"key": "humidity", "colorscale": "Teal", "units": "%"},
        "Wind Speed (km/h)": {"key": "wind_speed", "colorscale": "Viridis", "units": "km/h"}
    }
    
    st.sidebar.header("Filters")
    state = st.sidebar.selectbox("Select State", states_list)
    available_dates = ds.time.dt.strftime('%Y-%m-%d').values
    date = st.sidebar.selectbox("Select Date", available_dates)
    selected_var_label = st.sidebar.selectbox("Variable", list(VAR_MAP.keys()))
    
    var_key = VAR_MAP[selected_var_label]["key"]
    
    if var_key not in ds.variables:
        st.error(f"Variable '{var_key}' not found in the dataset. Available variables: {list(ds.data_vars)}")
        st.stop()

    units = VAR_MAP[selected_var_label]["units"]
    chart_color = {"temperature": "orange", "rainfall": "blue", "humidity": "teal", "wind_speed": "purple"}.get(var_key, "blue")

    # Slice data and mask for the selected state and date
    day_data = ds.sel(time=date)[var_key]
    masked_data, bounds, centroid, outline_coords = mask_region_boundary_local(day_data, state)
    min_lon, max_lon, min_lat, max_lat = bounds
    
    # Extract bounding box region
    lon_mask = (masked_data.lon >= min_lon) & (masked_data.lon <= max_lon)
    lat_mask = (masked_data.lat >= min_lat) & (masked_data.lat <= max_lat)
    bbox_data = masked_data.where(lon_mask, drop=True).where(lat_mask, drop=True)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        valid_cells = ~np.isnan(bbox_data.values)
        n_valid = valid_cells.sum()
        mean_val = np.nanmean(bbox_data.values) if n_valid > 0 else np.nan
        max_val = np.nanmax(bbox_data.values) if n_valid > 0 else np.nan
        
        # Calculate trend delta
        date_idx = list(available_dates).index(date)
        if date_idx > 0:
            prev_date = available_dates[date_idx - 1]
            prev_day_data = ds.sel(time=prev_date)[var_key]
            prev_masked_data, _, _, _ = mask_region_boundary_local(prev_day_data, state)
            prev_bbox = prev_masked_data.where(lon_mask, drop=True).where(lat_mask, drop=True)
            prev_mean = np.nanmean(prev_bbox.values) if ~np.isnan(prev_bbox.values).all() else np.nan
            trend_delta = mean_val - prev_mean if not np.isnan(mean_val) and not np.isnan(prev_mean) else 0.0
        else:
            trend_delta = 0.0

    # Coverage
    coverage_pct = (n_valid / bbox_data.size * 100) if bbox_data.size > 0 else 0.0
    if var_key == "rainfall" and n_valid > 0:
        coverage_pct = ((bbox_data.values > 0) & valid_cells).sum() / n_valid * 100

    # Header
    st.header(f"{state} - {selected_var_label} Overview ({date})")
    
    # KPI Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1.container(border=True):
        st.metric("State Mean", f"{mean_val:.1f} {units}" if not np.isnan(mean_val) else f"N/A")
    with col2.container(border=True):
        st.metric("State Peak / Max", f"{max_val:.1f} {units}" if not np.isnan(max_val) else f"N/A")
    with col3.container(border=True):
        st.metric("Active Coverage %", f"{coverage_pct:.1f}%")
    with col4.container(border=True):
        st.metric("7-Day Trend Delta", f"{trend_delta:+.1f} {units}" if trend_delta != 0 else f"- {units}", delta=f"{trend_delta:+.1f} {units}")

    st.divider()
    
    # Tabular GIS Analytics
    tab1, tab2 = st.tabs(["National Overview & Leaderboard", "State Grid-Level Explorer"])
    
    with tab1:
        st.subheader("National Overview")
        with st.spinner("Calculating national metrics..."):
            national_df = get_national_data(ds, date, var_key)
        
        if not national_df.empty:
            nat_max = float(national_df['Peak'].max())
            
            def get_category(val):
                if var_key == "rainfall":
                    if val > 64.5: return "Very Heavy (>64.5 mm)"
                    elif val > 35.4: return "Heavy (35.5-64.4 mm)"
                    elif val >= 7.5: return "Moderate (7.5-35.4 mm)"
                    else: return "Light/Dry (<7.5 mm)"
                else:
                    return "Available"

            national_df['Category / Status'] = national_df['Mean'].apply(get_category)
            national_df['Relative Intensity'] = national_df['Mean']
            
            st.dataframe(
                national_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "State / UT": st.column_config.TextColumn("State / UT"),
                    "Mean": st.column_config.NumberColumn(f"Mean ({units})", format="%.2f"),
                    "Peak": st.column_config.NumberColumn(f"Peak ({units})", format="%.2f"),
                    "Relative Intensity": st.column_config.ProgressColumn(
                        "Relative Intensity",
                        help="Relative to national peak",
                        format="%.2f",
                        min_value=0,
                        max_value=nat_max,
                    ),
                    "Category / Status": st.column_config.TextColumn("Category / Status")
                }
            )

    with tab2:
        st.subheader(f"Grid-Level Inspection: {state}")
        
        df_masked = masked_data.to_dataframe().reset_index()
        df_valid = df_masked.dropna(subset=[var_key]).copy()
        
        if not df_valid.empty:
            min_thr = float(df_valid[var_key].min())
            max_thr = float(df_valid[var_key].max())
            
            if min_thr < max_thr:
                threshold = st.slider(f"Filter cells by minimum {selected_var_label}", min_value=min_thr, max_value=max_thr, value=min_thr)
                df_filtered = df_valid[df_valid[var_key] >= threshold]
            else:
                df_filtered = df_valid
            
            # Format dataframe for display
            display_df = df_filtered[['lat', 'lon', var_key]].rename(columns={
                'lat': 'Latitude', 'lon': 'Longitude', var_key: f'Value ({units})'
            })
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Latitude": st.column_config.NumberColumn("Latitude", format="%.2f°"),
                    "Longitude": st.column_config.NumberColumn("Longitude", format="%.2f°"),
                    f"Value ({units})": st.column_config.NumberColumn(f"Value ({units})", format="%.2f")
                }
            )
            
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Filtered State Data as CSV",
                data=csv,
                file_name=f"{state}_{date}_{var_key}_grid.csv",
                mime="text/csv",
            )
        else:
            st.info("No valid grid cells found for the selected state and date.")

    st.divider()

    # Analytics & Temporal Trends
    st.subheader("Analytics & Temporal Trends")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown(f"**7-Day History ({selected_var_label})**")
        ts_df = get_time_series(ds, state, var_key)
        fig_ts = px.line(ts_df, x='time', y=var_key, markers=True, title=f"{state} - Mean Trend")
        fig_ts.update_traces(line_color=chart_color, marker_color=chart_color)
        st.plotly_chart(fig_ts, theme="streamlit", use_container_width=True)

    with chart_col2:
        st.markdown("**Distribution Spread**")
        if not df_valid.empty:
            fig_hist = px.histogram(
                df_valid,
                x=var_key,
                nbins=30,
                title=f"Grid-Cell Frequency on {date}",
                labels={var_key: selected_var_label},
                color_discrete_sequence=[chart_color]
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, theme="streamlit", use_container_width=True)
        else:
            st.info("Insufficient data for distribution plot.")
