import os
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import xarray as xr

# Ensure root directory is in sys.path for internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.spatial_predictions import mask_region_boundary_local, get_all_states
from src.climate_alerts import assess_climate_alert
from src.visualizations import build_state_heatmap
from src.climate_indices import calculate_spi, classify_spi, assess_drought_condition

st.set_page_config(
    layout="wide",
    page_title="Miraj Climate Digital Twin",
    page_icon="🌦️",
)

st.markdown(
    """
    <style>
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(30, 136, 229, 0.08), rgba(0, 172, 193, 0.04));
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 1rem;
            border-radius: 0.75rem;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            line-height: 1.4;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_data(file_path: str = None) -> xr.Dataset:
    """
    Load NetCDF climate dataset from disk with resource caching.

    Uses @st.cache_resource to cache the open xarray Dataset instance across reruns
    without serialization overhead. Handles file absence and engine/backend parsing
    errors gracefully, rendering informative user messages and halting execution safely.
    """
    if file_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "data", "sample", "sample_rainfall.nc")

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file missing at: {file_path}")

        return xr.open_dataset(file_path)

    except FileNotFoundError:
        st.error("Dataset missing. Run scripts/decoder.py first.")
        st.stop()

    except (ValueError, OSError) as exc:
        st.error(
            f"Unable to parse NetCDF dataset. The file may be corrupt or a required engine (netcdf4 or scipy) is missing. Error: {exc}"
        )
        st.stop()

    except Exception as exc:
        st.error(f"Unexpected error loading dataset: {exc}")
        st.stop()


@st.cache_data
def load_states():
    """Retrieve and cache all available state/UT names from geojson."""
    return get_all_states()


@st.cache_data
def get_time_series(_full_ds, state_name, var_name):
    """Compute and cache temporal mean time-series for a given state and variable."""
    masked_full, _, _, _ = mask_region_boundary_local(_full_ds[var_name], state_name)
    return masked_full.mean(dim=["lat", "lon"]).to_dataframe().reset_index()


@st.cache_data
def get_national_data(_full_ds, date_str, var_name):
    """Aggregate state-level mean and peak statistics across India for the selected date."""
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
                "State / UT": st_name,
                "Mean": mean_val,
                "Peak": max_val,
            })
    return pd.DataFrame(records).dropna(subset=["Mean"])


def reset_filters():
    """Restore dashboard widgets to their initial selections."""
    for key in ("state_filter", "date_filter", "variable_filter", "grid_threshold"):
        st.session_state.pop(key, None)


st.title("Miraj · Indian Climate Digital Twin")
st.caption(
    "Explore boundary-masked climate conditions across Indian states and union territories. "
    "This MVP currently uses the bundled seven-day sample dataset."
)

# Load dataset safely; fails gracefully via st.error + st.stop on missing file or engine error
ds = load_data()

states_list = load_states()

# Variable mappings
VAR_MAP = {
    "Rainfall (mm)": {"key": "rainfall", "colorscale": "Blues", "units": "mm"},
    "Max Temperature (°C)": {"key": "temperature", "colorscale": "Turbo", "units": "°C"},
    "Relative Humidity (%)": {"key": "humidity", "colorscale": "Teal", "units": "%"},
    "Wind Speed (km/h)": {"key": "wind_speed", "colorscale": "Viridis", "units": "km/h"},
}

st.sidebar.title("Explore climate data")
st.sidebar.caption("Choose a region, date and climate variable. Every chart and export updates automatically.")
default_state_index = states_list.index("Karnataka") if "Karnataka" in states_list else 0
state = st.sidebar.selectbox(
    "Select State / UT",
    states_list,
    index=default_state_index,
    key="state_filter",
)
available_dates = ds.time.dt.strftime("%Y-%m-%d").values
date = st.sidebar.selectbox("Select Date", available_dates, index=len(available_dates) - 1, key="date_filter")
selected_var_label = st.sidebar.selectbox("Climate variable", list(VAR_MAP.keys()), key="variable_filter")
st.sidebar.button("Reset filters", on_click=reset_filters, width="stretch")
st.sidebar.divider()
st.sidebar.success(f"Dataset online · {len(available_dates)} daily observations")
st.sidebar.caption(f"Coverage: {available_dates[0]} to {available_dates[-1]}")

var_key = VAR_MAP[selected_var_label]["key"]

if var_key not in ds.variables:
    st.error(f"Variable '{var_key}' not found in the dataset. Available variables: {list(ds.data_vars)}")
    st.stop()

units = VAR_MAP[selected_var_label]["units"]
chart_color = {
    "temperature": "orange",
    "rainfall": "blue",
    "humidity": "teal",
    "wind_speed": "purple",
}.get(var_key, "blue")

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

    # Calculate change from the immediately preceding observation.
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
    st.metric("State Mean", f"{mean_val:.1f} {units}" if not np.isnan(mean_val) else "N/A")
with col2.container(border=True):
    st.metric("State Peak / Max", f"{max_val:.1f} {units}" if not np.isnan(max_val) else "N/A")
with col3.container(border=True):
    st.metric("Active Coverage %", f"{coverage_pct:.1f}%")
with col4.container(border=True):
    delta_display = f"{trend_delta:+.1f} {units}" if date_idx > 0 else "N/A"
    st.metric(
        "Change vs previous day",
        delta_display,
        delta=delta_display if date_idx > 0 else None,
    )

alert = assess_climate_alert(var_key, mean_val, max_val)
st.subheader("Climate Risk Screening")
alert_text = f"**{alert.title}.** {alert.message}  \n**Suggested next step:** {alert.action}"
if alert.level in {"extreme", "high"}:
    st.error(alert_text)
elif alert.level == "elevated":
    st.warning(alert_text)
elif alert.level == "unavailable":
    st.info(alert_text)
else:
    st.success(alert_text)
st.caption(
    "Screening indicator only—not an official warning. The current dashboard uses synthetic sample data; "
    "always consult IMD and local authorities for operational decisions."
)

st.divider()

# Spatial and tabular GIS analytics
tab1, tab2, tab3 = st.tabs([
    "State Climate Map",
    "National Overview",
    "Grid Data & Export",
])

with tab1:
    st.subheader(f"Boundary-masked view of {state}")
    st.caption(
        "Only grid cells inside the selected administrative boundary are shown. "
        "Hover over the map to inspect individual values."
    )
    state_map = build_state_heatmap(
        bbox_data,
        outline_coords,
        state,
        selected_var_label,
        units,
        VAR_MAP[selected_var_label]["colorscale"],
    )
    st.plotly_chart(state_map, theme="streamlit", width="stretch")

with tab2:
    st.subheader("National Overview")
    with st.spinner("Calculating national metrics..."):
        national_df = get_national_data(ds, date, var_key)

    if not national_df.empty:
        nat_max = float(national_df["Peak"].max())

        def get_category(val):
            if var_key == "rainfall":
                if val > 64.5:
                    return "Very Heavy (>64.5 mm)"
                elif val > 35.4:
                    return "Heavy (35.5-64.4 mm)"
                elif val >= 7.5:
                    return "Moderate (7.5-35.4 mm)"
                else:
                    return "Light/Dry (<7.5 mm)"
            else:
                return "Available"

        national_df["Category / Status"] = national_df["Mean"].apply(get_category)
        national_df["Relative Intensity"] = national_df["Mean"]

        st.dataframe(
            national_df,
            width="stretch",
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
                "Category / Status": st.column_config.TextColumn("Category / Status"),
            },
        )

        national_csv = national_df.drop(columns=["Relative Intensity"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download national summary as CSV",
            data=national_csv,
            file_name=f"india_{date}_{var_key}_summary.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.info("No national data is available for this date and variable.")

with tab3:
    st.subheader(f"Grid-Level Inspection: {state}")

    df_masked = masked_data.to_dataframe().reset_index()
    df_valid = df_masked.dropna(subset=[var_key]).copy()

    if not df_valid.empty:
        min_thr = float(df_valid[var_key].min())
        max_thr = float(df_valid[var_key].max())

        if min_thr < max_thr:
            threshold = st.slider(
                f"Filter cells by minimum {selected_var_label}",
                min_value=min_thr,
                max_value=max_thr,
                value=min_thr,
                key="grid_threshold",
            )
            df_filtered = df_valid[df_valid[var_key] >= threshold]
        else:
            df_filtered = df_valid

        # Format dataframe for display
        display_df = df_filtered[["lat", "lon", var_key]].rename(columns={
            "lat": "Latitude", "lon": "Longitude", var_key: f"Value ({units})",
        })

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Latitude": st.column_config.NumberColumn("Latitude", format="%.2f°"),
                "Longitude": st.column_config.NumberColumn("Longitude", format="%.2f°"),
                f"Value ({units})": st.column_config.NumberColumn(f"Value ({units})", format="%.2f"),
            },
        )

        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered State Data as CSV",
            data=csv,
            file_name=f"{state}_{date}_{var_key}_grid.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.info("No valid grid cells found for the selected state and date.")

st.divider()

# Analytics & Temporal Trends
st.subheader("Analytics & Temporal Trends")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown(f"**Available history ({selected_var_label})**")
    ts_df = get_time_series(ds, state, var_key)
    fig_ts = px.line(ts_df, x="time", y=var_key, markers=True, title=f"{state} - Mean Trend")
    fig_ts.update_traces(line_color=chart_color, marker_color=chart_color)
    fig_ts.update_layout(yaxis_title=f"Mean ({units})", xaxis_title="Date")
    st.plotly_chart(fig_ts, theme="streamlit", width="stretch")
    st.download_button(
        label="Download state time series as CSV",
        data=ts_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{state}_{var_key}_time_series.csv",
        mime="text/csv",
        width="stretch",
    )

with chart_col2:
    st.markdown("**Distribution Spread**")
    if not df_valid.empty:
        fig_hist = px.histogram(
            df_valid,
            x=var_key,
            nbins=30,
            title=f"Grid-Cell Frequency on {date}",
            labels={var_key: selected_var_label},
            color_discrete_sequence=[chart_color],
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, theme="streamlit", width="stretch")
    else:
        st.info("Insufficient data for distribution plot.")

st.divider()

# Drought Conditions & Standardized Precipitation Index (SPI)
st.subheader("Drought Conditions")
st.caption(
    "Meteorological and hydrological drought monitoring using the World Meteorological Organization (WMO) "
    "Standardized Precipitation Index (SPI) across 30-day (agricultural) and 90-day (hydrological) timescales."
)

total_observations = len(available_dates)
min_required_obs = 30

if total_observations < min_required_obs:
    st.info(
        f"**Historical observations required for drought index.** SPI calculations require at least "
        f"{min_required_obs} consecutive daily observations to fit Gamma distribution parameters without "
        f"statistical bias. The active session is currently running on the bundled {total_observations}-day "
        f"demonstration dataset ({available_dates[0]} to {available_dates[-1]}).  \n\n"
        f"**Next Steps:** Decode extended historical archives using `scripts/decoder.py` to activate "
        f"30-day and 90-day SPI spatial heatmaps and anomaly tracking."
    )
else:
    if "rainfall" not in ds.variables:
        st.warning("Precipitation variable 'rainfall' not found in dataset; SPI cannot be computed.")
    else:
        drought_col1, drought_col2 = st.columns([1, 3])
        with drought_col1:
            spi_window = st.radio(
                "Accumulation window",
                [30, 90],
                format_func=lambda w: f"{w}-day window ({'agricultural' if w == 30 else 'hydrological'})",
                key="spi_window_selection",
            )

        with st.spinner(f"Computing {spi_window}-day gridded SPI..."):
            spi_grid = calculate_spi(ds["rainfall"], window=spi_window, min_samples=min_required_obs)
            day_spi = spi_grid.sel(time=date)
            masked_spi, spi_bounds, _, spi_outline = mask_region_boundary_local(day_spi, state)
            spi_min_lon, spi_max_lon, spi_min_lat, spi_max_lat = spi_bounds
            spi_bbox = masked_spi.where(
                (masked_spi.lon >= spi_min_lon) & (masked_spi.lon <= spi_max_lon), drop=True
            ).where(
                (masked_spi.lat >= spi_min_lat) & (masked_spi.lat <= spi_max_lat), drop=True
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            spi_valid = ~np.isnan(spi_bbox.values)
            spi_mean = float(np.nanmean(spi_bbox.values)) if spi_valid.any() else np.nan
            spi_min = float(np.nanmin(spi_bbox.values)) if spi_valid.any() else np.nan

        assessment = assess_drought_condition(spi_mean, window=spi_window)

        scol1, scol2, scol3 = st.columns(3)
        with scol1.container(border=True):
            st.metric("State Mean SPI", f"{spi_mean:.2f}" if not np.isnan(spi_mean) else "N/A")
        with scol2.container(border=True):
            st.metric("Peak Drought Intensity (Min SPI)", f"{spi_min:.2f}" if not np.isnan(spi_min) else "N/A")
        with scol3.container(border=True):
            st.metric("Drought Category", assessment.category)

        if assessment.severity in {"extreme_drought", "severe_drought"}:
            st.error(f"**{assessment.category}:** {assessment.message}  \n**Recommended Action:** {assessment.recommendation}")
        elif assessment.severity == "moderate_drought":
            st.warning(f"**{assessment.category}:** {assessment.message}  \n**Recommended Action:** {assessment.recommendation}")
        elif assessment.severity == "unavailable":
            st.info(f"**{assessment.category}:** {assessment.message}")
        else:
            st.success(f"**{assessment.category}:** {assessment.message}")

        if spi_valid.any():
            spi_map = build_state_heatmap(
                spi_bbox,
                spi_outline,
                state,
                f"{spi_window}-Day SPI",
                "index",
                "RdBu",
            )
            st.plotly_chart(spi_map, theme="streamlit", width="stretch")

            df_spi_masked = masked_spi.to_dataframe().reset_index()
            df_spi_valid = df_spi_masked.dropna(subset=[f"spi_{spi_window}"]).copy()
            df_spi_display = df_spi_valid[["lat", "lon", f"spi_{spi_window}"]].rename(columns={
                "lat": "Latitude", "lon": "Longitude", f"spi_{spi_window}": f"SPI_{spi_window}",
            })
            st.download_button(
                label=f"Download {state} {spi_window}-day SPI as CSV",
                data=df_spi_display.to_csv(index=False).encode("utf-8"),
                file_name=f"{state}_{date}_spi_{spi_window}.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            st.info(
                f"No valid {spi_window}-day SPI values could be calculated for {state} on {date}. "
                f"Ensure the dataset includes at least {spi_window} consecutive daily observations prior to this date."
            )
