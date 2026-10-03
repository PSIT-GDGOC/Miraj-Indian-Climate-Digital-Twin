## Miraj- India's Climate Digital Twin: 2D Spatiotemporal Geospatial Reanalysis and Prognostic Forecasting Platform

This repository houses the 2D Geospatial Reanalysis and Prognostic Forecasting Platform. The platform assimilates space-based remote sensing datasets from geostationary satellites (MOSDAC INSAT-3D/3DR) and ground-based observations (India Meteorological Department - IMD Pune) to model atmospheric and land-surface processes at high spatial and temporal resolutions.

## Architectural System Overview

The system is split into three core layers: a data engineering and decoding pipeline, a hybrid deep-learning-statistical forecasting engine, and an interactive GIS dashboard.

```mermaid
graph TD
    A[IMD Raw Binary .grd] -->|decode_imd_binary.py| B(CF-NetCDF4 .nc Archive)
    C[MOSDAC Geostationary H5] -->|download_and_decode.py| B
    B -->|mask_region_boundary_local| D[State Boundary Slices]
    D -->|ConvLSTM Anomaly Layer| E[Predictive Ensemble Blend]
    D -->|NOAA Spatial Analogs| E
    E -->|Mean Bias Correction| F[Final 2D Forecast Grid]
```

## Directory Structure and Repository Layout

```text
.
├── .streamlit/
│   └── config.toml              # Streamlit server static configuration parameters
├── app/
│   └── streamlit_app.py         # Main dashboard interface
├── checkpoints/
│   ├── climate_twin_convlstm_final.pth # PyTorch rainfall ConvLSTM model
│   └── climate_twin_convlstm_temp.pth  # PyTorch temperature ConvLSTM model
├── data/
│   ├── india_states.geojson     # Administrative boundary vector geometries
│   └── processed/               # CF-compliant NetCDF4 datasets (.nc)
├── scripts/
│   ├── decode_imd_binary.py     # Rainfall binary daily decoder
│   ├── decode_imd_temp.py       # Temperature binary daily decoder
│   ├── download_and_decode_all_real.py  # Ingestion orchestration runner
│   └── train_convlstm.py        # PyTorch ConvLSTM training loop
└── src/
    ├── api/
    │   └── main.py              # FastAPI consumer microservice gateway
    ├── models/
    │   └── pytorch_convlstm.py  # ConvLSTM layers and dimensions
    ├── spatial_predictions.py   # Spatiotemporal 5-layer forecasting engine
    ├── climate_indices.py       # Indices engine (Monsoon, SPI, CWSI, FFG)
    ├── teleconnections.py       # Index fetchers (ENSO, IOD, MJO)
    └── basin_analysis.py        # River basin spatial accumulation engine
```

---

## Data Engineering and Processing Pipeline


The ingestion layer translates raw, heterogeneous data formats into structured, multi-dimensional grids:

| Dataset File | Variable Name | Grid Resolution | Spatial Extent |
| :--- | :--- | :--- | :--- |
| `IMD_Gridded_Rainfall_0.25_Real_v4.nc` | `rainfall` | 0.25° × 0.25° | Lat 6.5°N–38.5°N \| Lon 66.5°E–100.0°E |
| `IMD_Gridded_MaxTemp_1.0_Real_v3.nc` | `max_temp` | 1.0° × 1.0° | Lat 7.5°N–37.5°N \| Lon 67.5°E–97.5°E |
| `IMD_Gridded_MinTemp_1.0_Real_v3.nc` | `min_temp` | 1.0° × 1.0° | Lat 7.5°N–37.5°N \| Lon 67.5°E–97.5°E |
| `MOSDAC_INSAT_LST_Real.nc` | `lst` | 0.1° × 0.1° | Bounded satellite swath |
| `MOSDAC_INSAT_Rainfall_Real.nc` | `rain` | 0.1° × 0.1° | Bounded satellite swath |


### 1. Daily Binary Decoding
* **Precipitation:** Reads IMD 0.25° gridded daily binary files (`.grd`). It reads a single-precision float array of size 129x135, georeferences it to the Indian subcontinent bounding box (Lat 6.5°N - 38.5°N | Lon 66.5°E - 100.0°E), and masks missing data flags (`99.9` or `-99.9` to `NaN`).
* **Temperature:** Reads IMD 1.0° gridded daily maximum and minimum binary files (`.grd`). It reads a single-precision float array of size 31x31, georeferences it to the bounding box (Lat 7.5°N - 37.5°N | Lon 67.5°E - 97.5°E), and masks invalid values (`-999.0` to `NaN`).
* **Storage Compilation:** Aggregates daily grids along the time axis into CF-compliant NetCDF4 (`.nc`) files.

### 2. Geostationary Satellite Integration
* Assimilates geostationary satellite telemetry from MOSDAC INSAT-3D/3DR (L2B SST, LST, and precipitation products in HDF5 format).
* Implements background scripts (`scripts/download_and_decode_all_real.py`) to scrape, regrid, and align satellite coordinates with the ground observation grid.

### 3. Strict Administrative Boundary Masking
* **Methodology:** Implemented in `src/spatial_predictions.py` via `mask_region_boundary_local`. It extracts administrative boundaries from `data/india_states.geojson` and constructs matplotlib `Path` vector polygons for the selected pilot state (e.g. Karnataka, Uttar Pradesh, Odisha).
* **Operation:** Evaluates a 2D coordinate grid of latitude and longitude coordinates. Points lying outside the boundary polygon are set to `NaN` using:
  ```python
  points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
  mask = path.contains_points(points).reshape(lat_grid.shape)
  data_array = data_array.where(mask, np.nan)
  ```
  This prevents grid calculations from leaking into neighboring states or oceans, maintaining regional integrity for localized water and agricultural modeling.

### 4. Data Ingestion Cron Orchestration
The ingestion pipeline automates daily file acquisition and regridding via Scheduled scripts (`scripts/download_and_decode_all_real.py`). The script:
1. Performs an HTTP GET query against IMD's daily reanalysis servers to retrieve the latest binary `.grd` records.
2. Applies a nearest-neighbor regridding interpolation to resolve coordinate spatial grids.
3. Appends the daily slices to the active, CF-compliant NetCDF4 archive on disk.
4. Updates NetCDF metadata parameters (time variables, scale factors, and add_offset values) to preserve historical data standards.

---

## Hybrid 5-Layer Forecast Engine


The spatial forecast engine (`src/spatial_predictions.py`) avoids compounding autoregressive drift through a hybrid 5-layer pipeline:

### Layer 1: PyTorch Spatio-Temporal ConvLSTM Anomaly Model
* **Architecture:** Swaps standard matrix multiplications in LSTM cells with 2D convolutions (kernel size 3x3, padding 1) to capture spatial correlations. Configured with 2 ConvLSTM layers (64 and 32 hidden dimensions).
* **Tensor Configuration:** Inputs are mapped as 5D tensors of shape:

$$\text{Input Shape} = [B, T, C, H, W] = [\text{Batch}, 10 \text{ days}, 1 \text{ channel}, 129 \text{ lat}, 135 \text{ lon}]$$

* **Anomaly-Space Mapping:** Rather than predicting absolute values (which leads to severe dampening), the network predicts anomalies relative to daily climatological means.
  * **Rainfall Anomaly Scaling:** Log-transformed to handle extreme events:

$$y = \text{sign}(x) \cdot \frac{\log(1 + |x|)}{3.0}$$

  * **Temperature Anomaly Scaling:** Standard-scaled:

$$y = \frac{x}{10.0}$$

* **Stochastic Uncertainty Estimation:** Implements Monte Carlo (MC) Dropout during testing. The model performs multiple stochastic forward passes ($N=5$) with active dropout layers to generate standard deviation grids representing forecast confidence bounds ($\pm1\sigma$).

### Layer 2: Daily Climatological Reanalysis Atlas
* Computes daily climatology profiles by grouping historical grid observations (2015-2023) by day-of-year:

$$\text{Climatology}(m, d) = \frac{1}{Y} \sum_{y=1}^{Y} \text{Observation}(y, m, d)$$

* Serves as a physical reference baseline to bound the neural network's predictions at longer horizons.

### Layer 3: NOAA CPC Spatial Analog Selection
* Identifies the three most spatially-similar historical 30-day windows by calculating the Pearson correlation coefficient ($r$) across valid (non-NaN) grid cells:

$$r = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2 \sum (Y_i - \bar{Y})^2}}$$

* Yields the top 3 analog years and extracts their subsequent actual trajectories (Day +1 to Day +7) to form a physical analog ensemble.

### Layer 4: Exponential Blending Schedule
* Blends the ConvLSTM neural anomaly prediction, the analog year ensemble mean, and the historical climatological grid.
* The neural component's weight decreases exponentially over the forecast window ($w_{\text{neural}} = 0.55 \cdot 0.88^t$), while the analog and climatology weights increase, stabilizing predictions up to Day +7.

### Layer 5: Mean Bias Correction (MBC)
* Calculates the spatial mean ratio of the climatological grid to the blended grid, applying the scaling factor to adjust regional biases:

$$\text{CorrectedGrid} = \text{BlendedGrid} \cdot \left( \frac{\text{Mean}(\text{ClimGrid})}{\text{Mean}(\text{BlendedGrid})} \right)$$

---
## Meteorological and Hydrological Indices

The engine (`src/climate_indices.py`) evaluates real-time data to derive high-level decision support indexes:

### 1. WMO Standardized Precipitation Index (SPI-30 & SPI-90)
* Measures meteorological drought and moisture surplus on 30-day and 90-day scales.
* Fits a two-parameter Gamma probability density function to the historical cumulative rainfall:

$$g(x) = \frac{1}{\beta^\alpha \Gamma(\alpha)} x^{\alpha - 1} e^{-x/\beta}$$

  where $\alpha$ is the shape parameter, $\beta$ is the scale parameter, and $\Gamma(\alpha)$ is the gamma function.
* Transforms the cumulative probability $G(x)$ to a standard normal distribution (mean 0, variance 1):

$$\text{SPI} = \Psi^{-1}(G(x))$$

* **Drought Categories:** Values $\le -1.0$ indicate moderate drought, $\le -1.5$ severe, and $\le -2.0$ extreme drought.

### 2. FAO-56 Crop Water Stress Index (CWSI)
* Estimates agricultural water stress by tracking actual vs. potential evapotranspiration ($ET$):

$$\text{CWSI} = 1.0 - \frac{ET_{\text{actual}}}{ET_{\text{potential}}}$$

  where $ET_{\text{potential}}$ is derived using the Penman-Monteith equation for reference crop evapotranspiration ($ET_0$).
* Indicates crop stress on a scale from 0.0 (no stress) to 1.0 (water-deficit stress).

### 3. NWS Flash Flood Guidance (FFG)
* Estimates the volume of rainfall (mm/day) required to initiate local flooding.
* Calculates soil capacity limits by evaluating observed precipitation accumulations relative to the active soil moisture saturation percentage ($SM_{\text{pct}}$):

$$\text{FFG} = \text{Capacity}_{\text{max}} \cdot (1.0 - SM_{\text{pct}})$$

### 4. Monsoon Onset and Northward Advance Tracker
* Evaluates waypoints using the IMD criteria: at least 5 consecutive days where the spatial mean daily rainfall over the waypoint exceeds $2.5\text{ mm/day}$, commencing after the earliest historical onset window.
* **Target Year Slicing:** Dynamically filters dataset records to the chosen target year. If the selection exceeds the dataset range (e.g. 2026), it falls back to the latest year of available observations (`2023`), updating all cards and headers dynamically.

### 5. IMD Meteorological Alert Thresholds
Monitors predicted grids against official India Meteorological Department thresholds to trigger warnings:
* **Severe Heatwave:** Triggered when the maximum temperature grid point exceeds $40^\circ\text{C}$ in plains or $37^\circ\text{C}$ in coastal areas, or when the positive temperature anomaly departs by $\ge 4.5^\circ\text{C}$ above climatology.
* **Heavy Rainfall:** Triggered based on daily accumulation thresholds:
  * **Yellow Warning (Heavy):** Precipitation rate between $64.5\text{ mm}$ and $115.5\text{ mm}$ per day.
  * **Orange Warning (Very Heavy):** Precipitation rate between $115.6\text{ mm}$ and $204.4\text{ mm}$ per day.
  * **Red Warning (Extremely Heavy):** Precipitation rate exceeding $204.5\text{ mm}$ per day.

### 6. Conversational RAG Copilot and Agricultural Contingency Mapping
The conversational agricultural copilot (`src/climate_copilot.py`) functions as an interactive decision support system:
* **Knowledge Retrieval:** Loads local Vector Databases containing pre-indexed regional crop contingency guidelines (compiled from ICAR-CRIDA district databases).
* **Context Grounding:** The query pipeline intersects user input with the current localized meteorological status (mean rainfall grid anomalies, SPI drought indexes, and CWSI crop stress). 
* **Synthesis:** Feeds the grounded context vector into a Large Language Model prompt template, constraining responses to official district-level guidelines (e.g., prescribing specific drought-resistant crop choices like Pearl Millet or Sorghum during prolonged precipitation deficits).

---

## WMS Layer Integration Protocols

The GIS layer integrates real-time satellite data using the Web Map Service (WMS) protocol:
* **Endpoints:** Communicates with NASA GIBS (Global Imagery Browse Services) and JAXA servers.
* **Execution:** Resolves map layer views dynamically inside Plotly `Scattermap` maps. The client browser issues a standard `GetMap` query to fetch raster tiles projected under EPSG:3857 coordinate systems:
  `https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&BBOX={bbox-epsg-3857}&CRS=EPSG:3857&WIDTH=256&HEIGHT=256&LAYERS=LPRM_AMSR2_Surface_Soil_Moisture_C1_Band_Day_Daily&STYLES=&FORMAT=image/png&TRANSPARENT=true`
* The Soil Moisture layer is automatically requested without rigid static date parameters, allowing the WMS server to default to the latest available daily satellite composite.

---

## FastAPI REST API Gateway Specification

The microservice architecture exposes predictions and geodata endpoints:
* **GET `/api/status`**: Returns the current dataset assimilation timestamp and scale factors.
* **GET `/api/predictions/rainfall`**: Returns a float array containing the 7-day spatial rainfall predictions for the active region.
* **GET `/api/predictions/temperature`**: Returns the maximum and minimum temperature forecasts.

---

## Model Verification and Skill Assessment

To evaluate forecast reliability, the platform implements holdout validation using observations from 2022-2023:

### 1. Brier Score and Brier Skill Score (BSS)
Evaluates the accuracy of probabilistic heavy rainfall forecasts ($>35\text{ mm/day}$):
* **Brier Score (BS):** Measures the mean squared error of probabilistic predictions:

$$\text{BS} = \frac{1}{N} \sum_{n=1}^{N} (p_n - o_n)^2$$

  where $p_n$ is the forecasted probability and $o_n$ is the binary observation (1 if $>35\text{ mm}$, else 0).
* **Brier Skill Score (BSS):** Measures the relative improvement over climatological forecasts:

$$\text{BSS} = 1.0 - \frac{\text{BS}_{\text{forecast}}}{\text{BS}_{\text{climatology}}}$$

  where the climatology score $\text{BS}_{\text{climatology}}$ is calculated by setting $p_n$ to the long-term historical probability of exceedance. A BSS $> 0$ indicates the model out-performs climatological probability.

### 2. Reliability Calibration Curves
Plots forecasted probabilities against observed frequencies across 10 bins. A perfectly calibrated model aligns along the $y=x$ diagonal line, revealing prediction biases (e.g., over-forecasting or under-forecasting).

### 3. Spatial Resolution Mapping and Grid Alignment
To resolve mathematical errors when computing verification statistics between grids of different coordinate shapes:
* Ground-based rainfall data operates on a $0.25^\circ$ spacing (yielding $129 \times 135$ grids).
* Temperature observations are formatted on a $1.0^\circ$ spacing (yielding $31 \times 31$ grids).
* **Grid Slicing Alignments:** The validation engine dynamically resolves sub-slices by mapping matching latitude and longitude index boundaries:

$$\text{Index}_{\text{start}} = \text{argmin}(|\text{Coords}_{\text{all}} - \text{Coords}_{\text{target}}|)$$

This ensures target holdout validation grids and forecast arrays match dimensions exactly, preventing shape broadcasting crashes during metric computations.

## Setup and Operational Deployment

### 1. Installation
Create a clean environment and install dependencies:
```bash
python -m venv venv312
source venv312/bin/activate
pip install -r requirements.txt
```

### 2. Execution
Launch the Streamlit reanalysis console:
```bash
streamlit run app/streamlit_app.py
```

### 3. FastAPI REST Gateway
Run the FastAPI service to serve predictions and geodata to external GIS clients:
```bash
uvicorn src.api.main:app --reload --port 8000
```
* **Endpoint example:** GET `http://localhost:8000/api/status` returns daily data status.

### 4. Live-Reload and Model Cache-Busting
To ensure that backend predictive edits take effect immediately inside the active Streamlit runtime process:
* **Active Module Reloading:** The main application imports `importlib` and calls `importlib.reload(spatial_predictions)` at the start of execution, bypassing Python's default cached imports.
* **Hash-Based Cache Invalidation:** The model singleton loader (`src/model_loader.py`) is annotated with `@st.cache_resource(hash_funcs={...})`, mapping the model state to a cryptographic file-hash of the neural layer definitions to force cache invalidation only when structural updates occur.

---

## License
MIT License
