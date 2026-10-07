# Project

## Data Sources and Licenses
- **India States Boundary**: The state boundary GeoJSON data (`data/india_states.geojson`) is sourced from the open-source repository [adarshbiradar/maps-geojson](https://github.com/adarshbiradar/maps-geojson).
## Setup Instructions

1. **Install dependencies**
   Install the necessary packages using `pip`:
   ```bash
   pip install -r requirements.txt
   pip install pytest httpx2
   ```

2. **Run Tests**
   To execute the spatial, API, and Streamlit integration test suites:
   ```bash
   python -m pytest -v tests/
   ```

3. **Run the Dashboard**
   Launch the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```
