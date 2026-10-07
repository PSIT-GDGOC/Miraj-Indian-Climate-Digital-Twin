import os
from fastapi import FastAPI
import xarray as xr

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "sample", "sample_rainfall_v3.nc")

@app.get("/api/status")
def get_status():
    try:
        ds = xr.open_dataset(DATA_PATH)
        latest_time = str(ds.time.values[-1])
        return {"status": "online", "latest_timestamp": latest_time}
    except FileNotFoundError:
        return {"status": "offline", "error": "Dataset not found"}
