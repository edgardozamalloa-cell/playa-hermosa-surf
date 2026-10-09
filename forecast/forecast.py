"""Build the Playa Hermosa surf forecast and write it to docs/data/forecast.json.

Same steps as the Colab notebook (playa_hermosa_forecast.ipynb):
  1. download NOAA GFS-Wave (WAVEWATCH III) for one offshore grid point
  2. turn each swell into a breaker height with the Komar-Gaylord formula
  3. label the wind as light / offshore / cross-shore / onshore
GitHub Actions runs this every 6 hours; the web page reads the JSON it writes.
"""
import datetime as dt
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

# ---------------------------------------------------------------- settings
SPOT = "Playa Hermosa, Puntarenas, Costa Rica"
LATITUDE = 9.25            # offshore model grid point, degrees north
LONGITUDE = 275.25         # degrees east (= 84.75 W)
FORECAST_HOURS = range(0, 169, 3)   # 7 days, every 3 hours
BEACH_FACING_DEG = 225     # direction the beach faces (225 = southwest), an estimate
LIGHT_WIND_MS = 3.0        # below this the wind barely matters
G = 9.81
M_TO_FT = 3.281
MS_TO_KN = 1.944
TIMEZONE = "America/Costa_Rica"

FIELDS = {
    ("HTSGW", "surface"):       "wave_height_m",
    ("PERPW", "surface"):       "peak_period_s",
    ("DIRPW", "surface"):       "peak_direction_deg",
    ("SWELL", "1 in sequence"): "swell1_height_m",
    ("SWPER", "1 in sequence"): "swell1_period_s",
    ("SWDIR", "1 in sequence"): "swell1_direction_deg",
    ("SWELL", "2 in sequence"): "swell2_height_m",
    ("SWPER", "2 in sequence"): "swell2_period_s",
    ("SWDIR", "2 in sequence"): "swell2_direction_deg",
    ("WIND",  "surface"):       "wind_speed_ms",
    ("WDIR",  "surface"):       "wind_direction_deg",
}

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "forecast.json"


# ---------------------------------------------------------------- download
def download(url, byte_range=None, tries=4):
    """Fetch a URL, retrying a few times: cloud storage occasionally drops a request."""
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    for attempt in range(tries):
        try:
            response = requests.get(url, headers=headers, timeout=60)
            if response.ok:
                return response
        except requests.RequestException:
            if attempt == tries - 1:
                raise
        time.sleep(2 ** attempt)
    response.raise_for_status()


def file_url(run_time, hour):
    day = run_time.strftime("%Y%m%d")
    cycle = run_time.strftime("%H")
    return f"{BUCKET}/gfs.{day}/{cycle}/wave/gridded/gfswave.t{cycle}z.global.0p25.f{hour:03d}.grib2"


def latest_complete_run():
    now = dt.datetime.now(dt.timezone.utc)
    newest = now.replace(hour=now.hour // 6 * 6, minute=0, second=0, microsecond=0)
    for runs_back in range(8):
        run_time = newest - dt.timedelta(hours=6 * runs_back)
        last_file = file_url(run_time, max(FORECAST_HOURS))
        if requests.head(last_file).ok and requests.head(last_file + ".idx").ok:
            return run_time
    raise RuntimeError("No complete run found in the last 2 days")


def byte_ranges(url):
    """Read the .idx file and return {(name, level): (start, end)} for the fields we want."""
    lines = download(url + ".idx").text.strip().splitlines()
    ranges = {}
    for i, line in enumerate(lines):
        parts = line.split(":")
        key = (parts[3], parts[4])
        if key in FIELDS:
            start = int(parts[1])
            end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else ""
            ranges[key] = (start, end)
    return ranges


def value_at_point(grib_bytes):
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(grib_bytes)
        path = f.name
    try:
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        variable = list(ds.data_vars)[0]
        point = ds[variable].sel(latitude=LATITUDE, longitude=LONGITUDE, method="nearest")
        return float(point.values)
    finally:
        os.remove(path)


def download_forecast(run_time):
    def download_hour(hour):
        url = file_url(run_time, hour)
        row = {"time_utc": run_time + dt.timedelta(hours=hour)}
        for key, byte_range in byte_ranges(url).items():
            row[FIELDS[key]] = value_at_point(download(url, byte_range).content)
        return row

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(download_hour, FORECAST_HOURS))
    forecast = pd.DataFrame(rows).set_index("time_utc")[list(FIELDS.values())]
    forecast.index = forecast.index.tz_convert(TIMEZONE)
    return forecast


# ---------------------------------------------------------------- physics
def angle_difference(a, b):
    """Smallest signed difference between two compass directions, degrees (-180 to 180)."""
    return (a - b + 180) % 360 - 180


def breaker_height(H0, T, direction_from):
    """Komar-Gaylord breaker height (m): 0.39 g^(1/5) (T H0^2 cos(alpha))^(2/5)."""
    alpha = np.radians(angle_difference(direction_from, BEACH_FACING_DEG))
    cos_alpha = np.clip(np.cos(alpha), 0, None)
    return 0.39 * G**0.2 * (T * H0**2 * cos_alpha) ** 0.4


def wind_quality(speed, direction_from):
    offshore_direction = (BEACH_FACING_DEG + 180) % 360
    off_angle = abs(angle_difference(direction_from, offshore_direction))
    if speed < LIGHT_WIND_MS:
        return "light"
    if off_angle < 45:
        return "offshore"
    if off_angle > 135:
        return "onshore"
    return "cross-shore"


def surf_estimate(forecast):
    surf = pd.DataFrame(index=forecast.index)
    s1 = breaker_height(forecast["swell1_height_m"], forecast["swell1_period_s"], forecast["swell1_direction_deg"])
    s2 = breaker_height(forecast["swell2_height_m"], forecast["swell2_period_s"], forecast["swell2_direction_deg"])
    surf["surf_ft"] = np.sqrt(s1**2 + s2**2) * M_TO_FT
    surf["sets_ft"] = 1.27 * surf["surf_ft"]
    surf["swell_m"] = forecast["swell1_height_m"]
    surf["period_s"] = forecast["swell1_period_s"]
    surf["swell_dir_deg"] = forecast["swell1_direction_deg"]
    surf["swell2_m"] = forecast["swell2_height_m"]
    surf["period2_s"] = forecast["swell2_period_s"]
    surf["swell2_dir_deg"] = forecast["swell2_direction_deg"]
    surf["wind_kn"] = forecast["wind_speed_ms"] * MS_TO_KN
    surf["wind_dir_deg"] = forecast["wind_direction_deg"]
    surf["wind"] = [wind_quality(s, d) for s, d in zip(forecast["wind_speed_ms"], forecast["wind_direction_deg"])]
    return surf


# ---------------------------------------------------------------- output
def to_json(surf, run_time):
    hours = [
        {
            "time": t.isoformat(),
            "surf_ft": round(r.surf_ft, 1),
            "sets_ft": round(r.sets_ft, 1),
            "swell_m": round(r.swell_m, 2),
            "period_s": round(r.period_s, 1),
            "swell_dir_deg": round(r.swell_dir_deg),
            "swells": [   # each swell train offshore: height (ft), period (s), direction it comes from (deg)
                {"ft": round(r.swell_m * M_TO_FT, 1), "s": round(r.period_s), "deg": round(r.swell_dir_deg)},
                {"ft": round(r.swell2_m * M_TO_FT, 1), "s": round(r.period2_s), "deg": round(r.swell2_dir_deg)},
            ],
            "wind_kn": round(r.wind_kn, 1),
            "wind_dir_deg": round(r.wind_dir_deg),
            "wind": r.wind,
        }
        for t, r in surf.iterrows()
    ]
    return {
        "spot": SPOT,
        "model_run_utc": run_time.isoformat(),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "beach_facing_deg": BEACH_FACING_DEG,
        "calibrated": False,
        "hours": hours,
    }


def main():
    run_time = latest_complete_run()
    print("Using model run", run_time.strftime("%Y-%m-%d %H:%M UTC"))
    forecast = download_forecast(run_time)
    surf = surf_estimate(forecast)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(to_json(surf, run_time), indent=1))
    print("Wrote", OUTPUT, "with", len(surf), "forecast times")


if __name__ == "__main__":
    main()
