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

try:
    import pyTMD.constituents
except ImportError:      # tides are optional; the forecast still works without them
    pyTMD = None

# ---------------------------------------------------------------- settings
# One entry per surf spot. "lat"/"lon" is the nearest model grid point in open water
# (degrees north / degrees east, so 84.75 W = 275.25 E). "facing" is the compass
# direction the beach looks out to sea; both facings are estimates from the map.
# "beach" is the spot itself (lat, lon), used for tides.
# "scale" corrects the physics estimate to match surf reports from the beach
# (1.0 = no correction yet); "reports" is how many reports it's based on.
SPOTS = {
    "playa-hermosa": {"name": "Playa Hermosa", "area": "Puntarenas, near Jacó",
                      "lat": 9.25, "lon": 275.25, "beach": (9.56, -84.60), "facing": 225, "scale": 1.0, "reports": 0},
    "playitas":      {"name": "Playitas", "area": "Manuel Antonio, Quepos",
                      "lat": 9.25, "lon": 275.75, "beach": (9.396, -84.161), "facing": 230, "scale": 0.78, "reports": 4},
    "la-punta":      {"name": "La Punta", "area": "Dominicalito, near Dominical",
                      "lat": 9.0, "lon": 276.0, "beach": (9.2158, -83.8365), "facing": 220, "scale": 1.0, "reports": 0},
}
FORECAST_HOURS = range(0, 169, 3)   # 7 days, every 3 hours
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
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "data"


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


def values_at_spots(grib_bytes):
    """Open one downloaded field and return {spot slug: value at that spot's grid point}."""
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(grib_bytes)
        path = f.name
    try:
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        field = ds[list(ds.data_vars)[0]]
        return {slug: float(field.sel(latitude=spot["lat"], longitude=spot["lon"], method="nearest"))
                for slug, spot in SPOTS.items()}
    finally:
        os.remove(path)


def download_hour(run_time, hour):
    """Download every field for one forecast hour; return {spot slug: row}."""
    url = file_url(run_time, hour)
    rows = {slug: {"time_utc": run_time + dt.timedelta(hours=hour)} for slug in SPOTS}
    for key, byte_range in byte_ranges(url).items():
        for slug, value in values_at_spots(download(url, byte_range).content).items():
            rows[slug][FIELDS[key]] = value
    return rows


def download_forecast(run_time):
    """Return {spot slug: DataFrame of the forecast at that spot}."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        hours = list(pool.map(lambda h: download_hour(run_time, h), FORECAST_HOURS))
    forecasts = {}
    for slug in SPOTS:
        df = pd.DataFrame([h[slug] for h in hours]).set_index("time_utc")[list(FIELDS.values())]
        df.index = df.index.tz_convert(TIMEZONE)
        forecasts[slug] = df
    return forecasts


# ---------------------------------------------------------------- physics
def angle_difference(a, b):
    """Smallest signed difference between two compass directions, degrees (-180 to 180)."""
    return (a - b + 180) % 360 - 180


def breaker_height(H0, T, direction_from, facing):
    """Komar-Gaylord breaker height (m): 0.39 g^(1/5) (T H0^2 cos(alpha))^(2/5)."""
    alpha = np.radians(angle_difference(direction_from, facing))
    cos_alpha = np.clip(np.cos(alpha), 0, None)
    return 0.39 * G**0.2 * (T * H0**2 * cos_alpha) ** 0.4


def wind_quality(speed, direction_from, facing):
    offshore_direction = (facing + 180) % 360
    off_angle = abs(angle_difference(direction_from, offshore_direction))
    if speed < LIGHT_WIND_MS:
        return "light"
    if off_angle < 45:
        return "offshore"
    if off_angle > 135:
        return "onshore"
    return "cross-shore"


def surf_estimate(forecast, facing, scale=1.0):
    surf = pd.DataFrame(index=forecast.index)
    s1 = breaker_height(forecast["swell1_height_m"], forecast["swell1_period_s"], forecast["swell1_direction_deg"], facing)
    s2 = breaker_height(forecast["swell2_height_m"], forecast["swell2_period_s"], forecast["swell2_direction_deg"], facing)
    surf["surf_ft"] = np.sqrt(s1**2 + s2**2) * M_TO_FT * scale
    surf["sets_ft"] = 1.27 * surf["surf_ft"]
    surf["swell_m"] = forecast["swell1_height_m"]
    surf["period_s"] = forecast["swell1_period_s"]
    surf["swell_dir_deg"] = forecast["swell1_direction_deg"]
    surf["swell2_m"] = forecast["swell2_height_m"]
    surf["period2_s"] = forecast["swell2_period_s"]
    surf["swell2_dir_deg"] = forecast["swell2_direction_deg"]
    surf["wind_kn"] = forecast["wind_speed_ms"] * MS_TO_KN
    surf["wind_dir_deg"] = forecast["wind_direction_deg"]
    surf["wind"] = [wind_quality(s, d, facing) for s, d in zip(forecast["wind_speed_ms"], forecast["wind_direction_deg"])]
    return surf


# ---------------------------------------------------------------- tides
TIDE_FILE = Path(__file__).with_name("tide_constants.json")


def tide_curve(slug, start, hours=168, step_min=30):
    """Predicted tide (m above mean sea level) at a spot, from its harmonic constants.

    height(t) = sum over constituents of  f * A * cos(V(t) + u - g)
    A, g: amplitude and phase lag at the spot (from the GOT4.10c model)
    V(t): the constituent's astronomical argument (where the Moon and Sun are)
    f, u: slow 18.6-year corrections for the tilt of the Moon's orbit
    """
    if pyTMD is None or not TIDE_FILE.exists():
        return []
    consts = json.loads(TIDE_FILE.read_text()).get(slug)
    if not consts:
        return []
    names = list(consts)
    amp = np.array([consts[c][0] for c in names])
    lag = np.radians([consts[c][1] for c in names])
    start = pd.Timestamp(start).floor("h")
    times = pd.date_range(start, periods=hours * 60 // step_min + 1, freq=f"{step_min}min")
    mjd = (times - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds().values / 86400 + 40587.0  # Modified Julian Day
    u, f, V = pyTMD.constituents.arguments(mjd, names, corrections="GOT")
    height = (f * amp * np.cos(np.radians(V) + u - lag)).sum(axis=1)
    return [{"time": t.tz_convert(TIMEZONE).isoformat(), "m": round(float(h), 2)} for t, h in zip(times, height)]


# ---------------------------------------------------------------- output
def to_json(slug, surf, run_time):
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
        "slug": slug,
        "spot": SPOTS[slug]["name"],
        "area": SPOTS[slug]["area"],
        "model_run_utc": run_time.isoformat(),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "beach_facing_deg": SPOTS[slug]["facing"],
        "scale": SPOTS[slug]["scale"],
        "reports_used": SPOTS[slug]["reports"],
        "hours": hours,
        "tide": tide_curve(slug, surf.index[0].tz_convert("UTC")),
    }


def main():
    run_time = latest_complete_run()
    print("Using model run", run_time.strftime("%Y-%m-%d %H:%M UTC"))
    forecasts = download_forecast(run_time)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for slug, forecast in forecasts.items():
        surf = surf_estimate(forecast, SPOTS[slug]["facing"], SPOTS[slug]["scale"])
        (OUTPUT_DIR / f"{slug}.json").write_text(json.dumps(to_json(slug, surf, run_time), indent=1))
        print("Wrote", slug, "with", len(surf), "forecast times")
    spot_list = [{"slug": slug, "name": s["name"], "area": s["area"]} for slug, s in SPOTS.items()]
    (OUTPUT_DIR / "spots.json").write_text(json.dumps(spot_list, indent=1))


if __name__ == "__main__":
    main()
