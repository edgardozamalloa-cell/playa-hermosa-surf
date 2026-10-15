"""Pull tidal constants for each spot out of NASA's GOT4.10c global tide model.

Run once by the "Fetch tide constants" workflow (the model files are big, so
they are downloaded there, read, and thrown away). Writes forecast/tide_constants.json:
for each spot, each constituent's amplitude (m) and Greenwich phase lag (degrees).
"""
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from forecast import SPOTS  # noqa: E402

model_dir = Path(sys.argv[1])
out = {slug: {} for slug in SPOTS}
for f in sorted(model_dir.glob("*.nc")):
    ds = xr.open_dataset(f, decode_coords=False)
    name = ds.attrs["Constituent"].strip().lower()
    lon = ds["longitude"].values
    lat = ds["latitude"].values
    amp = ds["amplitude"].values
    pha = ds["phase"].values
    scale = 0.01 if "cm" in ds["amplitude"].attrs.get("units", "cm") else 1.0
    for slug, spot in SPOTS.items():
        la, lo = spot["beach"]
        lo360 = lo % 360
        # nearest ocean cell (the model has no values on land)
        LON, LAT = np.meshgrid(lon, lat)
        dist = (LAT - la) ** 2 + ((LON - lo360) * np.cos(np.radians(la))) ** 2
        dist[~np.isfinite(amp) | (amp <= 0)] = np.inf
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        out[slug][name] = [round(float(amp[i, j]) * scale, 4), round(float(pha[i, j]), 2),
                           round(float(LAT[i, j]), 3), round(float(LON[i, j]), 3)]
    print(name, {k: v[name][:2] for k, v in out.items()})

Path(__file__).with_name("tide_constants.json").write_text(json.dumps(out, indent=1))
