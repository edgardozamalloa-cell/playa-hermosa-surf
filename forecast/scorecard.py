"""Score the site's past forecasts against surf reports from the beach.

For each report in the surf log, find what the site was showing before the session
(the last forecast published before it started) and the forecast published about a
day earlier, and compare them with what was seen. Every published forecast is
already saved in the repo's git history, so nothing extra has to be archived.

Usage: python forecast/scorecard.py surf_log.csv scorecard.csv
"""
import csv
import json
import subprocess
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Costa_Rica")
SLUGS = {"Playitas": "playitas", "La Punta": "la-punta", "Playa Hermosa": "playa-hermosa"}
FT_TO_M = 1 / 3.281


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def published_versions(slug):
    """Every version of the spot's forecast file, oldest first: (commit time, data)."""
    path = f"docs/data/{slug}.json"
    out = []
    for line in git("log", "--reverse", "--format=%H %cI", "--", path).splitlines():
        sha, when = line.split()
        try:
            out.append((datetime.fromisoformat(when), json.loads(git("show", f"{sha}:{path}"))))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
    return out


def surf_at(data, when):
    """Typical and set height (ft) at a time, interpolated between forecast hours."""
    hours = [(datetime.fromisoformat(h["time"]), h) for h in data["hours"]]
    for (t0, a), (t1, b) in zip(hours, hours[1:]):
        if t0 <= when <= t1:
            w = (when - t0) / (t1 - t0)
            return a["surf_ft"] + w * (b["surf_ft"] - a["surf_ft"]), a["sets_ft"] + w * (b["sets_ft"] - a["sets_ft"])
    return None


def session_times(date, time_local):
    start, _, end = time_local.partition("-")
    t0 = datetime.fromisoformat(f"{date}T{start}").replace(tzinfo=TZ)
    t1 = datetime.fromisoformat(f"{date}T{end}").replace(tzinfo=TZ) if end else t0
    return t0, t0 + (t1 - t0) / 2


def main(log_path, out_path):
    versions = {}
    rows = []
    for r in csv.DictReader(open(log_path)):
        slug = SLUGS.get(r["spot"])
        if not slug:
            continue
        versions.setdefault(slug, published_versions(slug))
        start, mid = session_times(r["date"], r["time_local"])
        obs_lo, obs_hi = float(r["surf_ft_low"]), float(r["surf_ft_high"])
        row = {"date": r["date"], "time": r["time_local"], "spot": r["spot"],
               "seen_m": f"{obs_lo * FT_TO_M:.1f}-{obs_hi * FT_TO_M:.1f}"}
        for label, cutoff in (("same_day", start), ("day_before", start - timedelta(hours=24))):
            before = [d for t, d in versions[slug] if t <= cutoff]
            pred = surf_at(before[-1], mid) if before else None
            if pred:
                row[f"{label}_m"] = f"{pred[0] * FT_TO_M:.1f}-{pred[1] * FT_TO_M:.1f}"
                row[f"{label}_error_m"] = f"{(pred[0] - obs_lo) * FT_TO_M:+.2f}"
            else:
                row[f"{label}_m"] = row[f"{label}_error_m"] = ""
        rows.append(row)
    fields = ["date", "time", "spot", "seen_m", "same_day_m", "same_day_error_m", "day_before_m", "day_before_error_m"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print(" | ".join(row[k] or "-" for k in fields))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
