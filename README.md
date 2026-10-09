# Playa Hermosa Surf

A free, self-updating 7-day surf forecast for Playa Hermosa, Puntarenas, Costa Rica.

- `forecast/forecast.py` downloads NOAA's GFS-Wave (WAVEWATCH III) forecast for an offshore point, turns each swell into a breaking height with the Komar–Gaylord formula, labels the wind, and writes `docs/data/forecast.json`.
- `docs/index.html` is the web page. It reads `forecast.json` and draws the chart, day cards and hourly tables.
- `.github/workflows/update-forecast.yml` runs the script every 6 hours on GitHub's free servers and commits the new data.

## Settings

The numbers you're most likely to change are at the top of `forecast/forecast.py`: the grid point, the forecast length, and `BEACH_FACING_DEG`. Chart and table settings are at the top of the `<script>` in `docs/index.html`.

## Hosting (free)

1. Create a public repository on GitHub and upload these files, keeping the folders.
2. Settings > Pages: Source "Deploy from a branch", branch `main`, folder `/docs`.
3. Actions tab: enable workflows, open "Update forecast" and click **Run workflow** once.

The site appears at `https://<your-username>.github.io/<repository-name>/`. Public repositories get unlimited free Actions minutes.
