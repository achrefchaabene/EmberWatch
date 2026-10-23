"""Minimal API that exposes the latest exported M6 forecast to the dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECAST_PATH = PROJECT_ROOT / "artifacts" / "forecast_sample.json"

app = FastAPI(title="EmberWatch Forecast API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this to the dashboard domain before deployment.
    allow_methods=["GET"],
    allow_headers=["*"],
)


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EmberWatch | Wildfire Forecast</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; min-height: 100vh; background: #0b1110; color: #e9f2ed; }
    main { max-width: 1050px; padding: 36px 20px 48px; margin: auto; }
    h1 { margin: 0; color: #f6c453; font-size: clamp(1.8rem, 4vw, 2.7rem); }
    .subtitle { color: #9eb3a7; margin: 7px 0 28px; }
    .grid { display: grid; grid-template-columns: minmax(280px, 560px) 1fr; gap: 22px; align-items: start; }
    .card { background: #121d1a; border: 1px solid #273a32; border-radius: 14px; padding: 18px; }
    canvas { width: 100%; image-rendering: pixelated; border-radius: 8px; background: #050807; }
    .legend { display: flex; align-items: center; gap: 10px; color: #aebcb4; font-size: .85rem; margin-top: 10px; }
    .gradient { height: 10px; flex: 1; border-radius: 99px; background: linear-gradient(90deg,#180b08,#a51d14,#ff821c,#ffe66b); }
    .stat { border-bottom: 1px solid #263a31; padding: 12px 0; }
    .stat:last-child { border: 0; }
    .label { color: #9eb3a7; font-size: .83rem; text-transform: uppercase; letter-spacing: .08em; }
    .value { font-size: 1.45rem; font-weight: 700; margin-top: 3px; }
    .warning { color: #ffd875; font-size: .9rem; line-height: 1.45; margin-top: 18px; }
    #status { color: #9eb3a7; }
    @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <h1>EmberWatch</h1>
    <p class="subtitle">M6 — Next-day wildfire spread forecast</p>
    <p id="status">Loading latest forecast…</p>
    <section class="grid" id="dashboard" hidden>
      <div class="card">
        <canvas id="heatmap" width="64" height="64" aria-label="Wildfire probability heatmap"></canvas>
        <div class="legend"><span>Low risk</span><div class="gradient"></div><span>High risk</span></div>
      </div>
      <aside class="card">
        <div class="stat"><div class="label">Model</div><div class="value" id="model"></div></div>
        <div class="stat"><div class="label">Forecast threshold</div><div class="value" id="threshold"></div></div>
        <div class="stat"><div class="label">Predicted fire pixels</div><div class="value" id="firePixels"></div></div>
        <div class="stat"><div class="label">Maximum probability</div><div class="value" id="maxProbability"></div></div>
        <p class="warning">This is a local 64×64 patch, not yet a geographic map. M5 coordinates are needed before field use.</p>
      </aside>
    </section>
  </main>
  <script>
    function color(value) {
      const stops = [[24,11,8],[165,29,20],[255,130,28],[255,230,107]];
      const x = Math.max(0, Math.min(1, value)) * (stops.length - 1);
      const i = Math.min(stops.length - 2, Math.floor(x)); const t = x - i;
      return stops[i].map((v, n) => Math.round(v + (stops[i + 1][n] - v) * t));
    }
    function draw(grid) {
      const canvas = document.getElementById('heatmap'); const ctx = canvas.getContext('2d');
      const image = ctx.createImageData(64, 64);
      grid.flat().forEach((value, i) => { const [r,g,b] = color(value); const p = i * 4; image.data[p]=r; image.data[p+1]=g; image.data[p+2]=b; image.data[p+3]=255; });
      ctx.putImageData(image, 0, 0);
    }
    fetch('/api/v1/forecast/latest').then(r => r.ok ? r.json() : Promise.reject(r.status)).then(data => {
      draw(data.probability_grid);
      document.getElementById('model').textContent = data.model;
      document.getElementById('threshold').textContent = data.threshold.toFixed(2);
      document.getElementById('firePixels').textContent = data.summary.predicted_fire_pixels.toLocaleString();
      document.getElementById('maxProbability').textContent = (data.summary.max_probability * 100).toFixed(1) + '%';
      document.getElementById('status').hidden = true; document.getElementById('dashboard').hidden = false;
    }).catch(() => { document.getElementById('status').textContent = 'Forecast unavailable. Run the export script first.'; });
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> str:
    """Serve a compact visual dashboard for the latest forecast."""
    return DASHBOARD_HTML


@app.get("/health")
def health() -> dict[str, str]:
    """Simple service check for the dashboard or deployment platform."""
    return {"status": "ok"}


@app.get("/api/v1/forecast/latest")
def latest_forecast() -> dict:
    """Return the newest M6 JSON forecast generated in artifacts/."""
    if not FORECAST_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No forecast found. Run scripts/export_forecast_json.py first.",
        )

    with FORECAST_PATH.open(encoding="utf-8") as forecast_file:
        return json.load(forecast_file)
