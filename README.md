## EmberWatch M6 - Next-Day Wildfire Spread Forecasting

This project predicts a 64x64 next-day fire mask from weather, terrain,
vegetation, population, and previous-fire inputs.

### Test results

| Model | AP | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Persistence baseline | 0.0864 | 0.2677 | - | - |
| U-Net v2 | 0.2738 | 0.3649 | 0.2890 | 0.4947 |

The selected decision threshold is `0.57`.

### Export a demo prediction

From the project root:

```cmd
.venv\Scripts\python.exe scripts\predict_sample.py --split test --index 0
```

The script writes a four-panel image to `artifacts/prediction_sample.png`:
previous fire, true next-day fire, U-Net probability, and thresholded forecast.

To automatically choose a clear demo example with at least 100 true fire pixels:

```cmd
.venv\Scripts\python.exe scripts\predict_sample.py --split test --min-fire-pixels 100
```

### Create a report figure

```cmd
.venv\Scripts\python.exe scripts\create_metrics_figure.py
```

For the complete M6 technical summary, see [the model card](docs/M6_MODEL_CARD.md).

### Evaluate probability calibration

```cmd
.venv\Scripts\python.exe scripts\evaluate_calibration.py
```

### Measure channel importance

```cmd
.venv\Scripts\python.exe scripts\evaluate_ablation.py
```

### Analyze errors by wind and fire size

```cmd
.venv\Scripts\python.exe scripts\error_analysis.py
```

### Export a forecast for the dashboard

```cmd
.venv\Scripts\python.exe scripts\export_forecast_json.py --split test --min-fire-pixels 100
```

This writes `artifacts/forecast_sample.json`. The current source dataset has no map coordinates, so the grid is local patch pixels until M5 provides a georeferenced forecast layer.

### Serve the forecast to the dashboard

Install the two API packages once:

```cmd
.venv\Scripts\pip.exe install fastapi uvicorn
```

Then start the local API:

```cmd
.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

The dashboard can call `http://127.0.0.1:8000/api/v1/forecast/latest` to receive the forecast JSON. Check `http://127.0.0.1:8000/health` to verify that the service is running.

Open `http://127.0.0.1:8000/` for the built-in visual forecast dashboard.

### Tunisia real-data pipeline

The Kaggle records have no geographic coordinates. The real Tunisia training
pipeline and the first data-acquisition command are in [the Tunisia pipeline guide](docs/TUNISIA_PIPELINE.md).
