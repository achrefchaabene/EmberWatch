# EmberWatch M6 Model Card

## Purpose

This model forecasts the next-day wildfire mask for each 64x64 spatial patch.
It supports the EmberWatch M6 fire-spread forecasting module.

## Model

- Architecture: compact U-Net
- Trainable parameters: 1,931,297
- Input shape: 64x64x13
- Output shape: 64x64x1
- Output: per-pixel probability of fire on the next day
- Decision threshold: 0.57

## Input channels

1. Previous fire mask
2. Elevation
3. Wind-direction sine
4. Wind-direction cosine
5. Wind speed
6. Minimum temperature
7. Maximum temperature
8. Specific humidity
9. Precipitation
10. Palmer drought severity index
11. Energy release component
12. NDVI
13. Population

## Data preparation

- Raw TFRecord examples contain 64x64 layers.
- Target pixel `-1` is unknown and excluded from training loss and evaluation.
- Invalid values are imputed using the train-sample median.
- Wind direction is represented as sine and cosine to preserve circular behavior.
- Population and precipitation use `log1p` before standardization.
- All continuous channels use train-sample normalization statistics.

## Training run

- Training data used: 2,000 patches
- Validation data used: 400 patches from the `eval` split
- Epochs: 5
- Batch size: 4
- Optimizer: Adam, learning rate 0.001
- Loss: weighted binary cross-entropy, positive class weight 10

## Locked test results

The threshold was selected on the `eval` split before testing. The `test` split was evaluated with the fixed threshold of 0.57.

| Model | Average Precision | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Persistence baseline | 0.0864 | 0.2677 | - | - |
| U-Net v2 | 0.2738 | 0.3649 | 0.2890 | 0.4947 |

## Interpretation

The U-Net improves average precision and F1 over persistence. It detects about half of next-day fire pixels at the selected operating threshold. The model still generates false positives, so its outputs are forecast layers for operator review, not confirmed fire alerts.

## Calibration check

Calibration was measured on 1,579,548 valid pixels from the `eval` split.

- Brier score: 0.0228
- Expected calibration error: 0.0346

The reliability diagram is saved at `artifacts/reliability_diagram_eval.png`. The model probabilities are not yet calibrated for operational risk communication; use the 0.57 thresholded forecast for this prototype and calibrate probabilities before exposing them as confidence values.

## Channel-ablation check

Grouped permutation importance was measured on 400 `eval` patches. The largest AP drops were:

| Input group | AP drop when permuted |
|---|---:|
| Previous fire mask | 0.1322 |
| NDVI | 0.0133 |
| Elevation | 0.0067 |
| Temperature | 0.0032 |
| Precipitation | 0.0027 |

The model relies most strongly on the previous-day fire geometry. The remaining values are conditional importance estimates and may be affected by correlations between weather and environmental layers. The figure is saved at `artifacts/channel_ablation_eval.png`.

## Error analysis

The model was evaluated on positive-fire patches from 400 `eval` samples using the locked threshold of 0.57.

| Segment | AP | F1 |
|---|---:|---:|
| Low wind (<= 3.3) | 0.3195 | 0.3973 |
| Medium wind (3.3-4.4) | 0.2150 | 0.2856 |
| High wind (> 4.4) | 0.1525 | 0.2464 |
| Small fire (<= 15 pixels) | 0.2374 | 0.2742 |
| Medium fire (15-53 pixels) | 0.2496 | 0.3453 |
| Large fire (> 53 pixels) | 0.2143 | 0.2944 |

Performance falls as wind speed rises, which identifies high-wind spread as a priority for additional data and model refinement. The figure is saved at `artifacts/error_analysis_eval.png`.

## Known limitations

- This is a prototype trained on a 2,000-patch subset, not the complete training corpus.
- It predicts on the dataset grid only. It does not yet produce a geographically referenced GeoTIFF layer.
- It does not replace M5 camera-based geolocation.
- It needs calibration, channel-ablation analysis, and error analysis by wind regime and fire size before final delivery.

## Inference

Use the saved model at `artifacts/unet_m6_v2.keras` and the preprocessing code in `src/data/preprocessing.py`.

```cmd
.venv\Scripts\python.exe scripts\predict_sample.py --split test --min-fire-pixels 100
```
