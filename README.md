# Grid Outage Prediction Engine 🔌

REST API for predicting power-grid outages **30 minutes ahead** at telecom sites.  
Built with **FastAPI** · serves four trained models via a unified endpoint · fully documented with **Swagger UI**.

---

## Architecture

```
grid-outage-prediction-engine/
├── app/
│   ├── __init__.py            # FastAPI app factory + lifespan
│   ├── config.py              # Pydantic-settings (reads .env)
│   ├── routes/
│   │   └── predict.py         # POST /predict · GET /models · GET /health
│   ├── services/
│   │   └── model_loader.py    # Singleton model registry (lazy load)
│   ├── utils/
│   │   └── helpers.py         # Feature engineering + preprocessing
│   └── artifacts/             # ← place trained model files here
├── scripts/
│   └── export_artifacts.py    # Run in notebook env to export .joblib / .keras
├── main.py                    # Uvicorn entry point
├── requirements.txt
├── .env.example
└── README.md
```

### Models

| Key | Type | Min rows | Description |
|-----|------|----------|-------------|
| `xgboost`  | Tree ensemble | 12 | XGBoost trained on top-20 features |
| `lightgbm` | Tree ensemble | 12 | LightGBM, same features |
| `lstm`     | Bidirectional LSTM | 23 | 4-layer BiLSTM, 60-min context window |
| `gan`      | CNN discriminator | 23 | Adversarially trained (LSTM generator) |

---

## Quick Start

### 1 — Clone & install

```bash
git clone https://github.com/your-org/grid-outage-prediction-engine.git
cd grid-outage-prediction-engine

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2 — Configure environment

```bash
cp .env.example .env
# Edit .env if needed (paths, threshold, port …)
```
### 3 — Run the API

```bash
python main.py
```

Or with uvicorn directly (production):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

Open **http://localhost:8000** → redirects to Swagger UI.

---

## API Reference

### `POST /api/v1/predict`

```json
{
  "model": "xgboost",
  "threshold": 0.30,
  "rows": [
    {"bucket_5m": "2026-04-01T07:00:00", "site_id": 0, "voltageA": 232.12, "voltageB": 231.47, "voltageC": 230.96, "current1A": 12.48, "current1B": 11.87, "current1C": 12.67, "activePower1A": 2896.9, "activePower1B": 2747.5, "activePower1C": 2926.3, "frequency": 50.039},
    {"bucket_5m": "2026-04-01T07:05:00", "site_id": 0, "voltageA": 232.11, "voltageB": 231.30, "voltageC": 230.86, "current1A": 12.24, "current1B": 11.84, "current1C": 12.73, "activePower1A": 2841.0, "activePower1B": 2738.6, "activePower1C": 2938.8, "frequency": 49.952},
    {"bucket_5m": "2026-04-01T07:10:00", "site_id": 0, "voltageA": 231.41, "voltageB": 231.20, "voltageC": 230.39, "current1A": 12.34, "current1B": 11.79, "current1C": 12.53, "activePower1A": 2855.6, "activePower1B": 2725.8, "activePower1C": 2886.8, "frequency": 50.037},
    {"bucket_5m": "2026-04-01T07:15:00", "site_id": 0, "voltageA": 231.70, "voltageB": 231.28, "voltageC": 230.20, "current1A": 12.23, "current1B": 11.91, "current1C": 12.56, "activePower1A": 2833.7, "activePower1B": 2754.5, "activePower1C": 2891.3, "frequency": 50.009},
    {"bucket_5m": "2026-04-01T07:20:00", "site_id": 0, "voltageA": 231.53, "voltageB": 231.11, "voltageC": 230.33, "current1A": 12.52, "current1B": 11.90, "current1C": 12.57, "activePower1A": 2898.8, "activePower1B": 2750.2, "activePower1C": 2895.2, "frequency": 50.021},
    {"bucket_5m": "2026-04-01T07:25:00", "site_id": 0, "voltageA": 231.29, "voltageB": 231.15, "voltageC": 229.91, "current1A": 12.14, "current1B": 11.92, "current1C": 12.79, "activePower1A": 2807.9, "activePower1B": 2755.3, "activePower1C": 2940.5, "frequency": 50.004},
    {"bucket_5m": "2026-04-01T07:30:00", "site_id": 0, "voltageA": 231.49, "voltageB": 230.94, "voltageC": 229.95, "current1A": 12.21, "current1B": 11.84, "current1C": 12.83, "activePower1A": 2826.5, "activePower1B": 2734.3, "activePower1C": 2950.3, "frequency": 50.009},
    {"bucket_5m": "2026-04-01T07:35:00", "site_id": 0, "voltageA": 231.00, "voltageB": 231.02, "voltageC": 230.14, "current1A": 12.22, "current1B": 11.97, "current1C": 12.82, "activePower1A": 2822.8, "activePower1B": 2765.3, "activePower1C": 2950.4, "frequency": 50.023},
    {"bucket_5m": "2026-04-01T07:40:00", "site_id": 0, "voltageA": 231.15, "voltageB": 230.78, "voltageC": 230.24, "current1A": 12.42, "current1B": 11.84, "current1C": 12.68, "activePower1A": 2870.9, "activePower1B": 2732.4, "activePower1C": 2919.4, "frequency": 49.972},
    {"bucket_5m": "2026-04-01T07:45:00", "site_id": 0, "voltageA": 230.98, "voltageB": 230.98, "voltageC": 230.42, "current1A": 12.29, "current1B": 12.02, "current1C": 12.74, "activePower1A": 2838.7, "activePower1B": 2776.4, "activePower1C": 2935.6, "frequency": 49.984},
    {"bucket_5m": "2026-04-01T07:50:00", "site_id": 0, "voltageA": 231.29, "voltageB": 231.08, "voltageC": 229.99, "current1A": 12.49, "current1B": 11.59, "current1C": 12.80, "activePower1A": 2888.8, "activePower1B": 2678.2, "activePower1C": 2943.9, "frequency": 50.002},
    {"bucket_5m": "2026-04-01T07:55:00", "site_id": 0, "voltageA": 231.05, "voltageB": 230.64, "voltageC": 229.42, "current1A": 12.27, "current1B": 11.94, "current1C": 12.88, "activePower1A": 2835.0, "activePower1B": 2753.8, "activePower1C": 2954.9, "frequency": 49.987}
  ]
}
```

**Response:**

```json
{
  "model": "xgboost",
  "probability": 0.073,
  "prediction": 0,
  "threshold": 0.30,
  "latency_ms": 1.2,
  "message": "✅ No outage predicted in the next 30 minutes."
}
```

For **LSTM / GAN**, provide **≥ 23 rows** (chronologically ordered).

### `GET /api/v1/models`

Returns which models are loaded and ready.

### `GET /api/v1/health`

Liveness + readiness probe (use for Docker / Kubernetes health checks).

---

## Row requirements by model

| Model | Rows needed | Why |
|-------|-------------|-----|
| `xgboost` / `lightgbm` | ≥ 12 (use last row) | Rolling features are pre-computed; tabular input |
| `lstm` / `gan` | ≥ 23 | Needs `sequence_len` timesteps for the recurrent / convolutional input |

> **Tip:** Always send more rows than the minimum (e.g. 20–30) so that
> rolling windows and lag features computed inside `engineer_features()` 
> are well-defined for the final row.

---

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

```bash
docker build -t grid-outage-api .
docker run -p 8000:8000 --env-file .env grid-outage-api
```

### Environment variables (production)

```bash
APP_ENV=production
RELOAD=false
DEFAULT_THRESHOLD=0.30
```

---

## Development

```bash
# Run with hot reload
python main.py

# Swagger UI
open http://localhost:8000/docs

# ReDoc
open http://localhost:8000/redoc

# OpenAPI JSON
curl http://localhost:8000/openapi.json
```

---
