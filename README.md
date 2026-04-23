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
| `xgboost`  | Tree ensemble | 1 | XGBoost trained on top-20 features |
| `lightgbm` | Tree ensemble | 1 | LightGBM, same features |
| `lstm`     | Bidirectional LSTM | 12 | 4-layer BiLSTM, 60-min context window |
| `gan`      | CNN discriminator | 12 | Adversarially trained (LSTM generator) |

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

### 3 — Export artifacts from the training notebook

Open `Main_improved_GAN.ipynb`, run all cells, then run the export script
**inside the same kernel** (or paste it as a new cell):

```bash
python scripts/export_artifacts.py
```

This writes 6 files to `app/artifacts/`:

```
app/artifacts/
├── scaler.joblib
├── feature_names.joblib
├── best_lstm_outage.keras
├── best_gan_discriminator.keras
├── xgboost_model.joblib
└── lightgbm_model.joblib
```

> **Large files** — add these to Git LFS or store them in an object store
> (S3, Azure Blob) and download at deploy time. The `.gitignore` excludes
> them from regular commits.

### 4 — Run the API

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
    {
      "bucket_5m": "2026-04-01T08:00:00",
      "site_id": 0,
      "voltageA": 232.1, "voltageB": 231.5, "voltageC": 230.8,
      "current1A": 12.3, "current1B": 11.9, "current1C": 12.7,
      "activePower1A": 2850.0, "activePower1B": 2760.0, "activePower1C": 2930.0,
      "frequency": 50.02
    }
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

For **LSTM / GAN**, provide **≥ 12 rows** (chronologically ordered).

### `GET /api/v1/models`

Returns which models are loaded and ready.

### `GET /api/v1/health`

Liveness + readiness probe (use for Docker / Kubernetes health checks).

---

## Row requirements by model

| Model | Rows needed | Why |
|-------|-------------|-----|
| `xgboost` / `lightgbm` | ≥ 1 (use last row) | Rolling features are pre-computed; tabular input |
| `lstm` / `gan` | ≥ 12 | Needs `sequence_len` timesteps for the recurrent / convolutional input |

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

## Adding a new model

1. Train and serialise the model (`joblib.dump` or `.save()`).
2. Add its path to `.env.example` and `app/config.py`.
3. Add a loader method to `app/services/model_loader.py`.
4. Add `"new_model"` to the `ModelName` Literal in `app/routes/predict.py`.
5. Add its inference branch in `predict()` (tabular or sequence).

---

## License

MIT
