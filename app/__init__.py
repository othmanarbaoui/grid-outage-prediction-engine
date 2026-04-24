"""
app/__init__.py
────────────────
FastAPI application factory.

Call `create_app()` to get a configured FastAPI instance.
The app factory pattern keeps the application testable and avoids
import-time side effects.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import settings
from app.routes.predict import router as predict_router
from app.services.model_loader import model_registry

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────
    logger.info("Starting %s v%s [%s]", settings.app_name, settings.app_version, settings.app_env)
    model_registry.load_all()
    logger.info("Startup complete.")
    yield
    # ── Shutdown ──────────────────────────────────────────────
    logger.info("Shutting down.")


# ── App factory ───────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
## Grid Outage Prediction Engine

Predict power-grid outages **30 minutes ahead** using four trained models:

| Model | Type | Latency |
|-------|------|---------|
| **XGBoost** | Gradient boosted trees | ~1 ms |
| **LightGBM** | Gradient boosted trees | ~1 ms |
| **LSTM** | Bidirectional recurrent network | ~10 ms |
| **GAN** | CNN discriminator (adversarially trained) | ~10 ms |

### Quick start
1. Choose a model with `GET /models`
2. Send telemetry rows to `POST /predict`
3. Receive a probability score + binary prediction

### Threshold
The default decision threshold is **0.30** — optimised for high recall
(missing an outage is more costly than a false alarm in telecom sites).
You can override it per request.
        """,
        contact={
            "name": "ARBAOUI OTHMANE",
            "email": "othmanearbaoui75@gmail.com",
        },
        license_info={
            "name": "MIT",
        },
        openapi_tags=[
            {"name": "Prediction", "description": "Run outage predictions"},
            {"name": "Models",     "description": "Inspect loaded models"},
            {"name": "Health",     "description": "Service health probes"},
        ],
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────
    app.include_router(predict_router, prefix="/api/v1")

    # Redirect root → Swagger UI
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

    return app
