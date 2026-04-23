"""
app/config.py
─────────────
Centralised application settings loaded from environment variables / .env file.
All paths are resolved relative to the project root so the app works regardless
of the working directory from which it is launched.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Project root = parent of the directory that contains this file (app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App metadata ─────────────────────────────────────────────
    app_name: str = "Grid Outage Prediction Engine"
    app_version: str = "1.0.0"
    app_env: str = "development"

    # ── Server ───────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

    # ── Prediction defaults ───────────────────────────────────────
    default_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    sequence_len: int = Field(default=12, ge=1)
    top_n_features: int = Field(default=20, ge=1)

    # ── Artifact paths (kept as strings; resolved below) ──────────
    artifacts_dir: str = "app/artifacts"
    scaler_path: str = "app/artifacts/scaler.joblib"
    feature_names_path: str = "app/artifacts/feature_names.joblib"
    lstm_model_path: str = "app/artifacts/best_lstm_outage.keras"
    gan_discriminator_path: str = "app/artifacts/best_gan_discriminator.keras"
    xgboost_model_path: str = "app/artifacts/xgboost_model.joblib"
    lightgbm_model_path: str = "app/artifacts/lightgbm_model.joblib"

    # ── Resolved paths (computed properties) ─────────────────────
    @property
    def artifacts_dir_path(self) -> Path:
        return PROJECT_ROOT / self.artifacts_dir

    @property
    def scaler_file(self) -> Path:
        return PROJECT_ROOT / self.scaler_path

    @property
    def feature_names_file(self) -> Path:
        return PROJECT_ROOT / self.feature_names_path

    @property
    def lstm_file(self) -> Path:
        return PROJECT_ROOT / self.lstm_model_path

    @property
    def gan_discriminator_file(self) -> Path:
        return PROJECT_ROOT / self.gan_discriminator_path

    @property
    def xgboost_file(self) -> Path:
        return PROJECT_ROOT / self.xgboost_model_path

    @property
    def lightgbm_file(self) -> Path:
        return PROJECT_ROOT / self.lightgbm_model_path


# Singleton — import this everywhere
settings = Settings()
