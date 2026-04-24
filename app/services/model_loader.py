"""
app/services/model_loader.py
─────────────────────────────
Singleton registry that loads each model artifact once on first use.
Supports: LSTM, GAN (CNN discriminator), XGBoost, LightGBM.

Usage
-----
    from app.services.model_loader import model_registry
    model_registry.load_all()          # call once at startup
    lstm = model_registry.get("lstm")  # retrieve any time
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


# ── Model keys ────────────────────────────────────────────────
MODEL_LSTM     = "lstm"
MODEL_GAN      = "gan"
MODEL_XGBOOST  = "xgboost"
MODEL_LIGHTGBM = "lightgbm"

SUPPORTED_MODELS = {MODEL_LSTM, MODEL_GAN, MODEL_XGBOOST, MODEL_LIGHTGBM}


class ModelRegistry:
    """Thread-safe lazy registry for all prediction models + preprocessing."""

    def __init__(self) -> None:
        self._models:       Dict[str, Any] = {}
        self._scaler:       Any = None
        self._feature_names: list[str] = []
        self._loaded:       bool = False

    # ── Public API ────────────────────────────────────────────
    def load_all(self) -> None:
        """Load every artifact that exists on disk. Missing artifacts are
        logged as warnings (the API will return 503 for that model)."""
        if self._loaded:
            return

        self._load_preprocessing()
        self._load_sklearn_model(MODEL_XGBOOST,  settings.xgboost_file)
        self._load_sklearn_model(MODEL_LIGHTGBM, settings.lightgbm_file)
        self._load_keras_model(MODEL_LSTM,    settings.lstm_file)
        self._load_keras_discriminator(MODEL_GAN, settings.gan_discriminator_file)

        self._loaded = True
        logger.info(
            "ModelRegistry ready. Loaded models: %s",
            list(self._models.keys()),
        )

    def get(self, model_name: str) -> Any:
        if model_name not in self._models:
            raise KeyError(
                f"Model '{model_name}' is not loaded. "
                f"Available: {list(self._models.keys())}"
            )
        return self._models[model_name]

    @property
    def scaler(self) -> Any:
        if self._scaler is None:
            raise RuntimeError("Scaler is not loaded.")
        return self._scaler

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    def available_models(self) -> list[str]:
        return list(self._models.keys())

    @property
    def is_ready(self) -> bool:
        return self._loaded

    # ── Private loaders ───────────────────────────────────────
    def _load_preprocessing(self) -> None:
        for path, attr, label in [
            (settings.scaler_file,        "_scaler",        "Scaler"),
            (settings.feature_names_file, "_feature_names", "Feature names"),
        ]:
            if Path(path).exists():
                setattr(self, attr, joblib.load(path))
                logger.info("%s loaded from %s", label, path)
            else:
                logger.warning("%s artifact not found at %s", label, path)

    def _load_sklearn_model(self, name: str, path: Path) -> None:
        if Path(path).exists():
            self._models[name] = joblib.load(path)
            logger.info("Model '%s' loaded from %s", name, path)
        else:
            logger.warning(
                "Model '%s' artifact not found at %s — it will be unavailable.", name, path
            )

    # ── Keras helpers ─────────────────────────────────────────
    @staticmethod
    def _resolve_keras_path(path: Path) -> Optional[Path]:
        """
        Try the exact path first, then common alternative extensions.
        Covers cases where the notebook saved as .h5 but config expects .keras
        or vice-versa.
        """
        candidates = [path, path.with_suffix(".keras"), path.with_suffix(".h5")]
        for p in candidates:
            if p.exists():
                return p
        return None

    @staticmethod
    def _keras_custom_objects() -> dict:
        """
        Return a dict of every custom / non-standard layer used in the
        notebook so that load_model() can reconstruct them without error.
        This covers both the LSTM and the GAN discriminator.
        """
        import tensorflow as tf
        from tensorflow.keras.layers import LeakyReLU

        return {
            "LeakyReLU": LeakyReLU,
            # Add more here if you use custom loss functions or metrics, e.g.:
            # "focal_loss": focal_loss,
        }

    def _load_keras_model(self, name: str, path: Path) -> None:
        """Load a standard single-output Keras model (LSTM)."""
        resolved = self._resolve_keras_path(path)
        if resolved is None:
            logger.warning(
                "Model '%s' artifact not found at %s (also tried .keras / .h5) "
                "— it will be unavailable.",
                name, path,
            )
            return
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(
                str(resolved),
                custom_objects=self._keras_custom_objects(),
                compile=False,   # skip recompiling — we only need predict()
            )
            self._models[name] = model
            logger.info("Keras model '%s' loaded from %s", name, resolved)
        except Exception as exc:
            logger.error(
                "Failed to load Keras model '%s' from %s.\n"
                "  Error : %s\n"
                "  Tip   : make sure the file was saved with model.save() "
                "and that all custom layers are listed in _keras_custom_objects().",
                name, resolved, exc,
            )

    def _load_keras_discriminator(self, name: str, path: Path) -> None:
        """
        Load the GAN discriminator (two outputs: adversarial + classification).
        Wraps the model so callers only receive the classification head (index 1).
        """
        resolved = self._resolve_keras_path(path)
        if resolved is None:
            logger.warning(
                "Model '%s' artifact not found at %s (also tried .keras / .h5) "
                "— it will be unavailable.",
                name, path,
            )
            return
        try:
            import tensorflow as tf

            raw = tf.keras.models.load_model(
                str(resolved),
                custom_objects=self._keras_custom_objects(),
                compile=False,
            )

            class _DiscriminatorWrapper:
                """Exposes only the classification output (index 1)."""
                def __init__(self, model):
                    self._model = model

                def predict(self, X, **kwargs):
                    outputs = self._model.predict(X, **kwargs)
                    # outputs is a list [adv_out, cls_out] or a tuple
                    cls_out = outputs[1] if isinstance(outputs, (list, tuple)) else outputs
                    return cls_out  # shape (N, 1)

            self._models[name] = _DiscriminatorWrapper(raw)
            logger.info("GAN discriminator '%s' loaded from %s", name, resolved)
        except Exception as exc:
            logger.error(
                "Failed to load GAN discriminator '%s' from %s.\n"
                "  Error : %s\n"
                "  Tip   : make sure the file was saved with discriminator.save() "
                "and that all custom layers are listed in _keras_custom_objects().",
                name, resolved, exc,
            )


# ── Singleton ─────────────────────────────────────────────────
model_registry = ModelRegistry()