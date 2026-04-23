"""
scripts/export_artifacts.py
─────────────────────────────
Run this script ONCE inside your training environment (where the notebook
was executed) to export all model artifacts to the app/artifacts/ directory.

Usage
-----
    python scripts/export_artifacts.py

It expects the following variables to already exist in memory — paste it as a
notebook cell or run it as a script after training is complete.
"""

import os
import shutil
import joblib
import numpy as np
from pathlib import Path

ARTIFACTS_DIR = Path("app/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Exporting artifacts to: {ARTIFACTS_DIR.resolve()}")
print("=" * 55)

# ── 1. StandardScaler ─────────────────────────────────────────
# Uses `scaler_lstm` from notebook Section 8.1
try:
    joblib.dump(scaler_lstm, ARTIFACTS_DIR / "scaler.joblib")
    print("✓ scaler.joblib")
except NameError:
    print("✗ scaler_lstm not found — run Section 8.1 first")

# ── 2. Feature names ──────────────────────────────────────────
# Uses `TOP_FEATURES` from notebook Section 5
try:
    joblib.dump(TOP_FEATURES, ARTIFACTS_DIR / "feature_names.joblib")
    print(f"✓ feature_names.joblib  ({len(TOP_FEATURES)} features)")
except NameError:
    print("✗ TOP_FEATURES not found — run Section 5 first")

# ── 3. LSTM model ─────────────────────────────────────────────
lstm_src = Path("best_lstm_outage.keras")
lstm_dst  = ARTIFACTS_DIR / "best_lstm_outage.keras"
if lstm_src.exists():
    shutil.copy2(lstm_src, lstm_dst)
    print("✓ best_lstm_outage.keras")
else:
    print("✗ best_lstm_outage.keras not found — run Section 8.4 first")

# ── 4. GAN discriminator ──────────────────────────────────────
gan_src = Path("best_gan_discriminator.keras")
gan_dst  = ARTIFACTS_DIR / "best_gan_discriminator.keras"
if gan_src.exists():
    shutil.copy2(gan_src, gan_dst)
    print("✓ best_gan_discriminator.keras")
else:
    print("✗ best_gan_discriminator.keras not found — run Section 10.4 first")

# ── 5. XGBoost ────────────────────────────────────────────────
try:
    xgb_model = MODELS["XGBoost"]
    joblib.dump(xgb_model, ARTIFACTS_DIR / "xgboost_model.joblib")
    print("✓ xgboost_model.joblib")
except (NameError, KeyError):
    print("✗ XGBoost model not found — run Section 7 first")

# ── 6. LightGBM ───────────────────────────────────────────────
try:
    lgbm_model = MODELS["LightGBM"]
    joblib.dump(lgbm_model, ARTIFACTS_DIR / "lightgbm_model.joblib")
    print("✓ lightgbm_model.joblib")
except (NameError, KeyError):
    print("✗ LightGBM model not found — run Section 7 first")

print("=" * 55)
print("Done. Copy the app/artifacts/ folder to your API server.")
