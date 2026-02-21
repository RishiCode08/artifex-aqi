"""
ARTIFEX AQI — Random Forest Model Trainer
==========================================
Run this script ONCE on your local machine to train the model.
It reads your Firebase data, trains a Random Forest, and saves model.pkl

Requirements:
    pip install firebase-admin scikit-learn numpy joblib python-dotenv

Usage:
    python train_model.py
"""

import json
import numpy as np
import joblib
import os
import sys
from datetime import datetime

# ── Try loading .env ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional

try:
    import firebase_admin
    from firebase_admin import credentials, db as firebase_db
except ImportError:
    print("ERROR: Run  pip install firebase-admin  first")
    sys.exit(1)

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
except ImportError:
    print("ERROR: Run  pip install scikit-learn  first")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
DB_URL    = os.environ.get("FIREBASE_DATABASE_URL",
            "https://esp32-airquality-monitor-default-rtdb.asia-southeast1.firebasedatabase.app")
DEVICE_ID = os.environ.get("DEVICE_ID", "AIR-SENSE-PRO")
CRED_FILE = "serviceAccountKey.json"
WINDOW    = 6   # use last 6 readings (each 15s apart = ~1.5 min window) to predict next
OUT_PATH  = "model.pkl"

# ── Connect Firebase ──────────────────────────────────────────────────────────
if not os.path.exists(CRED_FILE):
    print(f"ERROR: {CRED_FILE} not found.")
    print("  → Go to Firebase Console → Project Settings → Service Accounts → Generate new private key")
    print(f"  → Save the downloaded file as  {CRED_FILE}  in this folder")
    sys.exit(1)

print("Connecting to Firebase...")
cred = credentials.Certificate(CRED_FILE)
firebase_admin.initialize_app(cred, {"databaseURL": DB_URL})

ref  = firebase_db.reference(f"devices/{DEVICE_ID}/readings")
data = ref.get()

if not data:
    print("ERROR: No data found in Firebase. Make sure your ESP32 has uploaded readings.")
    sys.exit(1)

print(f"Fetched {len(data)} raw records from Firebase")

# ── Parse ─────────────────────────────────────────────────────────────────────
rows = []
for key, val in sorted(data.items()):
    try:
        rows.append({
            "ts":          key,
            "gas_ppm":     float(val.get("gas_ppm",     0)),
            "temperature": float(val.get("temperature", 0)),
            "humidity":    float(val.get("humidity",    0)),
        })
    except Exception:
        continue

print(f"Parsed {len(rows)} valid readings")

if len(rows) < WINDOW + 10:
    print(f"WARNING: Only {len(rows)} readings. Need at least {WINDOW + 10} for meaningful training.")
    print("  → Let your ESP32 run longer and retry")

# ── Feature Engineering ───────────────────────────────────────────────────────
# Features per sample:
#   - last WINDOW ppm values       (sliding window)
#   - last temperature value       (context)
#   - last humidity value          (context)
#   - hour of day (0-23)           (time pattern)
#   - rolling mean of window       (trend context)
#   - rolling std  of window       (volatility)

ppms  = [r["gas_ppm"]     for r in rows]
temps = [r["temperature"] for r in rows]
hums  = [r["humidity"]    for r in rows]

def parse_hour(ts_str):
    """Extract hour from ISO timestamp key."""
    try:
        s = ts_str.replace("_", ":").strip()
        # handle YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD HH:MM:SS
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(s[:len(fmt)], fmt).hour
            except ValueError:
                continue
    except Exception:
        pass
    return 12  # fallback midday

X, y = [], []
for i in range(WINDOW, len(ppms)):
    window_ppm = ppms[i - WINDOW : i]
    hour       = parse_hour(rows[i]["ts"])
    roll_mean  = float(np.mean(window_ppm))
    roll_std   = float(np.std(window_ppm))
    features   = window_ppm + [
        temps[i - 1],   # most recent temp
        hums[i - 1],    # most recent humidity
        hour,
        roll_mean,
        roll_std,
    ]
    X.append(features)
    y.append(ppms[i])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(f"Dataset: {len(X)} samples, {X.shape[1]} features each")

# ── Train ─────────────────────────────────────────────────────────────────────
if len(X) >= 20:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = X, X, y, y
    print("NOTE: Small dataset — using all data for both train and test")

print("Training Random Forest (100 trees)...")
model = RandomForestRegressor(
    n_estimators = 100,
    max_depth    = 8,
    min_samples_split = 3,
    random_state = 42,
    n_jobs       = -1,
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"\n── Model Performance ──────────────────")
print(f"   MAE (mean abs error) : {mae:.2f} PPM")
print(f"   R² score             : {r2:.3f}  (1.0 = perfect)")
print(f"───────────────────────────────────────")

# Feature importance
feature_names = [f"ppm_t-{WINDOW-i}" for i in range(WINDOW)] + [
    "temperature", "humidity", "hour", "roll_mean", "roll_std"
]
importances = model.feature_importances_
top = sorted(zip(feature_names, importances), key=lambda x: -x[1])
print("\nTop feature importances:")
for name, imp in top[:5]:
    print(f"   {name:<20} {imp:.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
meta = {
    "window":        WINDOW,
    "feature_names": feature_names,
    "n_features":    X.shape[1],
    "mae":           round(float(mae), 3),
    "r2":            round(float(r2),  3),
    "n_samples":     len(X),
    "trained_at":    datetime.utcnow().isoformat(),
}
joblib.dump({"model": model, "meta": meta}, OUT_PATH)
print(f"\nModel saved → {OUT_PATH}")
print("Next step: git add model.pkl && git commit -m 'add RF model' && git push")
