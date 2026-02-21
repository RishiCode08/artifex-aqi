"""
ARTIFEX AQI — Random Forest Prediction API
Vercel Python Serverless Function

Endpoint: GET /api/predict?ppm=v1,v2,v3,v4,v5,v6&temp=29.8&hum=51&hour=19
Returns:  { predicted_ppm, status, method, confidence, meta }
"""

import json
import os
import numpy as np

# ── Model loader (cached across warm invocations) ─────────────────────────────
_CACHE = {}

def load_model():
    if "model" in _CACHE:
        return _CACHE["model"], _CACHE["meta"]

    model_path = os.path.join(os.path.dirname(__file__), "..", "model.pkl")

    try:
        import joblib
        bundle = joblib.load(model_path)
        _CACHE["model"] = bundle["model"]
        _CACHE["meta"]  = bundle["meta"]
        return _CACHE["model"], _CACHE["meta"]
    except Exception as e:
        raise RuntimeError(f"Could not load model.pkl: {e}")


# ── Handler ───────────────────────────────────────────────────────────────────
def handler(request):
    headers = {
        "Content-Type":                "application/json",
        "Access-Control-Allow-Origin": "*",
        "Cache-Control":               "no-store",
    }

    try:
        # Parse query params
        params   = request.args if hasattr(request, "args") else {}
        ppm_str  = params.get("ppm",  "")
        temp_str = params.get("temp", "25")
        hum_str  = params.get("hum",  "50")
        hour_str = params.get("hour", "12")

        if not ppm_str:
            return _resp(400, {"error": "Missing required param: ppm"}, headers)

        ppm_values = [float(x) for x in ppm_str.split(",") if x.strip()]
        temp       = float(temp_str)
        hum        = float(hum_str)
        hour       = int(float(hour_str))

        model, meta = load_model()
        WINDOW      = meta.get("window", 6)

        if len(ppm_values) < WINDOW:
            return _resp(400, {
                "error": f"Need at least {WINDOW} PPM values, got {len(ppm_values)}"
            }, headers)

        # Build feature vector (must match training exactly)
        window_ppm = ppm_values[-WINDOW:]
        roll_mean  = float(np.mean(window_ppm))
        roll_std   = float(np.std(window_ppm))

        features   = window_ppm + [temp, hum, hour, roll_mean, roll_std]
        X          = np.array([features], dtype=np.float32)

        # Predict + get tree-level variance as confidence proxy
        prediction    = float(model.predict(X)[0])
        tree_preds    = np.array([t.predict(X)[0] for t in model.estimators_])
        std_dev       = float(np.std(tree_preds))
        confidence    = max(0, round(100 - (std_dev / max(prediction, 1)) * 100, 1))

        pred_rounded  = round(prediction, 1)

        # AQI label
        def aqi_label(ppm):
            if ppm < 80:  return "Good"
            if ppm < 150: return "Moderate"
            if ppm < 250: return "Unhealthy"
            return "Hazardous"

        body = {
            "predicted_ppm": pred_rounded,
            "status":        aqi_label(pred_rounded),
            "method":        "Random Forest Regression",
            "confidence":    confidence,
            "std_dev":       round(std_dev, 2),
            "meta": {
                "window":    WINDOW,
                "mae":       meta.get("mae"),
                "r2":        meta.get("r2"),
                "n_samples": meta.get("n_samples"),
                "trained_at":meta.get("trained_at"),
            }
        }
        return _resp(200, body, headers)

    except RuntimeError as e:
        # model not found
        return _resp(503, {"error": str(e), "hint": "Run train_model.py first"}, headers)
    except Exception as e:
        return _resp(500, {"error": str(e)}, headers)


def _resp(status, body, headers):
    from http import HTTPStatus
    return (json.dumps(body), status, headers)
