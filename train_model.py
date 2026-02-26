"""
ARTIFEX AQI — Random Forest Model Trainer (v2)
Saves model as JSON inside public/ — readable by Node.js on Vercel Hobby plan

Run: python train_model.py
"""

import json, os, sys, numpy as np
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import firebase_admin
    from firebase_admin import credentials, db as firebase_db
except ImportError:
    print("ERROR: pip install firebase-admin"); sys.exit(1)

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
except ImportError:
    print("ERROR: pip install scikit-learn"); sys.exit(1)

DB_URL    = os.environ.get("FIREBASE_DATABASE_URL",
            "https://esp32-airquality-monitor-default-rtdb.asia-southeast1.firebasedatabase.app")
DEVICE_ID = os.environ.get("DEVICE_ID", "AIR-SENSE-PRO")
CRED_FILE = "serviceAccountKey.json"
WINDOW    = 6
OUT_PATH  = "public/model.json"

if not os.path.exists(CRED_FILE):
    print(f"ERROR: {CRED_FILE} not found"); sys.exit(1)

print("Connecting to Firebase...")
cred = credentials.Certificate(CRED_FILE)
firebase_admin.initialize_app(cred, {"databaseURL": DB_URL})
data = firebase_db.reference(f"devices/{DEVICE_ID}/readings").get()

if not data:
    print("ERROR: No data in Firebase"); sys.exit(1)

print(f"Fetched {len(data)} records")

rows = []
for key, val in sorted(data.items()):
    try:
        rows.append({
            "ts":          key,
            "gas_ppm":     float(val.get("gas_ppm",     0)),
            "temperature": float(val.get("temperature", 0)),
            "humidity":    float(val.get("humidity",    0)),
        })
    except:
        continue

print(f"Parsed {len(rows)} valid readings")

def parse_hour(ts):
    try:
        s = ts.replace("_",":").strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S","%Y-%m-%d %H:%M:%S"):
            try: return datetime.strptime(s[:19], fmt).hour
            except: continue
    except: pass
    return 12

ppms  = [r["gas_ppm"]     for r in rows]
temps = [r["temperature"] for r in rows]
hums  = [r["humidity"]    for r in rows]

X, y = [], []
for i in range(WINDOW, len(ppms)):
    w = ppms[i-WINDOW:i]
    X.append(w + [temps[i-1], hums[i-1], parse_hour(rows[i]["ts"]),
                  float(np.mean(w)), float(np.std(w))])
    y.append(ppms[i])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
print(f"Dataset: {len(X)} samples, {X.shape[1]} features")

if len(X) >= 20:
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    Xtr,Xte,ytr,yte = X,X,y,y

print("Training Random Forest (100 trees)...")
model = RandomForestRegressor(n_estimators=100, max_depth=8,
                               min_samples_split=3, random_state=42, n_jobs=-1)
model.fit(Xtr, ytr)
ypred = model.predict(Xte)
mae   = float(mean_absolute_error(yte, ypred))
r2    = float(r2_score(yte, ypred))
print(f"\n── Model Performance ──")
print(f"   MAE : {mae:.2f} PPM")
print(f"   R2  : {r2:.3f}")

def export_tree(tree):
    t = tree.tree_
    def recurse(node):
        if t.children_left[node] == -1:
            return {"leaf": True, "value": float(t.value[node][0][0])}
        return {"leaf": False, "feat": int(t.feature[node]),
                "thr": float(t.threshold[node]),
                "left": recurse(t.children_left[node]),
                "right": recurse(t.children_right[node])}
    return recurse(0)

print(f"Exporting {len(model.estimators_)} trees to JSON...")
trees = [export_tree(e) for e in model.estimators_]

bundle = {
    "window": WINDOW, "n_features": int(X.shape[1]),
    "n_trees": len(trees), "mae": round(mae,3), "r2": round(r2,3),
    "n_samples": len(X), "trained_at": datetime.utcnow().isoformat(),
    "trees": trees
}

os.makedirs("public", exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(bundle, f, separators=(',',':'))

size_mb = os.path.getsize(OUT_PATH)/1024/1024
print(f"Model saved -> {OUT_PATH}  ({size_mb:.1f} MB)")
print("Next: git add public/model.json && git commit -m 'RF model JSON' && git push")
