"""
Ford Car Price Predictor — Model Trainer
Run this ONCE to train and save the model before starting app.py
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib, json, os

# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------
DATA_PATH  = 'ford.csv'
MODEL_DIR  = 'model'
MODEL_FILE = os.path.join(MODEL_DIR, 'ford_model.pkl')
META_FILE  = os.path.join(MODEL_DIR, 'metadata.json')

os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------------------------------------------
# LOAD & CLEAN
# ----------------------------------------------------------------
print("📦 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df['model'] = df['model'].str.strip()

print(f"   Raw records: {len(df)}")
df = df[(df['year'] >= 2000) & (df['year'] <= 2025)]
df = df[(df['price'] >= 500) & (df['price'] <= 60000)]
df = df.dropna()
print(f"   Clean records: {len(df)}")

# ----------------------------------------------------------------
# FEATURES & TARGET
# ----------------------------------------------------------------
features = ['model','year','transmission','mileage','fuelType','tax','mpg','engineSize']
X = df[features].copy()
y = df['price']

# ----------------------------------------------------------------
# ENCODING
# ----------------------------------------------------------------
le_model = LabelEncoder()
le_trans = LabelEncoder()
le_fuel  = LabelEncoder()

X['model']        = le_model.fit_transform(X['model'])
X['transmission'] = le_trans.fit_transform(X['transmission'])
X['fuelType']     = le_fuel.fit_transform(X['fuelType'])

# ----------------------------------------------------------------
# SCALING
# ----------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ----------------------------------------------------------------
# MODEL TRAINING
# ----------------------------------------------------------------
print("\n🌲 Training Random Forest (200 trees)...")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ----------------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------------
y_pred = rf.predict(X_test)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 Model Performance:")
print(f"   R² Score : {r2:.4f} ({r2*100:.1f}%)")
print(f"   MAE      : £{mae:,.0f}")
print(f"   RMSE     : £{rmse:,.0f}")

# ----------------------------------------------------------------
# SAVE MODEL
# ----------------------------------------------------------------
joblib.dump({
    'model':    rf,
    'scaler':   scaler,
    'le_model': le_model,
    'le_trans': le_trans,
    'le_fuel':  le_fuel,
}, MODEL_FILE)
print(f"\n✅ Model saved → {MODEL_FILE}")

# ----------------------------------------------------------------
# SAVE METADATA
# ----------------------------------------------------------------
raw = pd.read_csv(DATA_PATH)
raw.columns = raw.columns.str.strip()
raw['model'] = raw['model'].str.strip()
raw = raw[(raw['year'] >= 2000) & (raw['year'] <= 2025)]

meta = {
    'models':        sorted(raw['model'].unique().tolist()),
    'transmissions': raw['transmission'].unique().tolist(),
    'fuelTypes':     raw['fuelType'].unique().tolist(),
    'yearMin':       int(raw['year'].min()),
    'yearMax':       int(raw['year'].max()),
    'mileageMax':    int(raw['mileage'].max()),
    'engineSizes':   sorted(raw['engineSize'].unique().tolist()),
    'r2':            round(r2 * 100, 1),
    'mae':           round(mae, 0),
    'samples':       len(raw)
}

with open(META_FILE, 'w') as f:
    json.dump(meta, f, indent=2)
print(f"✅ Metadata saved → {META_FILE}")
print(f"\n🚀 Ready! Now run:  python app.py")