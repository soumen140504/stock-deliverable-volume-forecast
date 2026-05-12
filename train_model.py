import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/ADANIPORTS.csv"
df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date")

# We do NOT use %Deliverble because it may create data leakage.
features = [
    "Prev Close", "Open", "High", "Low", "Last",
    "Close", "VWAP", "Volume", "Turnover"
]

target = "Deliverable Volume"

model_df = df[features + [target]].copy().dropna()

X = model_df[features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("Prediction Type: Current-Day Deliverable Volume Prediction")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/stock_deliverable_model.joblib")
joblib.dump(features, "models/features.joblib")

metrics = {
    "Prediction Type": "Current-Day Deliverable Volume Prediction",
    "MAE": round(float(mae), 2),
    "RMSE": round(float(rmse), 2),
    "R2 Score": round(float(r2), 4),
    "Target": target,
    "Features": features
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel saved successfully in models/ folder.")
