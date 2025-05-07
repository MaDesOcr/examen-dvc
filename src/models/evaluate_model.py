import pandas as pd
import joblib
import json
import os
import numpy as np
from sklearn import metrics

def main():
    processed_dir = os.path.join("data", "processed_data")
    models_dir = "models"
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    model = joblib.load(os.path.join(models_dir, "model_trained.pkl"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()

    y_pred = model.predict(X_test)

    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)

    scores = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    with open(os.path.join(metrics_dir, "scores.json"), 'w') as f:
        json.dump(scores, f, indent=4)

    preds = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
    preds.to_csv(os.path.join("data", "predictions.csv"), index=False)

if __name__ == "__main__":
    main()