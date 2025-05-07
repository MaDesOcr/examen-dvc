import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def main():
    processed_dir = os.path.join("data", "processed")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Charger paramètres et données
    best_params = joblib.load(os.path.join(model_dir, "best_params.pkl"))
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()

    # Entraînement
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    # Sauvegarde du modèle entraîné
    joblib.dump(model, os.path.join(model_dir, "model_trained.pkl"))

if __name__ == "__main__":
    main()