import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

def main():
    processed_dir = os.path.join("data", "processed")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Chargement des données
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()

    # Choix du modèle et grille
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)

    # Sauvegarder les meilleurs paramètres
    best_params = grid.best_params_
    joblib.dump(best_params, os.path.join(model_dir, "best_params.pkl"))

if __name__ == "__main__":
    main()