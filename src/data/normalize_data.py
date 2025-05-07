import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def main():
    processed_dir = os.path.join("data", "processed_data")
    scaler_path = os.path.join("models", "scaler.pkl")

    # Lire jeux de données numériques uniquement
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    # Assurer uniquement colonnes numériques
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Sauvegarde
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(os.path.join(processed_dir, "X_train_scaled.csv"), index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(os.path.join(processed_dir, "X_test_scaled.csv"), index=False)

    # Sauvegarder le scaler pour usage ultérieur
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

if __name__ == "__main__":
    main()