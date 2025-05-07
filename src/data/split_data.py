import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    raw_path = os.path.join("data", "raw_data", "raw.csv")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    df = pd.read_csv(raw_path)
    X = df.drop(columns=["silica_concentrate"])
    y = df["silica_concentrate"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Sauvegarde
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

if __name__ == "__main__":
    main()