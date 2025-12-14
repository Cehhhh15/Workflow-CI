import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import os

def train_model():
    # Setup Path
    dataset_path = 'Sleep_health_clean.csv'
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join('MLProject', 'Sleep_health_clean.csv')

    print(f"Memuat data dari: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print("Error: File csv tidak ditemukan.")
        return

    # Handle Target Column
    target_col = 'Sleep Disorder' if 'Sleep Disorder' in df.columns else 'Sleep_Disorder'
    if target_col not in df.columns:
        print(f"Error: Kolom target '{target_col}' tidak ditemukan.")
        return

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Autolog
    mlflow.autolog()

    print("Memulai training...")
    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi: {acc}")
        
        # Log manual metrik tambahan
        mlflow.log_metric("accuracy_manual", acc)

        # Simpan Run ID
        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        print(f"Run ID {run_id} saved to run_id.txt")

if __name__ == "__main__":
    train_model()