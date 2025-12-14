import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import os

def train_model():
    # Load Data
    dataset_path = 'Sleep_health_clean.csv'
    
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join('MLProject', 'Sleep_health_clean.csv')

    print(f"Memuat data dari: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print("Error: File csv tidak ditemukan.")
        return

    # Sesuaikan nama kolom target
    target_col = 'Sleep Disorder' if 'Sleep Disorder' in df.columns else 'Sleep_Disorder'
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set Experiment
    mlflow.set_experiment("CI_Workflow_Experiment")

    # Autolog
    mlflow.autolog()

    with mlflow.start_run(run_name="CI_Run_RandomForest"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi: {acc}")
        
        # Simpan metrik manual agar terbaca di log
        mlflow.log_metric("accuracy_manual", acc)

if __name__ == "__main__":
    train_model()