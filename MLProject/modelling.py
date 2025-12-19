import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

def train_model():
    # Load Data
    print("Memuat data...")
    df = pd.read_csv('Sleep_health_clean.csv')

    # Pisahkan Fitur dan Target
    X = df.drop('Sleep Disorder', axis=1)
    y = df['Sleep Disorder']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set Experiment
    #mlflow.set_experiment("Sleep_Disorder_Basic")

    # Training dengan Autolog
    print("Mulai Training Basic...")
    mlflow.autolog()

    with mlflow.start_run(run_name="Basic_RandomForest"):
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi Model: {acc}")

if __name__ == "__main__":
    train_model()