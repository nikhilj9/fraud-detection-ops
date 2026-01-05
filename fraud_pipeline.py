# fraud_pipeline.py
from prefect import flow, task
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
from prefect.artifacts import create_markdown_artifact
from sklearn.metrics import confusion_matrix  # For the scoreboard
from datetime import timedelta
from prefect.tasks import task_input_hash


# 1. THE SCOUT: Finds and retrieves data
# Update ONLY the load_data task
@task(
    name="load-data", 
    log_prints=True, 
    retries=3, 
    retry_delay_seconds=2,
    # CACHING STRATEGY:
    # 1. Calculate a "fingerprint" of the inputs (task_input_hash)
    # 2. Keep the result valid for 1 hour
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)
def load_data():
    print("Attempting to connect to data warehouse...")
    
    # Simulate a network glitch (70% chance of failure)
    if random.random() < 0.7:
        print("⚠ CONNECTION FAILED! (Simulating 503 Error)")
        raise ValueError("Network Glitch detected")
        
    print("✓ Connection established.")
    print("dataset loading...")
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'amount': np.random.exponential(100, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'risk_score': np.random.uniform(0, 1, n_samples)
    })
    y = (X['amount'] > 500) | (X['risk_score'] > 0.8)
    y = y.astype(int)
    return X, y

# 2. THE BARRACKS: Trains the unit
@task(name="train-model", log_prints=True)
def train_model(X, y):
    print("Training Random Forest...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# 3. THE JUDGE: Evaluates performance
@task(name="evaluate-model", log_prints=True)
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Generate Confusion Matrix (True Scoreboard)
    cm = confusion_matrix(y_test, predictions)
    
    # Format as Markdown Table for UI
    markdown = f"""
    # Fraud Detection Scoreboard

    **Accuracy: {accuracy:.2%}**

    | Predicted \\ True | Non-Fraud (0) | Fraud (1) |
    |-------------------|---------------|-----------|
    | **Non-Fraud (0)** | {cm[0,0]}     | {cm[0,1]} |
    | **Fraud (1)**     | {cm[1,0]}     | {cm[1,1]} |
        
    **Total Tested: {len(y_test)} transactions**
        """
    
    # Push to UI (appears in Flow Run details)
    create_markdown_artifact(
        markdown=markdown,
        key="confusion-matrix"
    )
    
    print(f"Model Accuracy: {accuracy:.2%}")
    return accuracy

# THE COMMANDER: Orchestrates the sequence
@flow(name="fraud-detection-modular", log_prints=True)
def fraud_pipeline():
    # Notice we pass data between tasks just like Python functions
    X, y = load_data()
    model, X_test, y_test = train_model(X, y)
    accuracy = evaluate_model(model, X_test, y_test)
    return accuracy

if __name__ == "__main__":
    fraud_pipeline()