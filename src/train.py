import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def train_model(data_path, model_path):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Split features and target
    X = df.drop(columns=['is_fraud', 'order_id']) # Drop ID and Target
    X = df.drop(columns=['is_fraud', 'order_id', 'customer_id'], errors='ignore') # Drop ID and Target
    y = df['is_fraud']
    
    # Split train/test
    # We use a larger test set to ensure we have enough fraud examples for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"Training on {X_train.shape[0]} rows, testing on {X_test.shape[0]} rows.")
    
    # Preprocessing
    categorical_features = ['product_category', 'payment_method']
    numerical_features = ['order_value', 'return_ratio', 'days_since_purchase', 'returns_velocity_30d']
    
    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Classifier
    # Using Random Forest as baseline. 
    # Class weight 'balanced' helps with the 5% fraud rate.
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    
    # Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', rf)])
    
    # Probability Calibration
    # Although RFs are decent, explicitly calibrating them ensures that '0.8 confidence' actually means 80% risk.
    # This is critical for our Abstention logic.
    calibrated_clf = CalibratedClassifierCV(pipeline, method='isotonic', cv=3)
    
    print("Training Random Forest...")
    calibrated_clf.fit(X_train, y_train)
    
    # Evaluation
    print("\nEvaluating...")
    y_pred = calibrated_clf.predict(X_test)
    y_proba = calibrated_clf.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    brier = brier_score_loss(y_test, y_proba)
    print(f"Brier Score (Calibration): {brier:.4f} (Lower is better)")
    
    # Save model
    joblib.dump(calibrated_clf, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Feature Importance (Tricky with CalibratedClassifierCV + Pipeline, accessing inner RF)
    # We'll just hint at it or skip for checking the inner model if needed, 
    # but for production pipelines, we rely on the calibrated output.
    
if __name__ == "__main__":
    DATA_PATH = "decision-safe-ml/data/processed/returns_normal.csv"
    MODEL_PATH = "decision-safe-ml/models/rf_baseline.joblib"
    
    # Ensure model directory exists
    import os
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    train_model(DATA_PATH, MODEL_PATH)
