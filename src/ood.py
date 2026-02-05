import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

def train_ood_detector(data_path, model_path):
    print("Loading data for OOD training...")
    df = pd.read_csv(data_path)
    
    # We only care about features, not the target. 
    # And specifically, we train on ALL data (fraud included) or just legitimate?
    # Usually, we want to know if a sample is different from "normal operation".
    # Since fraud is rare (5%), training on all data is fine for IF.
    # It will learn the "bulk" of the distribution.
    # Drop IDs
    X = df.drop(columns=['is_fraud', 'order_id', 'customer_id'], errors='ignore')
    
    # Preprocessing
    categorical_features = ['product_category', 'payment_method']
    numerical_features = ['order_value', 'return_ratio', 'days_since_purchase', 'returns_velocity_30d']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Isolation Forest
    # contamination='auto' or 0.05 (match fraud rate). 
    # 'auto' is safer if we don't assume we know the outlier rate.
    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('ood_model', iso_forest)])
    
    print("Training Isolation Forest...")
    pipeline.fit(X)
    
    # Save
    joblib.dump(pipeline, model_path)
    print(f"OOD Model saved to {model_path}")

def score_ood(model_path, data_path, output_path=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"OOD Model not found at {model_path}")
        
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X = df.drop(columns=['is_fraud', 'order_id'], errors='ignore')
    
    print(f"Scoring OOD for {X.shape[0]} rows...")
    
    # decision_function: Average anomaly score of X of the base classifiers.
    # The anomaly score of an input sample is computed as
    # the mean anomaly score of the trees in the forest.
    # The measure of normality of an observation given a tree is the depth
    # of the leaf containing this observation, which is equivalent to
    # the number of splittings required to isolate this point.
    # In case of several observations n_left in the leaf, the average
    # path length of a n_left samples isolation tree is added.
    
    # Higher is better (more normal). Lower/Negative is anomalous.
    raw_scores = model.decision_function(X)
    
    # We want a "Risk Score" where 1 = Anomaly, 0 = Normal.
    # IF output: positive = normal, negative = outlier.
    # Min/Max scaling is tricky because range isn't fixed.
    # Let's simple invert and normalize somewhat based on training distribution, 
    # or just use the raw score for now and threshold later.
    # For abstain logic, we'll want a threshold.
    
    # Let's save raw scores.
    results = df[['order_id']].copy()
    if 'is_fraud' in df.columns:
        results['is_fraud'] = df['is_fraud']
        
    results['ood_score'] = raw_scores # Higher = Normal
    results['is_anomaly'] = model.predict(X) # -1 = Anomaly, 1 = Normal
    
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"OOD scores saved to {output_path}")
        
    return results

if __name__ == "__main__":
    # Create models dir if not exists (already done in step 2 but good practice)
    os.makedirs("decision-safe-ml/models", exist_ok=True)
    
    train_ood_detector(
        "decision-safe-ml/data/processed/returns_normal.csv",
        "decision-safe-ml/models/ood_iforest.joblib"
    )
