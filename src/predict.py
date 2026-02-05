import pandas as pd
import joblib
import sys
import os

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)

def predict(model_path, data_path, output_path=None):
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Check if target column exists (ground truth for evaluation)
    has_target = 'is_fraud' in df.columns
    if has_target:
        X = df.drop(columns=['is_fraud', 'order_id'], errors='ignore')
    else:
        X = df.drop(columns=['order_id'], errors='ignore')
        
    print(f"Predicting on {X.shape[0]} rows...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Create results DataFrame
    results = df[['order_id']].copy()
    results['predicted_fraud'] = predictions
    results['fraud_probability'] = probabilities
    
    if has_target:
        results['actual_fraud'] = df['is_fraud']
        
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run predictions using trained model")
    parser.add_argument("data_path", help="Path to input CSV data")
    parser.add_argument("--model_path", default="decision-safe-ml/models/rf_baseline.joblib", help="Path to trained model")
    parser.add_argument("--output_path", default=None, help="Path to save predictions CSV")
    
    args = parser.parse_args()
    
    predict(args.model_path, args.data_path, args.output_path)
