import pandas as pd
import numpy as np
import sys

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI) for a single feature.
    
    Args:
        expected: numpy array of original values (training data)
        actual: numpy array of new values (production data)
        buckets: number of quantile buckets
    
    Returns:
        psi_value: float
    """
    
    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    
    if len(np.unique(expected)) < buckets:
        # For categorical or low cardinality, simply use value counts
        # But for this simple implementation, let's skip or handle gracefully.
        # We'll stick to numerical for now.
        return 0.0

    breakpoints = np.percentile(expected, breakpoints)
    
    # Handle duplicate breakpoints (e.g. many zeros)
    breakpoints = np.unique(breakpoints)
    
    # Calculate counts
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    
    return psi_value

def monitor_drift(train_path, current_path, output_path=None):
    print(f"Comparing {train_path} (Reference) vs {current_path} (Current)...")
    
    df_train = pd.read_csv(train_path)
    df_current = pd.read_csv(current_path)
    
    features = ['order_value', 'return_ratio', 'days_since_purchase']
    
    drift_report = []
    
    print("\n--- Drift Report ---")
    for feat in features:
        psi = calculate_psi(df_train[feat].values, df_current[feat].values)
        
        status = "STABLE"
        if psi > 0.1:
            status = "WARNING"
        if psi > 0.2:
            status = "CRITICAL"
            
        print(f"Feature: {feat:<20} | PSI: {psi:.4f} | Status: {status}")
        
        drift_report.append({
            'feature': feat,
            'psi': psi,
            'status': status
        })
        
    if output_path:
        pd.DataFrame(drift_report).to_csv(output_path, index=False)
        print(f"\nDrift report saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    TRAIN_DATA = "decision-safe-ml/data/processed/returns_normal.csv"
    
    # Let's check mild drift
    MILD_DATA = "decision-safe-ml/data/drifted/returns_mild_drift.csv"
    SEVERE_DATA = "decision-safe-ml/data/drifted/returns_severe_drift.csv"
    
    print("Checking MILD drift:")
    monitor_drift(TRAIN_DATA, MILD_DATA)
    
    print("\nChecking SEVERE drift:")
    monitor_drift(TRAIN_DATA, SEVERE_DATA)
