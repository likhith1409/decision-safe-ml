import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_base_data(n_rows):
    """Generates base synthetic e-commerce returns data."""
    
    # Generate Order IDs
    order_ids = [f"ORD-{i:08d}" for i in range(1, n_rows + 1)]
    
    # Feature 1: Order Value (Log-normal distribution to simulate prices)
    # Mean around $150, but with a long tail
    order_values = np.random.lognormal(mean=4.5, sigma=0.8, size=n_rows)
    order_values = np.round(order_values, 2)
    
    # Feature 2: Product Category
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Beauty', 'Toys']
    cat_probs = [0.3, 0.4, 0.15, 0.1, 0.05]
    product_categories = np.random.choice(categories, size=n_rows, p=cat_probs)
    
    # Feature 3: Customer Return History Ratio (previous_returns / previous_orders)
    # Most customers have low return rates. Some serial returners.
    return_history_ratio = np.random.beta(a=2, b=10, size=n_rows)
    
    # Feature 4: Time Since Purchase (Days)
    # Most returns happen quickly, some take longer.
    days_since_purchase = np.random.exponential(scale=7, size=n_rows).astype(int) + 1
    # Cap at 90 days
    days_since_purchase = np.minimum(days_since_purchase, 90)
    
    # Feature 5: Payment Method
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Gift Card']
    pay_probs = [0.6, 0.25, 0.1, 0.05]
    methods = np.random.choice(payment_methods, size=n_rows, p=pay_probs)
    
    # Feature: Customer ID (New for Phase 3)
    # 10,000 unique customers for base data
    customer_ids = [f"CUST-{i:05d}" for i in np.random.randint(1, 10001, size=n_rows)]
    
    # Feature 6: Returns Velocity (30d) - NEW for Phase 2
    # Most people return 0-1 items. Fraudsters return many.
    returns_velocity_30d = np.random.poisson(lam=0.5, size=n_rows)
    
    # DataFrame construction
    df = pd.DataFrame({
        'order_id': order_ids,
        'customer_id': customer_ids,
        'order_value': order_values,
        'product_category': product_categories,
        'return_ratio': np.round(return_history_ratio, 3),
        'returns_velocity_30d': returns_velocity_30d,
        'days_since_purchase': days_since_purchase,
        'payment_method': methods
    })
    
    return df

def inject_fraud(df):
    """Injects fraud patterns based on rules + noise."""
    n = len(df)
    is_fraud = np.zeros(n, dtype=int)
    
    # Fraud signals (probabilistic)
    
    # 1. High value electronics are higher risk
    mask_high_value_elec = (df['product_category'] == 'Electronics') & (df['order_value'] > 500)
    is_fraud[mask_high_value_elec] = np.random.choice([0, 1], size=mask_high_value_elec.sum(), p=[0.8, 0.2])
    
    # 2. Serial returners (high return ratio)
    mask_serial = df['return_ratio'] > 0.7
    is_fraud[mask_serial] = np.random.choice([0, 1], size=mask_serial.sum(), p=[0.7, 0.3])
    
    # NEW: High Velocity Fraud (but keep order value normal-ish to test feature importance)
    mask_velocity = df['returns_velocity_30d'] > 4
    is_fraud[mask_velocity] = np.random.choice([0, 1], size=mask_velocity.sum(), p=[0.4, 0.6])
    
    # 3. Quick returns on Gift Cards (classic money laundering / fraud pattern)
    mask_gc_quick = (df['payment_method'] == 'Gift Card') & (df['days_since_purchase'] < 3)
    is_fraud[mask_gc_quick] = np.random.choice([0, 1], size=mask_gc_quick.sum(), p=[0.6, 0.4])
    
    # 4. Random background noise (hard to detect fraud)
    noise_idx = np.random.choice(n, size=int(n * 0.01), replace=False)
    is_fraud[noise_idx] = 1
    
    df['is_fraud'] = is_fraud
    return df

def apply_drift(df, severity='none'):
    """Applies drift to the dataset."""
    df_drifted = df.copy()
    n = len(df)
    
    if severity == 'none':
        return df_drifted
        
    elif severity == 'mild':
        print("Applying MILD drift...")
        # 1. Inflation: Shift order values up by 15%
        df_drifted['order_value'] = df_drifted['order_value'] * 1.15
        
        # 2. Category shift: Winter season? More clothing returns.
        mask_clothing = np.random.choice([True, False], size=n, p=[0.2, 0.8])
        df_drifted.loc[mask_clothing, 'product_category'] = 'Clothing'
        
    elif severity == 'severe':
        print("Applying SEVERE drift...")
        # 1. New payment fraud ring: PayPal exploitation
        mask_paypal_shift = np.random.choice([True, False], size=n, p=[0.3, 0.7])
        df_drifted.loc[mask_paypal_shift, 'payment_method'] = 'PayPal'
        
        # 2. MIMICRY ATTACK (New for Phase 2)
        # Force-inject 200 mimicry cases (Low Value, High Velocity)
        # FRAUD RING SIMULATION (Phase 3): 
        # These 200 cases come from just 20 compromised accounts (10 returns each).
        # This allows the Budget Engine to catch them after ~$50 loss.
        
        n_mimics = 200
        n_compromised_accts = 20
        compromised_ids = [f"RING-{i:03d}" for i in range(n_compromised_accts)]
        
        mimic_customer_ids = np.random.choice(compromised_ids, size=n_mimics)
        mimic_ids = [f"MIMIC-{i:04d}" for i in range(n_mimics)]
        mimic_vals = np.random.uniform(20, 30, n_mimics) # Low value ($20-$30)
        mimic_vels = np.random.randint(4, 10, n_mimics)  # High velocity
        mimic_cats = np.random.choice(['Electronics', 'Beauty'], n_mimics)
        mimic_ratios = np.random.beta(2, 5, n_mimics) # Normal-ish ratio
        mimic_days = np.random.randint(1, 10, n_mimics)
        mimic_pay = ['Credit Card'] * n_mimics 
        
        df_mimics = pd.DataFrame({
            'order_id': mimic_ids,
            'customer_id': mimic_customer_ids,
            'order_value': mimic_vals,
            'product_category': mimic_cats,
            'return_ratio': np.round(mimic_ratios, 3),
            'returns_velocity_30d': mimic_vels,
            'days_since_purchase': mimic_days,
            'payment_method': mimic_pay,
            'is_fraud': 1
        })
        
        df_drifted = pd.concat([df_drifted, df_mimics], ignore_index=True)
        print(f"  -> Injected Mimicry Attack: {n_mimics} cases from {n_compromised_accts} compromised IDs (The 'Ring')")
            
    return df_drifted

def main():
    base_path = "decision-safe-ml/data"
    
    # 1. Generate Training Data (Normal)
    print("Generating Normal Data...")
    df_normal = generate_base_data(10000)
    df_normal = inject_fraud(df_normal)
    df_normal.to_csv(f"{base_path}/processed/returns_normal.csv", index=False)
    print(f"Saved returns_normal.csv: {df_normal.shape}, Fraud Rate: {df_normal['is_fraud'].mean():.2%}")
    
    # 2. Generate Mild Drift
    print("\nGenerating Mild Drift Data...")
    df_mild = generate_base_data(5000) # Smaller batch for testing
    df_mild = inject_fraud(df_mild)    # Base fraud logic
    df_mild = apply_drift(df_mild, severity='mild')
    df_mild.to_csv(f"{base_path}/drifted/returns_mild_drift.csv", index=False)
    print(f"Saved returns_mild_drift.csv: {df_mild.shape}")
    
    # 3. Generate Severe Drift
    print("\nGenerating Severe Drift Data...")
    df_severe = generate_base_data(5000)
    df_severe = inject_fraud(df_severe)
    df_severe = apply_drift(df_severe, severity='severe')
    df_severe.to_csv(f"{base_path}/drifted/returns_severe_drift.csv", index=False)
    print(f"Saved returns_severe_drift.csv: {df_severe.shape}, Fraud Rate: {df_severe['is_fraud'].mean():.2%}")

if __name__ == "__main__":
    main()
