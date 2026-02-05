import pandas as pd
import numpy as np
import os

def analyze_failures(decisions_path, output_md_path):
    print(f"Analyzing decisions from {decisions_path}...")
    df = pd.read_csv(decisions_path)
    
    # 1. Identify Leakage: Decision = APPROVE but is_fraud = 1
    leakage = df[(df['decision'] == 'APPROVE') & (df['is_fraud'] == 1)].copy()
    
    n_leakage = len(leakage)
    n_total_fraud = df['is_fraud'].sum()
    n_total_approvals = (df['decision'] == 'APPROVE').sum()
    
    print(f"Total Fraud Cases: {n_total_fraud}")
    print(f"Total Approvals: {n_total_approvals}")
    print(f"Leaked Fraud Cases: {n_leakage}")
    
    # merge with original data to get features?
    # Ideally decisions.csv should have been saved with features.
    # But abstention.py only saved ID, probs, decisions.
    # We need to load severe drift data to match.
    
    # Let's assume we can merge on order_id if needed, but for now let's just use what we have in decisions
    # if we saved features there. We didn't save features in abstention.py, only ID/Prob/Reason.
    # So we MUST load the source data.
    
    # HACK: Hardcoding the source data path for this analysis, knowing it came from severe drift.
    source_data_path = "decision-safe-ml/data/drifted/returns_severe_drift.csv"
    print(f"Loading source features from {source_data_path}...")
    df_source = pd.read_csv(source_data_path)
    
    # Merge
    # DROP columns from source that are already in decisions (except key)
    cols_to_use = df_source.columns.difference(df.columns).tolist()
    cols_to_use.append('order_id')
    
    leakage_full = leakage.merge(df_source[cols_to_use], on='order_id', how='left')
    
    # Analysis of Leakage
    report_lines = []
    report_lines.append("# Failure Analysis Postmortem (Phase 2: Economic View)")
    report_lines.append(f"**Date:** {pd.Timestamp.now().date()}")
    report_lines.append(f"**Dataset:** Severe Drift Scenario")
    report_lines.append(f"**Total Leaked Fraud Cases:** {n_leakage} (out of {n_total_fraud} total fraud)")
    
    # ECONOMIC IMPACT
    leakage_cost = leakage_full['order_value'].sum()
    report_lines.append(f"**Total Leakage Cost (Realized Loss):** ${leakage_cost:,.2f}")
    
    if n_leakage > 0:
        report_lines.append("## Why did we miss them?")
        
        # 1. Probability Distribution
        avg_leak_prob = leakage_full['fraud_prob'].mean()
        report_lines.append(f"- **Average Model Confidence on Leaks:** {avg_leak_prob:.4f}")
        
        # 2. OOD Scores
        avg_ood_score = leakage_full['ood_score'].mean()
        report_lines.append(f"- **Average OOD Score on Leaks:** {avg_ood_score:.4f}")
        
        # 3. Categorical Breakdown
        report_lines.append("\n## Leakage by Category")
        cat_counts = leakage_full['product_category'].value_counts().to_dict()
        for cat, count in cat_counts.items():
            report_lines.append(f"- {cat}: {count} cases")
            
        # 4. Specific Examples
        report_lines.append("\n## Top 5 Leaked Examples (Highest Cost)")
        examples = leakage_full.sort_values('order_value', ascending=False).head(5)[
            ['order_id', 'order_value', 'product_category', 'returns_velocity_30d', 'fraud_prob', 'ood_score']
        ]
        report_lines.append(examples.to_markdown(index=False))
        
        report_lines.append("\n## Root Cause Hypothesis")
        # Logic for Mimicry
        mimics = leakage_full[
            (leakage_full['order_value'] < 50) & 
            (leakage_full['returns_velocity_30d'] > 3)
        ]
        if len(mimics) > 0:
             report_lines.append(f"- **Mimicry Attack Detected**: {len(mimics)} cases had low value (<$50) but high velocity. They were approved because 'Cost(Approve)' [<$50] was cheaper than 'Cost(Review)' [$50].")
             report_lines.append("  - *Economic Rationale*: The system made the 'right' economic choice (cheaper to lose $30 than pay $50 for review), but this exposes a strategic vulnerability to high-volume low-value attacks.")
             
    else:
        report_lines.append("No leakage detected! System is perfectly safe (or too conservative).")

    # Write Report
    with open(output_md_path, 'w') as f:
        f.write('\n'.join(report_lines))
        
    print(f"Analysis saved to {output_md_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", default="decision-safe-ml/reports/decisions.csv")
    parser.add_argument("--output", default="decision-safe-ml/reports/failure_cases.md")
    args = parser.parse_args()
    
    analyze_failures(args.decisions, args.output)
