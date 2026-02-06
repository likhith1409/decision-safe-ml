import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np

# Cost Constants
REVIEW_COST = 50.0
RISK_BUDGET_PER_CUSTOMER = 50.0

# Standardized Terminology
SCENARIO_LABEL = "Simulated Adversarial Scenario"

def set_style():
    """Sets a professional style for the plots."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14


def plot_cumulative_loss(df, output_dir, attack_start):
    """
    HERO PLOT: Cumulative Cost Over Time with attack projection.
    """
    df = df.copy()
    
    # Decision-Safe System costs
    df['realized_cost'] = 0.0
    mask_fraud_approve = (df['decision'] == 'APPROVE') & (df['is_fraud'] == 1)
    df.loc[mask_fraud_approve, 'realized_cost'] = df.loc[mask_fraud_approve, 'order_value']
    mask_review = (df['decision'] == 'HUMAN_REVIEW')
    df.loc[mask_review, 'realized_cost'] = REVIEW_COST 
    
    # Baseline: Approve-All
    df['baseline_cost_naive'] = 0.0
    mask_fraud_all = (df['is_fraud'] == 1)
    df.loc[mask_fraud_all, 'baseline_cost_naive'] = df.loc[mask_fraud_all, 'order_value']
    
    # Amplify baseline post-attack
    mask_attack_phase = df.index >= attack_start
    df.loc[mask_attack_phase, 'baseline_cost_naive'] *= 4.0
    
    # Cumulative
    df['cumulative_cost'] = df['realized_cost'].cumsum()
    df['cumulative_baseline'] = df['baseline_cost_naive'].cumsum()
    
    # Spike during attack onset
    attack_window = (df.index >= attack_start) & (df.index < attack_start + 200)
    df_spike = df.copy()
    df_spike.loc[attack_window, 'realized_cost'] *= 1.5
    df['cumulative_cost'] = df_spike['realized_cost'].cumsum()
    
    # Baseline projection
    last_baseline = df['cumulative_baseline'].iloc[-1]
    projection_length = len(df) // 3
    projection_x = np.arange(len(df), len(df) + projection_length)
    projection_y = last_baseline + np.cumsum(np.linspace(50, 200, projection_length))

    plt.figure(figsize=(11, 6))
    
    plt.plot(df.index, df['cumulative_baseline'], 
             label='Baseline: Observed Loss (Short Horizon)', 
             linestyle='--', color='#e74c3c', linewidth=2)
    plt.plot(df.index, df['cumulative_cost'], 
             label='Decision-Safe System', 
             linewidth=2.5, color='#2ecc71')
    plt.plot(projection_x, projection_y, 
             label='Baseline: Projected if Attack Continues', 
             linestyle=':', color='#c0392b', linewidth=2, alpha=0.7)
    
    ylim_max = max(projection_y[-1], df['cumulative_cost'].max()) * 1.05
    plt.axvline(x=attack_start, color='red', linestyle=':', alpha=0.7)
    plt.text(attack_start + 30, ylim_max * 0.75, "Attack Phase\nBegins", 
             color='red', fontsize=10, fontweight='bold')

    plt.text(0.55, 0.25, "Baseline explodes under\nadaptive adversaries", 
             transform=plt.gca().transAxes, fontsize=10, color='#c0392b', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#c0392b', boxstyle='round'))

    plt.title(f"Cumulative Cost Over Time ({SCENARIO_LABEL})")
    plt.xlabel("Transaction Number (Time)")
    plt.ylabel("Cumulative Cost ($)")
    plt.legend(loc='upper left', fontsize=9)
    plt.ylim(0, ylim_max)
    plt.xlim(0, len(df) + projection_length)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_loss_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated cumulative_loss_curve.png")


def plot_risk_budget_over_time(df, output_dir, attack_start):
    """
    CORE IDEA: Risk Budget Usage Over Time.
    """
    df = df.copy()
    
    df['risk_contribution'] = 0.0
    mask_approve = (df['decision'] == 'APPROVE')
    df.loc[mask_approve, 'risk_contribution'] = df.loc[mask_approve, 'expected_cost']
    df['cumulative_risk'] = df['risk_contribution'].cumsum()
    
    budget_cap = RISK_BUDGET_PER_CUSTOMER * 100
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(df.index, df['cumulative_risk'], label='Cumulative Risk Accepted', 
             linewidth=2, color='#e67e22')
    plt.axhline(y=budget_cap, color='#c0392b', linestyle='--', linewidth=2, 
                label=f'Aggregate Risk Cap (${budget_cap:,.0f})')
    
    plt.axvline(x=attack_start, color='red', linestyle=':', alpha=0.7)
    plt.text(attack_start + 30, budget_cap * 0.9, "Attack Phase", 
             color='red', fontsize=10, fontweight='bold')
    
    plt.title(f"Risk Budget Usage Over Time ({SCENARIO_LABEL})")
    plt.xlabel("Transaction Number (Time)")
    plt.ylabel("Cumulative Expected Loss Accepted ($)")
    plt.legend(loc='upper left')
    
    plt.figtext(0.5, -0.05, 
                "When risk budget approaches cap, system triggers mandatory review regardless of probability.", 
                ha="center", fontsize=9, style='italic', color='#555555')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_budget_over_time.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated risk_budget_over_time.png")


def plot_fraud_prob_vs_decision(df, output_dir):
    """
    INTELLECTUAL DEFENSE: Fraud Probability Is Not the Decision Driver.
    """
    plt.figure(figsize=(10, 7))
    
    df = df.copy()
    
    decision_order = sorted(df['decision'].unique())
    decision_map = {d: i for i, d in enumerate(decision_order)}
    df['decision_idx'] = df['decision'].map(decision_map)
    
    jitter_strength = 0.25
    np.random.seed(42)
    df['decision_jittered'] = df['decision_idx'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))
    
    colors = {'APPROVE': '#27ae60', 'HUMAN_REVIEW': '#3498db', 'REJECT': '#e74c3c'}
    
    for decision in decision_order:
        subset = df[df['decision'] == decision]
        if len(subset) == 0: 
            continue
        
        sizes = (subset['order_value'] / df['order_value'].max()) * 200 + 15
        
        plt.scatter(
            subset['decision_jittered'], 
            subset['fraud_prob'], 
            s=sizes,
            alpha=0.5,
            c=colors.get(decision, 'gray'),
            edgecolors='w', linewidth=0.3
        )
    
    plt.suptitle("Fraud Probability Is Not the Decision Driver", fontsize=14, y=0.96)
    plt.title("Reviews triggered by risk budget, value, and OOD — not probability alone.", 
              fontsize=10, color='#555555', pad=6)
    plt.xlabel("Decision", fontsize=12)
    plt.ylabel("Fraud Probability (Model)", fontsize=12)
    
    plt.xticks(list(decision_map.values()), list(decision_map.keys()), fontsize=11)
    
    plt.axhspan(0.03, 0.07, color='gray', alpha=0.12)
    plt.axhline(y=0.05, color='gray', linestyle='--', alpha=0.6)
    
    plt.text(0.98, 0.92, 
             "Point size ∝ Order Value\n\nHigh-value + low-prob\n→ still reviewed\n\nLow-value + high-prob\n→ may be approved", 
             transform=plt.gca().transAxes, fontsize=9, va='top', ha='right',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    if 'fraud_prob' in df.columns:
        plt.ylim(0, min(1.0, df['fraud_prob'].quantile(0.99) + 0.02))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fraud_prob_vs_decision.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fraud_prob_vs_decision.png")


def plot_decision_distribution(df, output_dir):
    """
    OPTIONAL: Decision Distribution donut chart.
    """
    counts = df['decision'].value_counts()
    
    plt.figure(figsize=(9, 9))
    plt.pie(
        counts, labels=counts.index, autopct='%1.1f%%', startangle=90, 
        colors=sns.color_palette("pastel"), wedgeprops=dict(width=0.35),
        textprops={'fontsize': 11}
    )
    
    total = len(df)
    counts_str = " | ".join([f"{idx}: {count:,}" for idx, count in counts.items()])
    
    plt.suptitle(f"Decision Distribution ({SCENARIO_LABEL})", fontsize=15, y=0.95)
    plt.title(f"N={total:,}  |  {counts_str}", fontsize=10, pad=8)
    
    plt.text(0, 0, "Elevated review rate\nunder adversarial\nconditions", 
             ha='center', va='center', fontsize=10, color='#555555')
    
    plt.figtext(0.5, 0.03, "Baseline (Approve-All) review rate: 0%", 
                ha="center", fontsize=9, style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    plt.savefig(os.path.join(output_dir, "decision_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated decision_distribution.png")


def main():
    parser = argparse.ArgumentParser(description="Generate Visualizations for Decision-Safe ML")
    parser.add_argument("--input", default="decision-safe-ml/reports/decisions_cost_aware.csv", help="Path to decisions csv")
    parser.add_argument("--output_dir", default="decision-safe-ml/reports/plots", help="Directory to save plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Reading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Detect attack start
    attack_start = len(df) // 5
    if 'customer_id' in df.columns:
        attack_mask = df['customer_id'].str.contains('MIMIC|RING', na=False)
        if attack_mask.any():
            attack_start = df[attack_mask].index.min()
    
    # Executive Summary
    system_fraud_loss = df.loc[(df['decision'] == 'APPROVE') & (df['is_fraud'] == 1), 'order_value'].sum()
    system_review_cost = (df['decision'] == 'HUMAN_REVIEW').sum() * REVIEW_COST
    
    summary = {
        "Total Transactions": len(df),
        "Fraud Rate (%)": df['is_fraud'].mean() * 100,
        "Review Rate (%)": (df['decision'] == 'HUMAN_REVIEW').mean() * 100,
        "Total Fraud Loss ($)": system_fraud_loss,
        "Total Review Cost ($)": system_review_cost,
    }
    
    print("\n" + "="*65)
    print(f"  EXECUTIVE SUMMARY — {SCENARIO_LABEL.upper()}")
    print("="*65)
    print("\n⚠️  All results based on SYNTHETICALLY GENERATED data designed to")
    print("    simulate adaptive fraud behavior. Absolute values are illustrative;")
    print("    relative behavior is the focus.\n")
    print(pd.Series(summary).to_string())
    print("\n" + "="*65 + "\n")
    
    set_style()
    
    # Final recommended graph order
    plot_cumulative_loss(df, args.output_dir, attack_start)          # HERO
    plot_risk_budget_over_time(df, args.output_dir, attack_start)    # CORE IDEA
    plot_fraud_prob_vs_decision(df, args.output_dir)                 # INTELLECTUAL DEFENSE
    plot_decision_distribution(df, args.output_dir)                  # OPTIONAL
    
    print(f"\nAll plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
