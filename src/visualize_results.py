import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np

def set_style():
    """Sets a professional style for the plots."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14

def plot_cumulative_loss(df, output_dir):
    """
    Plots the cumulative realized loss over time (transaction order).
    """
    df = df.copy()
    
    # Calculate costs
    df['realized_cost'] = 0.0
    
    # Cost of Approval (Fraud Loss)
    mask_fraud_approve = (df['decision'] == 'APPROVE') & (df['is_fraud'] == 1)
    df.loc[mask_fraud_approve, 'realized_cost'] = df.loc[mask_fraud_approve, 'order_value']
    
    # Cost of Review
    mask_review = (df['decision'] == 'HUMAN_REVIEW')
    df.loc[mask_review, 'realized_cost'] = 50.0 
    
    df['cumulative_cost'] = df['realized_cost'].cumsum()
    
    # Baseline
    df['baseline_cost'] = 0.0
    mask_fraud_all = (df['is_fraud'] == 1)
    df.loc[mask_fraud_all, 'baseline_cost'] = df.loc[mask_fraud_all, 'order_value']
    df['cumulative_baseline'] = df['baseline_cost'].cumsum()
    
    plt.figure()
    plt.plot(df.index, df['cumulative_baseline'], label='Baseline (Approve All)', linestyle='--', color='gray')
    plt.plot(df.index, df['cumulative_cost'], label='Decision-Safe System', linewidth=2, color='#2ecc71')
    
    # Annotation for adaptive attacks
    plt.axvline(x=1000, color='red', linestyle=':', alpha=0.5) 
    # Use a safe position for text, maybe relative to axes
    plt.text(0.5, 0.5, "Baseline does not model\nadaptive attackers or\ndelayed loss amplification", 
             transform=plt.gca().transAxes, fontsize=10, color='red', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.title("Cumulative Cost Accumulation")
    plt.xlabel("Transaction Number (Time)")
    plt.ylabel("Cumulative Cost ($)")
    plt.legend()
    
    # Footer note
    plt.figtext(0.5, -0.05, "This simulation isolates direct transactional cost; it does not include second-order losses\nsuch as chargeback penalties, account bans, or regulatory exposure.", 
                ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_loss_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated cumulative_loss_curve.png")

def plot_decision_distribution(df, output_dir):
    """Plots the distribution of decisions as a donut chart."""
    counts = df['decision'].value_counts()
    
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, 
            colors=sns.color_palette("pastel"), wedgeprops=dict(width=0.3))
    
    plt.title("Distribution of System Decisions")
    plt.suptitle("High abstention rate reflects uncertainty and active attack conditions.", y=0.92, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "decision_distribution.png"), dpi=300)
    plt.close()
    print("Generated decision_distribution.png")

def plot_cost_efficiency_stacked(df, output_dir):
    """
    Stacked Bar Chart: Fraud Loss vs Operational Cost (Safety).
    Replaces the misleading simple bar chart.
    """
    # 1. Decision-Safe System
    system_fraud_loss = df.loc[(df['decision'] == 'APPROVE') & (df['is_fraud'] == 1), 'order_value'].sum()
    system_review_cost = (df['decision'] == 'HUMAN_REVIEW').sum() * 50.0
    
    # 2. Baseline
    baseline_fraud_loss = df.loc[df['is_fraud'] == 1, 'order_value'].sum()
    baseline_review_cost = 0.0
    
    # Data for plot
    categories = ['Baseline (Approve All)', 'Decision-Safe System']
    fraud_losses = [baseline_fraud_loss, system_fraud_loss]
    review_costs = [baseline_review_cost, system_review_cost]
    
    x = np.arange(len(categories))
    width = 0.5
    
    plt.figure(figsize=(8, 6))
    p1 = plt.bar(x, fraud_losses, width, label='Direct Fraud Loss', color='#e74c3c')
    p2 = plt.bar(x, review_costs, width, bottom=fraud_losses, label='Safety Operations (Review)', color='#3498db')
    
    plt.ylabel('Total Spend ($)')
    plt.title('Cost Trade-off: Automation vs Safety')
    plt.xticks(x, categories)
    plt.legend()
    
    # Values on bars
    for i in range(len(categories)):
        total = fraud_losses[i] + review_costs[i]
        plt.text(i, total + (total*0.02), f"${total:,.0f}", ha='center', fontsize=12, fontweight='bold')
        
    # Mandatory Caption
    caption = (
        "The Decision-Safe system intentionally incurs higher operational cost to prevent catastrophic\n"
        "fraud escalation under attack scenarios. This comparison is not about minimizing spend,\n"
        "but bounding worst-case loss."
    )
    plt.figtext(0.5, -0.1, caption, ha="center", fontsize=10, style='italic', bbox=dict(facecolor='#f0f0f0', edgecolor='none', pad=10))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cost_efficiency_stacked.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated cost_efficiency_stacked.png")

def plot_risk_vs_decision(df, output_dir):
    """OOD Score distribution by Decision."""
    plt.figure()
    sns.violinplot(data=df, x='decision', y='ood_score', palette="muted", inner="quartile")
    plt.title("OOD Score Distribution by Decision")
    plt.xlabel("Decision")
    plt.ylabel("OOD Score (Lower = More Anomalous)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ood_score_vs_decision.png"), dpi=300)
    plt.close()
    print("Generated ood_score_vs_decision.png")
    
    # Fraud Prob vs Decision
    plt.figure()
    sns.stripplot(data=df, x='decision', y='fraud_prob', palette="muted", alpha=0.5, jitter=0.2)
    plt.title("Fraud Probability Model Confidence by Decision")
    plt.xlabel("Decision")
    plt.ylabel("Fraud Probability (Model)")
    
    # Add conceptual threshold line
    plt.axhline(y=0.05, color='gray', linestyle='--')
    plt.text(0.5, 0.055, "Conceptual Review Threshold", color='gray', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fraud_prob_vs_decision.png"), dpi=300)
    plt.close()
    print("Generated fraud_prob_vs_decision.png")


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
    
    set_style()
    
    plot_cumulative_loss(df, args.output_dir)
    plot_decision_distribution(df, args.output_dir)
    plot_cost_efficiency_stacked(df, args.output_dir)
    plot_risk_vs_decision(df, args.output_dir)
    
    print(f"All plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
