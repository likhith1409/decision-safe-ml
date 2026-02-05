import pandas as pd
import numpy as np
import joblib
import sys
import os

class RiskEngine:
    def __init__(self):
        # RISK BUDGETS
        self.BUDGET_CUSTOMER = 50.0  # Max cumulative expected leakage before stopping
        self.BUDGET_PAYMENT = 2000.0 # Max cumulative leakage per payment type
        
        # State: Cumulative Loss Trackers
        self.loss_customer = {} 
        self.loss_payment = {}
        
    def check_and_update(self, customer_id, payment_method, expected_loss):
        """
        Updates cumulative loss and returns decision override if budget exceeded.
        Returns: override_decision (or None), reason (or None)
        """
        # Initialize if new
        if customer_id not in self.loss_customer: self.loss_customer[customer_id] = 0.0
        if payment_method not in self.loss_payment: self.loss_payment[payment_method] = 0.0
        
        # Check Budgets
        cust_loss = self.loss_customer[customer_id]
        pay_loss = self.loss_payment[payment_method]
        
        # Flag if budgets exceeded
        budget_breach = False
        breach_reason = ""
        
        if cust_loss + expected_loss > self.BUDGET_CUSTOMER:
            budget_breach = True
            breach_reason = f"Customer Budget Exceeded (${cust_loss:.2f})"
        
        # (Optional: Payment budget might be too aggressive for this demo, 
        # but let's keep it to show we track it)
        # elif pay_loss + expected_loss > self.BUDGET_PAYMENT:
        #    budget_breach = True
        #    breach_reason = f"Payment Budget Exceeded (${pay_loss:.2f})"
            
        # Update State (We accumulate risk even if we block it, roughly speaking, 
        # or we count the attempted risk. Let's count attempted risk to keep them blocked.)
        self.loss_customer[customer_id] += expected_loss
        self.loss_payment[payment_method] += expected_loss
        
        if budget_breach:
            return 'HUMAN_REVIEW', breach_reason
        return None, None

class CostAwarePolicy:
    def __init__(self, fraud_model_path, ood_model_path):
        print(f"Loading Fraud Model: {fraud_model_path}")
        self.fraud_model = joblib.load(fraud_model_path)
        
        print(f"Loading OOD Model: {ood_model_path}")
        self.ood_model = joblib.load(ood_model_path)
        
        # COST CONSTANTS (The "Economic Layer")
        self.COST_FALSE_REJECT = 200.0  
        self.COST_HUMAN_REVIEW = 50.0   
        
        # RISK ENGINE (The "Control Layer")
        self.risk_engine = RiskEngine()
        
    def predict(self, data_path):
        df = pd.read_csv(data_path)
        X = df.drop(columns=['is_fraud', 'order_id', 'customer_id'], errors='ignore')
        
        # 1. Get Fraud Probability
        probs = self.fraud_model.predict_proba(X)[:, 1]
        
        # 2. Get OOD Prediction
        ood_scores = self.ood_model.decision_function(X)
        is_inlier = self.ood_model.predict(X) 
        
        decisions = []
        reasons = []
        expected_costs = []
        
        # Iterate row by row (Simulating Stream for State Updates)
        for i, (prob_fraud, ood_status, order_value, cust_id, pay_method) in enumerate(zip(
            probs, is_inlier, df['order_value'], df['customer_id'], df['payment_method']
        )):
            
            # --- 3. SAFETY LAYER: Out-of-Distribution ---
            if ood_status == -1:
                decisions.append('HUMAN_REVIEW')
                reasons.append(f'OOD (Score: {ood_scores[i]:.3f})')
                expected_costs.append(self.COST_HUMAN_REVIEW)
                continue

            # --- 4. OPTIMIZATION LAYER: Cost Minimization ---
            cost_approve = prob_fraud * order_value
            cost_reject = (1.0 - prob_fraud) * self.COST_FALSE_REJECT
            cost_review = self.COST_HUMAN_REVIEW
            
            costs = {'APPROVE': cost_approve, 'REJECT': cost_reject, 'HUMAN_REVIEW': cost_review}
            best_action = min(costs, key=costs.get)
            min_cost = costs[best_action]
            
            # --- 5. CONTROL LAYER: Aggregate Risk Budget ---
            # Even if "APPROVE" is locally optimal, is this entity bleeding us?
            # We track the 'Risk' we are taking. If we Approve, risk is cost_approve.
            # If we Review/Reject, we are mitigating fraud risk, so we update state differently?
            # Actually, standard practice: Track 'Detected/Suspected Loss'.
            # Here, we track 'Cumulative Expected Leakage' for Approvals.
            
            risk_to_add = 0.0
            if best_action == 'APPROVE':
                risk_to_add = cost_approve
            
            # Check Budget
            override, override_reason = self.risk_engine.check_and_update(cust_id, pay_method, risk_to_add)
            
            if override and best_action == 'APPROVE':
                # Budget says NO. Force Review.
                best_action = override
                min_cost = self.COST_HUMAN_REVIEW # We pay for review now
                reasons.append(f"BUDGET_HIT: {override_reason}")
            else:
                # Standard decision
                reasons.append(f"L={cost_approve:.0f}|R={cost_reject:.0f}|H={cost_review:.0f}")

            decisions.append(best_action)
            expected_costs.append(min_cost)

        
        # Build Result DF
        results = df[['order_id', 'customer_id', 'order_value']].copy()
        if 'is_fraud' in df.columns:
            results['is_fraud'] = df['is_fraud']
            
        results['fraud_prob'] = probs
        results['ood_score'] = ood_scores
        results['decision'] = decisions
        results['reason'] = reasons
        results['expected_cost'] = expected_costs
        
        return results

def run_decision_system(data_path, output_path):
    policy = CostAwarePolicy(
        fraud_model_path="decision-safe-ml/models/rf_baseline.joblib",
        ood_model_path="decision-safe-ml/models/ood_iforest.joblib"
    )
    
    results = policy.predict(data_path)
    
    # Summary Metrics
    print("\n--- ECONOMIC SUMMARY ---")
    print(results['decision'].value_counts(normalize=True))
    
    total_expected_cost = results['expected_cost'].sum()
    print(f"\nTotal Expected Liability: ${total_expected_cost:,.2f}")
    
    if 'is_fraud' in results.columns:
        # REALIZED COST (The "True" Cost)
        # Cost = 
        #   If Approve & Fraud: Order Value
        #   If Reject & Legit: 200
        #   If Review: 50
        
        realized_cost = 0
        for _, row in results.iterrows():
            if row['decision'] == 'HUMAN_REVIEW':
                realized_cost += 50
            elif row['decision'] == 'APPROVE':
                if row['is_fraud'] == 1:
                    realized_cost += row['order_value']
            elif row['decision'] == 'REJECT':
                if row['is_fraud'] == 0:
                    realized_cost += 200
                    
        print(f"Total REALIZED Cost: ${realized_cost:,.2f}")
        print(f"Cost per Decision: ${realized_cost/len(results):.2f}")
        
        # Safety Metrics
        leakage_count = len(results[(results['decision'] == 'APPROVE') & (results['is_fraud'] == 1)])
        print(f"Leakage Count (Fraud Approved): {leakage_count}")

    if output_path:
        results.to_csv(output_path, index=False)
        print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--output", default="decision-safe-ml/reports/decisions_cost_aware.csv")
    args = parser.parse_args()
    
    run_decision_system(args.data_path, args.output)
