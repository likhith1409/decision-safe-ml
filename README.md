# Decision-Safe ML: When Models Must Refuse to Predict

> **"It is better to abstain than to predict wrong."**

Welcome to the **Decision-Safe ML System**. This project is a production-grade machine learning system designed to handle high-stakes decision-making, specifically for **E-commerce Returns Fraud**. 

Unlike traditional ML models that blindly classify every input, this system implements **Safety, Economic Rationality, and Budget-Aware Control**. It knows when to say "I don't know" and hand off to a human, and it knows when to say "Stop" to prevent financial bleeding.

---

## 1. Project Overview: What is this?

This is a complete end-to-end framework for building an **Economically Safe AI**. 

In real-world business, `Accuracy` is not the most important metric. **`Profit`** (or Cost Minimization) is. A model that is 99% accurate but approves a single $1M fraudulent transaction is a failure. 

**This project solves three critical problems:**
1.  **Ambiguity**: When data is weird (Out-of-Distribution), the model should **Abstain** (ask for human review) rather than guess.
2.  **Cost Asymmetry**: Approving fraud costs real money (loss of item). Rejecting a good customer costs "opportunity" (insult). These costs are not equal. Our system optimizes for **Lowest Expected Cost**.
3.  **Aggregate Risk (The "Slow Bleed")**: Sophisticated attackers use "Mimicry" (looking like normal users) to stay under the radar. Single-transaction models miss this. Our **Risk Budget Engine** tracks cumulative loss per entity and shuts them down when they cross a dynamic safety threshold.

---

## 2. How it Started (The Journey)

The project evolved through three distinct phases of maturity:

*   **Phase 1: The Baseline (Naive ML)**
    *   We started with a standard **Random Forest Classifier**.
    *   It worked for obvious fraud but failed when fraudsters changed tactics. It had no concept of "Cost" or "Safety".
*   **Phase 2: Economic Safety (The "Rational" Phase)**
    *   We introduced **Cost-Aware Abstention**.
    *   Instead of `predict()`, we calculated `Expected_Cost(Approve)` vs `Expected_Cost(Reject)`.
    *   We added an **Isolation Forest** (OOD Detector) to catch anomalies that the fraud model had never seen.
*   **Phase 3: Aggregate Risk Control (The "System" Phase)**
    *   We realized "Mimicry Attacks" (fraudsters behaving "normally" but in high volume) could bypass Phase 2.
    *   We built a **Stateful Risk Engine** that tracks `Cumulative Loss` per customer.
    *   Even if a transaction looks safe, if the customer has "bled" us for $50 already, we force a STOP.

---

## 3. Technologies Used

*   **Python 3.x**: Core language to build the backend logic.
*   **Pandas & NumPy**: For extensive data manipulation and simulation.
*   **Scikit-Learn**:
    *   `RandomForestClassifier`: The core fraud detection brain.
    *   `IsolationForest`: The "Safety Valve" for anomaly detection.
    *   `CalibratedClassifierCV`: Critical for converting raw scores into true probabilities.
*   **Joblib**: For model persistence (saving/loading brains).
*   **Seaborn & Matplotlib**: For generating professional safety dashboards.

---

## 4. Codebase Deep Dive: Each Code Explained

Here is a detailed breakdown of every file in the `src/` directory, so you understand exactly how the engine works.

### 🏗️ The Simulation Layer
**`src/generate_data.py`** (The "World Builder")
*   **What it does**: Creates realistic synthetic data to train and test the system.
*   **Key Logic**:
    *   Generates customers, orders, and products via `generate_base_data`.
    *   **`inject_fraud`**: Simulates specific attack vectors:
        *   *High Value Electronics Fraud*
        *   *Serial Returners*
        *   *Gift Card Money Laundering*
    *   **`apply_drift`**: The most powerful part. It creates "Future Data" that is different from training data.
        *   **Severe Drift (Phase 3)**: Injects a **Mimicry Attack Ring** (20 accounts doing 200 low-value, high-velocity returns) specifically to test the Budget Engine.

### 🧠 The Intelligence Layer
**`src/train.py`** (The "Brain")
*   **What it does**: Trains the Fraud Detection Model.
*   **Key Logic**:
    *   Uses a **Random Forest** to learn patterns.
    *   **CRITICAL**: Wraps the model in `CalibratedClassifierCV`. Why? Because a raw score of "0.7" from a Random Forest doesn't mean "70% probability". Calibration fixes this so we can do math with the output.

**`src/ood.py`** (The "Intuition")
*   **What it does**: Trains the **Out-of-Distribution (OOD)** Detector.
*   **Key Logic**:
    *   Uses **Isolation Forest**.
    *   It learns what "Normal" looks like.
    *   If a new transaction comes in that is "Weird" (e.g., a payment method we've never seen, or a strange combination of values), this model flags it.
    *   **Role**: If `ood_score` is low, we **Abstain** immediately. Don't let the Fraud model guess on things it doesn't understand.

**`src/predict.py`** (The "Worker")
*   **What it does**: A simple utility to load the model and run predictions on new CSV files. Used for batch processing.

### 🛡️ The Control Layer (The Core Innovation)
**`src/abstention.py`** (The "CFO" & "Risk Manager")
*   **What it does**: The central decision-making engine. It joins the Fraud Model, OOD Model, and Economic Rules.
*   **Classes**:
    *   **`RiskEngine`**: A stateful tracker. It remembers "How much money has Customer X cost us?". If they exceed `$50` (Budget), it overrides any "Approval" and forces a Review. **This stops the "Slow Bleed" attacks.**
    *   **`CostAwarePolicy`**:
        1.  **Safety Check**: Is it OOD? -> `HUMAN_REVIEW`.
        2.  **Economic Calculation**: 
            *   `Cost(Approve) = Prob(Fraud) * Order_Value`
            *   `Cost(Reject) = Prob(Legit) * $200 (Insult Cost)`
            *   `Cost(Review) = $50`
            *   **Decision**: Pick the action with the **Minimum Expected Cost**.
        3.  **Budget Check**: Ask `RiskEngine`: "Can we afford this risk?".

### 📊 The Analysis Layer
**`src/monitor.py`** (The "Watchdog")
*   **What it does**: Checks for **Data Drift**.
*   **Key Logic**: Calculates **PSI (Population Stability Index)**. It compares Training Data vs. Live Data. If the data definition changes (drift), it alerts you that the model might be stale.

**`src/failure_analysis.py`** (The "Auditor")
*   **What it does**: Looks at the mistakes ("Leakage").
*   **Key Logic**:
    *   Finds cases where we said `APPROVE` but it was actually `FRAUD`.
    *   Calculates the **Realized Loss**.
    *   Generates a detailed Markdown report (`failure_cases.md`) explaining *why* we missed them (e.g., "It was a mimicry attack that stayed under the radar").

**`src/visualize_results.py`** (The "Storyteller")
*   **What it does**: Generates beautiful plots to prove the system works.
*   **Key Plots**:
    *   **Cumulative Loss Curve**: Shows how the system "flattens the curve" of financial loss compared to a baseline.
    *   **Decision Distribution**: A donut chart showing how often we Approve/Reject/Review.
    *   **Cost Efficiency**: Stacked bar chart showing the trade-off between "Fraud Loss" and "Review Cost".

---

## 5. End-to-End Usage Guide: How to Run It

Follow this sequence to see the entire lifecycle of the project.

### Step 1: Generate the Data
Create the "Past" (Training Data) and the "Future" (Drifted Data with Attacks).
```bash
python decision-safe-ml/src/generate_data.py
```
*Output: `data/processed/returns_normal.csv`, `data/drifted/returns_severe_drift.csv`*

### Step 2: Train the Intelligence
Train the Baseline Fraud Model and the Safety (OOD) Model.
```bash
python decision-safe-ml/src/train.py
python decision-safe-ml/src/ood.py
```
*Output: `models/rf_baseline.joblib`, `models/ood_iforest.joblib`*

### Step 3: Run the Decision Engine (The "Main Event")
Run the cost-aware engine on the "Severe Drift" dataset (which contains the Mimicry Attack).
```bash
python decision-safe-ml/src/abstention.py decision-safe-ml/data/drifted/returns_severe_drift.csv
```
*Output: `reports/decisions_cost_aware.csv`*
*Check the CLI output to see the "Leakage Count" and "Total Realized Cost".*

### Step 4: Audit the Failures
Perform a post-mortem to see what slipped through.
```bash
python decision-safe-ml/src/failure_analysis.py --decisions decision-safe-ml/reports/decisions_cost_aware.csv
```
*Output: `reports/failure_cases.md`*

### Step 5: Visualize the Victory
Generate the dashboard to visualize the economic savings.
```bash
python decision-safe-ml/src/visualize_results.py
```
*Output: Plots in `reports/plots/` (Cumulative Loss, Decision Distribution, etc.)*

---

## 6. Conclusion

This project demonstrates that **AI is not just about prediction**. It is about **Decision Making**. 

By adding an **Economic Layer** (Cost Minimization) and a **Control Layer** (Risk Budgets) on top of standard ML, we turn a "stupid" predictor into a **Decision-Safe System** that protects the business bottom line.
