# RetainNet

Deep survival analysis and individualized customer lifetime value (CLTV) framework for churn risk quantification and scenario planning.

---

## 1. Overview & 3-Stage Methodology

Standard classification models output static binary churn probabilities over arbitrary time windows (for example, will a customer churn within 30 days). In contrast, RetainNet uses continuous-time and discrete-time survival analysis to model the full survival curve $S(t | x) = P(T > t | x)$ for each customer across a multi-year horizon.

To account for behavioral heterogeneity and noisy records, RetainNet employs a 3-stage hybrid architecture:

```
[Raw Feature Matrix]
        │
        ▼
Stage 1: Behavioral Segmentation
  - Unsupervised K-Means clustering (k=4 archetypes)
  - Identifies distinct behavioral groups based on tenure, ARPU, support tickets, and service adoption
        │
        ▼
Stage 2: Propensity Filtering (Training Isolation)
  - Evaluates cluster assignment confidence
  - Applies filtering only to training data to dampen ambiguous boundary points
  - Preserves 100% of validation and test sets for unbiased evaluation
        │
        ▼
Stage 3: Multi-Model Survival Modeling
  - Trains semi-parametric (CoxPH), ensemble (Random Survival Forest), and deep neural models (DeepSurv, DeepHit)
  - Integrates survival probabilities over time to compute individualized Dynamic CLTV
```

---

## 2. Architecture Explanation & Data Flow

The project consists of a core modeling library (`retainnet`), a command-line interface (`main.py`), and a Streamlit analytics dashboard (`dashboard.py`).

### Data Flow

1. **Ingestion & Validation (`retainnet.data.validator`)**:
   - Validates CSV structure, column types, and value bounds.
   - Maps arbitrary headers (Telco, SaaS, Retail, E-Commerce) to canonical columns via alias normalization.
   - Sanitizes text and identifier fields against spreadsheet formula injection (`=`, `+`, `-`, `@`, `|`).
2. **Preprocessing & Feature Engineering (`retainnet.data.pipeline`)**:
   - Imputes missing numerical values using cohort medians and categoricals using cohort modes.
   - Computes tenure-scaled ratios and service usage density.
   - Standardizes numerical features using training-set statistics.
3. **Stage 1 Segmentation (`retainnet.pipeline.stage1_segmentation`)**:
   - Fits K-Means clustering on scaled training features.
   - Appends cluster distance features and archetypes to downstream stages.
4. **Stage 2 Filtering (`retainnet.pipeline.stage2_filtering`)**:
   - Computes silhouette or distance-based confidence scores for training records.
   - Retains representative records while leaving test sets completely untouched.
5. **Stage 3 Survival Models (`retainnet.models`)**:
   - Fits Cox Proportional Hazards, Random Survival Forest, DeepSurv (PyTorch negative log-likelihood), and DeepHit (joint discrete hazard and ranking loss).
   - Generates Individualized Survival Distributions (ISD) across a standardized time grid.
6. **Dynamic CLTV Engine (`retainnet.ltv.calculator`)**:
   - Calculates customer lifetime value by integrating survival curves weighted by monthly recurring charges and discount rates:
     $$\text{CLTV}_i = \int_0^H S_i(t) \cdot \text{ARPU}_i \cdot e^{-\delta t} \, dt - \text{CAC}_i$$
7. **What-If Scenario Simulator (`dashboard.py`)**:
   - A scenario-planning tool based on proportional-hazard sensitivity multipliers ($\gamma$).
   - Simulates policy impacts (discounts, lock-in terms, VIP concierge) on hazard decay and projected retained revenue.
   - Explicitly designed for heuristic planning rather than randomized causal uplift estimation.

---

## 3. Data Dictionary & Supported Column Aliases

### Canonical Schema

| Column Name | Type | Description | Constraints |
| :--- | :--- | :--- | :--- |
| `customer_id` | String | Unique subscriber or account identifier | Sanitized against formula injection |
| `duration` | Float | Observed tenure or time-to-event in months | $\ge 0.0$ |
| `event` | Integer | Binary event indicator (1 = Churned, 0 = Censored/Active) | $\{0, 1\}$ |
| `monthly_charges` | Float | Monthly recurring charge or average order value ($/month) | $\ge 0.0$ |
| `total_charges` | Float | Cumulative historical spend ($) | $\ge 0.0$ |
| `contract_type` | Integer / String | Contract commitment tier (0 = Month-to-Month, 1 = One Year, 2 = Two Year) | Maps to $\{0, 1, 2\}$ |
| `support_tickets` | Integer | Total customer service or technical support interactions | $\ge 0$ |
| `tech_support` | Integer | Indicator of active tech support service | $\{0, 1\}$ |
| `online_security` | Integer | Indicator of active security add-on | $\{0, 1\}$ |
| `paperless_billing`| Integer | Indicator of paperless billing adoption | $\{0, 1\}$ |

### Supported Column Aliases

The ingestion validator automatically maps common domain variants into the canonical schema:

- **Customer ID**: `customerid`, `subscriber_id`, `user_id`, `account_id`, `client_id`, `store_id`, `order_id`, `CustomerID`
- **Duration**: `tenure`, `tenure_months`, `months`, `days`, `account_age`, `period`, `TenureMonths`
- **Event**: `churn`, `churned`, `status`, `is_churn`, `attrition`, `cancelled`, `positive`/`negative`, `t`/`f`
- **Monthly Charges**: `monthlycharges`, `monthly_fee`, `arpu`, `mrr`, `sales`, `daily_sales`, `revenue`, `amount`, `spend`, `price`, `MonthlyCharges`
- **Total Charges**: `totalcharges`, `total_spend`, `lifetime_spend`, `cumulative_sales`, `arr`, `TotalCharges`
- **Contract Type**: `contract`, `plan_type`, `subscription_tier`, `category`, `tier`, `department`, `ContractType`
- **Support Tickets**: `support_calls`, `tickets`, `customer_service_calls`, `quantity`, `units`, `orders`, `SupportTickets`

---

## 4. Model Card

### DeepSurv (`retainnet.models.deep_surv`)
- **Type**: Deep Feedforward Neural Network for proportional hazards.
- **Architecture**: 3-layer MLP with BatchNorm, Dropout (0.2), ReLU activation, and a single linear risk output $h_\theta(x)$.
- **Loss Function**: Negative Cox partial log-likelihood:
  $$\mathcal{L}(\theta) = -\sum_{i: E_i=1} \left( h_\theta(x_i) - \log \sum_{j \in \mathcal{R}(T_i)} \exp(h_\theta(x_j)) \right)$$
- **Optimization**: AdamW optimizer with weight decay ($10^{-4}$) and gradient clipping.

### DeepHit (`retainnet.models.deep_hit`)
- **Type**: Multi-task discrete-time neural survival network.
- **Architecture**: Shared representation layer with residual connections and softmax output over discrete duration bins.
- **Loss Function**: Combined log-likelihood of failure times and concordance ranking loss:
  $$\mathcal{L} = \mathcal{L}_1 + \alpha \mathcal{L}_2$$
- **Characteristics**: Relaxes proportional hazards assumption by modeling non-constant hazard curves directly.

### Cox Proportional Hazards (`retainnet.models.baselines.CoxPHBaselineModel`)
- **Type**: Semi-parametric proportional hazards model with L2 regularization.
- **Implementation**: Utilizes `lifelines.CoxPHFitter` with numerical Hessian inversion fallback.
- **Baseline Hazard**: Breslow non-parametric estimator for baseline cumulative hazard $H_0(t)$.

### Random Survival Forest (`retainnet.models.baselines.RandomSurvivalForestBaselineModel`)
- **Type**: Non-parametric ensemble of survival trees.
- **Splitting Criterion**: Log-rank test statistic maximizing survival divergence between child nodes.
- **Fallback**: Incorporates a decision tree ensemble estimator if `scikit-survival` native binaries are uncompiled.

---

## 5. Benchmark Results Summary

Evaluated on the IBM Telco benchmark dataset using an unbiased, 100% evaluated held-out test set ($N=500$ test records, $N=2,000$ training records):

| Model | Concordance Index (C-Index) $\uparrow$ | Integrated Brier Score (IBS) $\downarrow$ | Brier Score @ 12M $\downarrow$ | Brier Score @ 24M $\downarrow$ | Brier Score @ 36M $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DeepSurv (PyTorch)** | **0.6111** | **0.1483** | 0.1610 | 0.1742 | 0.1504 |
| **CoxPH Baseline** | 0.6066 | 0.1480 | 0.1582 | 0.1738 | 0.1511 |
| **DeepHit Discrete Engine** | 0.6027 | 0.1569 | 0.1685 | 0.1812 | 0.1601 |
| **Random Survival Forest** | 0.5950 | 0.2123 | 0.2014 | 0.2240 | 0.2115 |

*Note: In compliance with sound evaluation methodology, Stage 2 filtering was restricted strictly to the training split (retaining 91.7% of training signal), ensuring zero data leakage and 0% test set filtering.*

---

## 6. Engineering Decisions Log

1. **Unbiased Test Set Evaluation**:
   - Initial survival pipelines occasionally filtered noisy records across the entire dataset. In RetainNet, Stage 2 filtering is strictly quarantined to training data. The validation and test sets remain 100% complete to provide realistic, unvarnished evaluation metrics.
2. **Two-Tier Disk Cache Architecture**:
   - To achieve sub-5-second startup times on Streamlit (`~3.2s` observed), model objects are separated into two tiers:
     - `.pipeline_cache.pkl` stores precomputed visualization structures, evaluation matrices, and customer summaries for immediate rendering.
     - `.models_cache.pkl` stores heavy PyTorch and scikit-learn model weights, loaded on demand only when a user uploads a new custom dataset.
3. **Formula Injection Sanitization**:
   - User-supplied CSV files and exported customer records are sanitized against spreadsheet formula injection by prepending single quotes to cells beginning with `=`, `+`, `-`, `@`, `|`, or control characters.
4. **Proportional Hazard Scenario Assumptions**:
   - The What-If Simulator scales individualized baseline survival using sensitivity multipliers ($\gamma$):
     $$S^*(t | x) = S(t | x)^\gamma$$
   - This formulation is documented clearly in the UI as a scenario-planning tool based on assumed proportional shifts rather than causal counterfactual inferences.

---

## 7. Demo Script

### Installation

Clone the repository and install dependencies in your virtual environment:

```powershell
pip install -r requirements.txt
```

### Running the Command-Line Pipeline

Train candidate models on synthetic Weibull data or local CSVs:

```powershell
# Run synthetic cohort training with CoxPH baseline
python main.py --dataset synthetic --model coxph --n-samples 1000

# Run full multi-model evaluation
python main.py --dataset synthetic --model all --epochs 100
```

### Launching the Interactive Dashboard

Launch the Streamlit dashboard:

```powershell
python -m streamlit run dashboard.py
```

The application will start and open in your default browser at `http://localhost:8501`.

#### Exploring the 6 Dashboard Tabs

1. **Executive Summary**: View overall subscriber counts, right-censoring rate, aggregate CLTV, at-risk revenue, and the Priority Customer Queue.
2. **Pipeline & Survival Benchmarks**: Inspect Stage 1 cluster archetypes, Stage 2 signal retention, Kaplan-Meier overlays, and model leaderboard metrics.
3. **Customer Retention**: Search individual subscribers by ID or segment, inspect individualized survival curves, confidence intervals, and risk driver breakdowns.
4. **What-If Simulator**: Adjust monthly discounts, 12-month lock-in agreements, and concierge interventions to project hazard reduction and net campaign ROI.
5. **MLOps & Drift Diagnostics**: Monitor Kolmogorov-Smirnov statistics and Population Stability Index (PSI) to detect distribution shifts between baseline and active cohorts.
6. **Audit Reports & CSV Export**: Review detailed model calibration scores and download sanitized CSV predictions for operational follow-up.

### Running the Test Suite

Execute the complete automated test suite (154 tests):

```powershell
python -m pytest -v
```

