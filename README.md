# RetainNet

**Continuous-Time Survival Analysis for Customer Churn & Dynamic Lifetime Value Estimation**

RetainNet is a Python framework developed to model customer retention over continuous time rather than treating churn as a fixed-window binary classification problem. In subscription and recurring-revenue businesses, active customers at the end of an observation window are right-censored—we know how long they have stayed so far, not when they will eventually leave. RetainNet handles right-censored records directly, estimates individualized survival probability curves $S(t \mid \mathbf{x})$ across a multi-month horizon, and integrates those curves to compute discounted Customer Lifetime Value (CLTV).

## Core Capabilities

- **3-Stage Hybrid Pipeline**:
  1. **Stage 1 (`retainnet/pipeline/stage1_segmentation.py`)**: Standardizes numeric customer attributes and fits K-Means ($k=4$) to group subscribers into behavioral segments.
  2. **Stage 2 (`retainnet/pipeline/stage2_filtering.py`)**: Trains a balanced L2-regularized logistic regression classifier on the training split and removes ambiguous boundary records ($0.30 \le \hat{p}_i \le 0.70$) before survival training. Held-out validation and test splits are never filtered.
  3. **Stage 3 (`retainnet/models/`)**: Fits survival models on the denoised training data, including PyTorch **DeepSurv** (`deep_surv.py`), discrete-time **DeepHit** (`deep_hit.py`), **Cox Proportional Hazards**, and **Random Survival Forest** (`baselines.py`).
- **Dynamic CLTV Engine (`retainnet/ltv/calculator.py`)**: Computes expected discounted lifetime value via trapezoidal integration over each subscriber's survival curve, accounting for monthly revenue (ARPU), monthly discount rate $d$, and Customer Acquisition Cost (CAC).
- **Universal CSV Ingestion (`retainnet/data/validator.py`)**: Validates uploaded CSV files up to 25 MB, cleans currency symbols and thousand separators, neutralizes spreadsheet formula injection characters (`=`, `+`, `-`, `@`), and maps Telco, Retail Sales, E-Commerce, and SaaS column schemas.
- **Evaluation & Telemetry (`retainnet/metrics/` & `retainnet/advanced/`)**: Includes Harrell's Concordance Index (C-Index), IPCW Brier Scores, Integrated Brier Score (IBS), D-Calibration, censoring-aware conformal prediction bounds (`ConformalSurvivalCalibrator`), population drift detection via PSI and Kolmogorov-Smirnov statistics (`DriftDetector`), and empirical cluster baseline priors for cold-start signups (`AmortizedSurvivalInference`).
- **Interactive Dashboard (`dashboard.py`)**: A six-tab Streamlit interface for exploring portfolio KPIs, model benchmarks, individual customer survival curves with 90% conformal bands, a counterfactual retention scenario simulator, drift metrics, and CSV prediction exports.

## Project Structure

```text
RetainNet/
├── dashboard.py
├── main.py
├── requirements.txt
├── README.md
├── RETAINNET_FULL_PROJECT_DOCUMENTATION.txt
├── retainnet/
│   ├── advanced/
│   │   ├── amortized_inference.py
│   │   ├── conformal_calibration.py
│   │   └── drift_detector.py
│   ├── data/
│   │   ├── generator.py
│   │   ├── pipeline.py
│   │   ├── telco_data.py
│   │   └── validator.py
│   ├── ltv/
│   │   └── calculator.py
│   ├── metrics/
│   │   └── evaluation.py
│   ├── models/
│   │   ├── baselines.py
│   │   ├── deep_hit.py
│   │   └── deep_surv.py
│   └── pipeline/
│       ├── hybrid_pipeline.py
│       ├── stage1_segmentation.py
│       └── stage2_filtering.py
└── tests/
    ├── test_audit_verification.py
    ├── test_dataset_switching.py
    ├── test_evaluation.py
    ├── test_final_verification.py
    ├── test_models.py
    ├── test_pipeline.py
    └── test_security_robustness.py
```

## Setup and Installation

Python 3.10 to 3.13 is supported. Create a virtual environment and install the required packages from `requirements.txt`:

```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### 1. Launch the Interactive Dashboard

```bash
python -m streamlit run dashboard.py
```

Open the local URL displayed in the terminal (default `http://localhost:8501`) to switch between the IBM Telco Benchmark cohort, a configurable Synthetic Weibull cohort, or your own uploaded CSV file.

### 2. Run the Command-Line Pipeline

```bash
# Train and evaluate all models on a synthetic Weibull cohort
python main.py --dataset synthetic --model all --epochs 50 --n-samples 1000

# Run with Stage 2 ablation (compares filtered vs. unfiltered training on the same test set)
python main.py --dataset synthetic --model all --epochs 20 --n-samples 500 --stage2-ablation

# Run on a Telco or custom CSV while dropping cumulative charge proxies
python main.py --dataset telco --telco-path data/WA_Fn-UseC_-Telco-Customer-Churn.csv --model all --drop-total-charges
```

### 3. Run the Automated Test Suite

```bash
python -m unittest discover -s tests -v
# or using pytest:
python -m pytest -q
```

All 119 unit and integration tests across the 7 test files run offline and verify data preprocessing, survival loss gradients, baseline hazard interpolation, CLTV integration, CSV schema validation, and dashboard state consistency.

## Practical Notes & Scope

- **Censoring and Collinearity**: In raw Telco datasets, cumulative `TotalCharges` closely tracks `MonthlyCharges * tenure`. Passing `--drop-total-charges` removes `TotalCharges` and `AvgMonthlyCharge` so models do not rely on a duration proxy.
- **Scenario Simulator**: The What-If Retention Simulator in Tab 4 applies a proportional hazard multiplier $\gamma$ to explore how discounts, annual contract lock-ins, or concierge support could shift a subscriber's survival curve and net LTV. These outputs are sensitivity projections rather than causal estimates from randomized trials.
- **Optional Dependencies**: If `scikit-survival` is not installed in the active environment, `RandomSurvivalForestBaselineModel` automatically runs in a constant-hazard baseline fallback mode (`RSF (Fallback)`) and logs a warning so benchmark tables remain transparent.
