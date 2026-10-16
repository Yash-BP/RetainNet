# RetainNet: Deep Survival Analysis Framework for Dynamic Customer Lifetime Value & Churn Mitigation

[![Python 3.10-3.13](https://img.shields.io/badge/Python-3.10--3.13-blue.svg)](https://python.org)
[![Tests Passing](https://img.shields.io/badge/Tests-91%20Passed-brightgreen.svg)](tests/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b.svg)](https://streamlit.io)

RetainNet is a clinical-grade deep survival analysis and customer retention engine. Rather than framing customer churn as a static binary classification ($y \in \{0, 1\}$ within an arbitrary 30-day window), RetainNet formulates churn as a **continuous-time-to-event duration process under right-censoring**:
$$S(t \mid X) = P(T > t \mid X) = \exp\left(-\int_0^t h(u \mid X) du\right)$$

This allows continuous integration of survival probabilities into dynamic, individualized Customer Lifetime Value (CLTV):
$$\text{CLTV}_i(H) = \int_0^H \frac{\text{ARPU}_i \cdot S_i(t)}{(1 + d)^t} \, dt - \text{CAC}_i$$

---

## 1. Quick Start Guide (Windows PowerShell)

### Prerequisites
- Python 3.10 to 3.13 installed.
- PowerShell 5.1+ or PowerShell Core 7+.

### Installation
```powershell
# 1. Create and activate a fresh virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Upgrade pip and install core dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Running the Test Suite
All 91 unit, integration, invariant, and security tests can be verified using:
```powershell
python -m pytest -q
```
*Expected Result:* `91 passed, 2 warnings in ~12s (OK)`

### Running the Command Line Interface (CLI)
Run the full 3-stage pipeline with all survival models on synthetic benchmark data:
```powershell
python main.py --dataset synthetic --model all --epochs 50 --n-samples 1000
```
Run with Stage 2 filtering ablation diagnostics:
```powershell
python main.py --dataset synthetic --model all --epochs 20 --n-samples 500 --stage2-ablation
```
Run with TotalCharges duration-proxy leakage prevention on real IBM Telco CSV:
```powershell
python main.py --dataset telco --telco-path data/WA_Fn-UseC_-Telco-Customer-Churn.csv --model all --drop-total-charges
```

### Launching the Enterprise Dashboard
Launch the interactive 6-tab Streamlit decision-support application:
```powershell
python -m streamlit run dashboard.py
```
Open your browser to: **`http://localhost:8501`**

---

## 2. Core Architecture: The 3-Stage Denoising Funnel

```
Raw Customer Features (Demographics, Subscriptions, Billing, Support)
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Behavioral Segmentation (Unsupervised KMeans) │
│ - Clusters heterogeneous subscriber behavior           │
│ - Appends one-hot cluster encodings into feature space │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Censoring-Aware Instance Filtering (LR)       │
│ - Eliminates non-informative / boundary noise          │
│ - Downweights early-censored instances proportional    │
│   to observed follow-up (T / max_T)                    │
│ - Evaluated on TRAIN set only; TEST set is 100% intact │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Deep Survival Analysis Ensembles              │
│ - DeepSurv: Non-linear Cox PH with Breslow ties        │
│ - DeepHit: Unconstrained discrete-time PMF network     │
│ - CoxPH & Random Survival Forest (RSF) baselines       │
│ - SurvivalPFN: Amortized zero-shot baseline inference  │
│ - CiPOT: Conformal 90% distribution-free bands         │
│ - DriftLens: PSI & Kolmogorov-Smirnov drift detection  │
└────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
RetainNet/
├── dashboard.py                  # Enterprise Streamlit application (Protected baseline)
├── main.py                       # CLI experiment runner & ablation engine
├── requirements.txt              # Core runtime dependencies
├── RETAINNET_FULL_PROJECT_DOCUMENTATION.txt  # Complete technical & scientific manual
├── README.md                     # Setup instructions & reproduction guide
├── retainnet/                    # Core library package
│   ├── data/                     # Data ingestion, synthetic generator & pipeline
│   ├── models/                   # DeepSurv, DeepHit, CoxPH, and RSF models
│   ├── pipeline/                 # 3-Stage segmentation, filtering, and hybrid pipeline
│   ├── metrics/                  # Harrell's C-Index, IPCW Brier, IBS, D-Calibration
│   ├── ltv/                      # Continuous quadrature dynamic CLTV engine
│   └── advanced/                 # SurvivalPFN, CiPOT conformal bands, DriftLens
└── tests/                        # 91 unit, integration, and security tests
    ├── test_pipeline.py          # Pipeline and data leakage tests
    ├── test_evaluation.py        # Mathematical metric validation tests
    ├── test_models.py            # DeepSurv, DeepHit, tied events, zero events
    ├── test_final_verification.py# Complete milestone audit suite
    └── test_security_robustness.py# CSV injection, NaN imputation, 1D tensor tests
```

---

## 4. Protected Asset Integrity

To verify baseline reproducibility:
```powershell
# 1. Verify dashboard.py is byte-for-byte identical to baseline commit 7963fc7:
git diff 7963fc7 -- dashboard.py
# (Output must be empty, exit code 0)

# 2. Verify protected archive RetainNet_clean.zip SHA-256 hash:
Get-FileHash D:\downloads\RetainNet_clean.zip -Algorithm SHA256
# Expected Hash: 1F774CA02FF1C2A1219410BD0053DE167430F62D49E28AD2B134C3CC2C801DEC
```

---

## 5. Scientific Methodology & Limitations

1. **Independent Right-Censoring:** Survival estimation assumes that censoring time $C$ is conditionally independent of survival time $T$ given features $X$.
2. **Proportional Hazards Assumption:** DeepSurv assumes constant hazard ratios over time. For non-proportional or crossing hazard curves, DeepHit's discrete-time PMF is the recommended specification.
3. **What-If Heuristic:** The interactive What-If Simulator in `dashboard.py` models retention interventions via parametric hazard modifiers ($\gamma = \exp(-\beta \cdot \Delta x)$). Results represent sensitivity scenarios rather than proven causal treatment effects.
