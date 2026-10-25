# RetainNet — Complete Engineering Specification Suite

**Project Name:** RetainNet: Deep Survival Analysis Framework for Dynamic Customer Lifetime Value (CLTV) & Proactive Churn Mitigation  
**Version:** 1.0.0 (Production Baseline)  
**Target Codebase:** `retainnet/` Python Package, `main.py` CLI Engine, `dashboard.py` Enterprise Intelligence System, `tests/` (119 Automated Verification Tests)

---

# 1. Product Requirements Document (PRD)

## 1.1 Executive Summary & Project Description
**RetainNet** is an end-to-end survival analysis and revenue intelligence platform that replaces static binary churn classification with continuous-time Individualized Survival Distributions ($S(t \mid \mathbf{x}) = P(T > t \mid \mathbf{x})$). Standard churn classifiers collapse time-to-event trajectories into an arbitrary fixed window, discarding right-censored subscriber tenure and producing miscalibrated Customer Lifetime Value (CLTV) estimates. RetainNet resolves this through a **3-Stage Hybrid Denoising Pipeline**:
1. **Stage 1 — Behavioral Segmentation (`BehavioralSegmentationStage`):** Unsupervised K-Means ($k=4$) clustering groups subscribers into four interpretable behavioral archetypes (*Loyal Long-Term*, *At-Risk Newbies*, *Price-Sensitive*, and *Premium High-ARPU*) and appends one-hot cluster indicators to the covariate matrix.
2. **Stage 2 — Propensity Instance Filtering (`InstanceFilteringStage`):** Class-balanced L2 logistic regression weighted by tenure reliability ($\omega_i = 0.25 + 0.75 \cdot (T_i / T_{\max})$ for right-censored records) excises noisy boundary records ($0.30 \le \hat{p}_i \le 0.70$) strictly from the training split while keeping validation and held-out test sets 100% unfiltered.
3. **Stage 3 — Multi-Architecture Survival & Dynamic CLTV Engine:** Trains and benchmarks four survival models (**DeepSurv PyTorch MLP**, **DeepHit Discrete-Time Neural Engine**, **Regularized Cox Proportional Hazards**, and **Random Survival Forest**), integrates continuous survival curves via $\Delta t$-weighted trapezoidal quadrature to compute net Dynamic CLTV, attaches distribution-free 90% conformal bounds (**CiPOT**), monitors covariate shift (**DriftLens** PSI & KS), and powers an interactive **What-If Counterfactual Retention Simulator**.

## 1.2 Problem Statement
Subscription businesses across telecommunications, SaaS, and retail commerce lose substantial recurring revenue because existing churn analytics tools suffer from four structural blind spots:
1. **Right-Censoring Bias:** Active subscribers whose observation window ends before a churn event ($E_i = 0$) have unknown true lifetimes ($T_i^* > T_i$). Treating censored records as permanent non-churners or dropping them distorts hazard estimation.
2. **Static Horizon Blindness:** A binary probability score at a fixed horizon cannot distinguish a subscriber who will cancel in Month 2 from one who will cancel in Month 11, leading to mistimed retention spend.
3. **Boundary Label Noise:** Subscribers with mid-range propensity scores ($0.30 \le \hat{p}_i \le 0.70$) and short censored tenures introduce gradient variance into deep survival networks.
4. **Disconnected Financial Actionability:** Data science teams output abstract hazard scores without translating survival trajectories into net discounted dollar value (Dynamic CLTV) or quantifying the Return on Investment (ROI) of retention promotions before deployment.

## 1.3 Target Users & Personas

| Persona | Role & Responsibilities | Primary Pain Points | Key RetainNet Workflows |
| :--- | :--- | :--- | :--- |
| **Retention Operations Analyst** | Manages daily churn intervention queues, customer outreach, and contract renewal offers. | Cannot tell which at-risk subscribers are worth saving or which incentive tier maximizes net lift. | Inspects **Tab 1 Priority Customer Queue**, searches subscribers in **Tab 3**, reviews clinical prescriptions, and simulates discounts in **Tab 4**. |
| **VP of Customer Success / CFO** | Oversees portfolio-wide recurring revenue, Customer Acquisition Cost (CAC) recovery, and retention budget allocation. | Static churn rates hide total dollar exposure in the bottom hazard quartile. | Monitors **Tab 1 Portfolio KPIs** (Active Subscribers, Average Dynamic CLTV, At-Risk Revenue, 12M Projected Churn) and exports **Tab 6 Audit CSVs**. |
| **Machine Learning Engineer / Data Scientist** | Trains, validates, and monitors survival models across benchmark and custom production cohorts. | Manual schema cleaning for messy CSV uploads, silent covariate drift, and lack of calibrated uncertainty bands. | Runs `main.py` CLI benchmarks and `--stage2-ablation` studies, inspects **Tab 2 Leaderboard**, monitors **Tab 5 DriftLens PSI/KS**, and ingests custom CSVs. |

## 1.4 Product Goals & Objectives
1. **Temporal Hazard Accuracy:** Achieve Harrell's Concordance Index ($C\text{-Index}$) $> 0.60$ and Integrated Brier Score ($\text{IBS}$) $\le 0.16$ on held-out right-censored telecommunications benchmarks.
2. **Sub-Millisecond Interactive Inference:** Deliver per-subscriber neural survival curve generation in $< 1.0\text{ ms}$ (`0.4 ms` achieved on the 2,500-subscriber IBM Telco benchmark) and instant client-side UI tab/dataset switching in $< 50\text{ ms}$.
3. **Universal Schema Adaptability:** Ingest standard Telco CSVs, Retail Daily Sales CSVs, SaaS subscription logs, or unlabeled inference-only cohorts (up to `25 MB`) with automated alias mapping, currency stripping, and interactive column override controls.
4. **Calibrated Uncertainty & MLOps Governance:** Maintain $\ge 90\%$ empirical coverage on individual survival curves via censoring-aware conformal calibration ($\alpha = 0.10$) and flag feature drift whenever Population Stability Index ($\text{PSI}$) $\ge 0.20$.

## 1.5 Core Features & MVP Scope

### In-Scope MVP Features
- **F1. Multi-Source Cohort Engine:** Pre-loaded **IBM Telco Benchmark** (`2,500` records, default), configurable **Synthetic Weibull Cohort Generator** (`50–10,000` UI / `1–20,000` backend records, custom random seed), and **Custom CSV Upload** with live schema remapping.
- **F2. 3-Stage Hybrid Denoising Pipeline:** K-Means ($k=4$) archetype discovery, tenure-weighted L2 logistic propensity boundary excision ($0.30 \le \hat{p}_i \le 0.70$) on the training split, and full 4-model survival training (`DeepSurv`, `DeepHit`, `CoxPH`, `RSF`).
- **F3. Dynamic CLTV & Portfolio Executive Dashboard (Tab 1):** Real-time calculation of Active Subscribers, Average Dynamic CLTV, At-Risk Revenue (bottom 25% CLTV quartile), 12-Month Projected Churn, 10-bin color-coded CLTV histogram, K-Means archetype donut chart, and top-6 Priority Customer Queue with interactive sort (12M Churn Risk, Dynamic CLTV, Monthly Fee) and segment filtering.
- **F4. Pipeline Architecture & Benchmark Leaderboard (Tab 2):** Visual 3-stage funnel specification, multi-horizon model evaluation table ($C\text{-Index}$, $\text{IBS}$, $\text{BS}@12\text{M}$, $\text{BS}@24\text{M}$, $\text{BS}@36\text{M}$, D-Calibration $\chi^2$), and Empirical Kaplan-Meier vs. DeepSurv overlay plot.
- **F5. Full-Cohort Subscriber Lookup & Clinical Prescriptions (Tab 3):** Instant search and selection across all $2,500+$ subscribers (by ID, archetype, urgency, or contract), individual survival curve with 90% CiPOT conformal bounds, and automated rule-based clinical prescription cards.
- **F6. What-If Counterfactual Retention Simulator (Tab 4):** Interactive monthly discount slider (`$0–$35/mo`), 12-Month Contract Lock-in toggle, and VIP Success Concierge toggle (`$25` one-time cost) computing effective hazard multiplier $\gamma$, counterfactual curve $S^*(t) = S(t)^\gamma$, Net LTV Lift, and Campaign ROI %.
- **F7. MLOps DriftLens & SurvivalPFN Cold-Start Engine (Tab 5):** 10-bin PSI and Two-Sample KS drift telemetry comparing reference training vs. active evaluation cohorts, plus zero-shot prior survival inference for brand-new signups.
- **F8. Audit Reports & Sanitized CSV Export (Tab 6 & CLI):** One-click CSV export of customer-level CLTV, 12M/24M retention probabilities, and prescribed actions with OWASP formula-injection sanitization, alongside a full CLI (`main.py`) supporting `--stage2-ablation` and `--drop-total-charges`.

### Out-of-Scope Features (Excluded from MVP)
- Multi-tenant cloud RBAC database authentication (OAuth2/SSO, PostgreSQL user accounts) — MVP operates as a self-contained single-tenant local/containerized analytics workstation.
- Automated outbound CRM webhook execution (e.g., triggering live Stripe billing credits or HubSpot emails directly from the UI).
- Time-varying covariate recurrent survival architectures (e.g., Dynamic-DeepHit with longitudinal monthly event streams).
- GPU cluster distributed training orchestration (Kubernetes/Ray) — MVP models train in $< 5\text{ seconds}$ on standard CPU cores.

## 1.6 User Stories

| ID | User Story | Priority | Target Component |
| :--- | :--- | :--- | :--- |
| **US-01** | As a **VP of Customer Success**, I want to view portfolio-wide Average Dynamic CLTV, At-Risk Revenue, and 12-Month Projected Churn upon launching the dashboard so that I can immediately assess financial exposure. | P0 (Must) | Tab 1 (`dashboard.py`, `DynamicLTVCalculator`) |
| **US-02** | As a **Retention Analyst**, I want the Priority Customer Queue to automatically rank subscribers by lowest 12-month survival probability and let me click "Prescribe" to jump directly to their individual survival profile. | P0 (Must) | Tab 1 & Tab 3 (`dashboard.py`) |
| **US-03** | As a **Retention Analyst**, I want to search across all 2,500+ subscribers by partial ID, segment name, urgency tier, or contract type so that no subscriber is hidden behind pagination caps. | P0 (Must) | Tab 3 (`search_subscribers`, `populateCustomerDropdown`) |
| **US-04** | As a **Retention Manager**, I want to simulate combining a `$15/mo` bill discount with a 12-month contract lock-in and see the exact Net LTV Lift and Campaign ROI % before approving a promotion. | P0 (Must) | Tab 4 (`runSimulation`) |
| **US-05** | As a **Data Scientist**, I want to upload a custom Telco, Retail Sales, or SaaS CSV file (labeled or inference-only) and remap columns interactively without crashing the dashboard or overwriting benchmark files. | P0 (Must) | Sidebar Uploader (`validate_and_parse_csv`, `validator.py`) |
| **US-06** | As an **ML Engineer**, I want to compare DeepSurv, CoxPH, DeepHit, and RSF on held-out test sets using IPCW Brier Scores and run `--stage2-ablation` from the CLI to verify propensity filtering tradeoffs. | P1 (Should) | Tab 2 & CLI (`Evaluator`, `main.py`) |
| **US-07** | As an **MLOps Engineer**, I want to monitor feature drift via PSI and KS statistics and generate zero-shot survival priors for cold-start signups who have zero tenure history. | P1 (Should) | Tab 5 (`DriftDetector`, `AmortizedSurvivalInference`) |
| **US-08** | As a **Finance Auditor**, I want to adjust global CAC (`$120` default) and monthly discount rate ($d = 0.01$ default) and export a formula-sanitized CSV of all subscriber CLTV predictions. | P1 (Should) | Sidebar & Tab 6 (`updateFinancials`, `downloadPredictionsCSV`) |

## 1.7 Success Metrics & KPIs

| Metric Category | Metric Name | Target Threshold | Verified Baseline (IBM Telco Cohort) |
| :--- | :--- | :--- | :--- |
| **Statistical Discrimination** | Harrell's Concordance Index ($C\text{-Index} \uparrow$) | $\ge 0.6000$ | **0.6070** (DeepSurv) / **0.6055** (CoxPH) |
| **Probabilistic Calibration** | Integrated Brier Score ($\text{IBS} \downarrow$) | $\le 0.1600$ | **0.1499** (DeepSurv) / **0.1483** (CoxPH) |
| **Uncertainty Coverage** | Conformal Interval Empirical Coverage ($\alpha=0.10$) | $90.0\% \pm 2.0\%$ | **90.6%** ($q = 0.080$) |
| **System Performance** | Per-Subscriber Neural Inference Latency | $< 2.0\text{ ms}$ | **0.4 ms** per subscriber |
| **Quality Assurance** | Automated Verification & Regression Suite | $100\%$ Pass Rate | **119 / 119 Tests Passing** (`0` failures) |

## 1.8 Assumptions & Dependencies
- **A1. Proportional & Monotonic Survival Structure:** Individual survival probabilities $S(t \mid \mathbf{x})$ start at $1.0$ at $t=0$ and are monotonically non-increasing over $t \in [1, 72]$ months.
- **A2. Independent Right-Censoring:** Administrative censoring time $C_i$ is conditionally independent of true failure time $T_i^*$ given subscriber covariates $\mathbf{x}_i$, enabling unbiased Inverse Probability of Censoring Weighting (IPCW).
- **A3. Local Runtime Environment:** Python $\ge 3.10$ with `numpy`, `pandas`, `scikit-learn`, `scipy`, `torch`, `lifelines`, and `streamlit` pre-installed. Optional `scikit-survival` (`sksurv`) is supported via automatic fallback to constant-hazard baseline mode if absent.

## 1.9 Risks & Mitigation Matrix

| Risk ID | Risk Description | Likelihood | Impact | Implemented Technical Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| **R-01** | **Target Leakage via Cumulative Billing:** `total_charges` scales linearly with `duration` ($\text{TotalCharges} \approx \text{ARPU} \times T_i$), potentially leaking survival duration to Stage 3 models. | High | High | Implemented `--drop-total-charges` CLI flag and `DataPreprocessingPipeline(drop_total_charges=True)` to strip `total_charges` and `AvgMonthlyCharge` for strict leakage-free evaluation. |
| **R-02** | **Test Set Optimism Bias:** Applying Stage 2 propensity filtering to validation or test sets would artificially inflate $C\text{-Index}$ by evaluating only easy-to-classify customers. | Medium | Critical | Enforced strict split isolation in `main.py` and `dashboard.py`: Stage 2 filtering is applied **only** to training data (`X_train`), while `100%` of validation and held-out test records (`500/500`) are evaluated unfiltered. |
| **R-03** | **Micro-Dataset or Zero-Event Crash:** Uploading a 1-row CSV ($N=1$) or a cohort with zero churn events ($\sum E_i = 0$) crashes standard `BatchNorm1d`, `KMeans(n_clusters=4)`, or `CoxPHFitter`. | Medium | High | Engineered `RobustBatchNorm1d` (switches to running stats when batch size is 1), clamped `effective_k = min(k, n_samples)` with zero-padding, and added flat survival fallbacks ($S(t)=1.0$) when $\sum E_i = 0$. |
| **R-04** | **CSV Formula Injection (CSV Injection):** Malicious strings starting with `=`, `+`, `-`, `@` in uploaded `customer_id` or category fields execute arbitrary commands when exported CSVs are opened in Excel. | Medium | High | Implemented `sanitize_csv_cell()` in Python (`retainnet/ltv/calculator.py`) and `sanitizeCSVCellJS()` in JavaScript to prefix dangerous leading characters with a single quote (`'`). |

## 1.10 PRD Acceptance Criteria
- **AC-PRD-01:** Launching `streamlit run dashboard.py` loads the 2,500-subscriber IBM Telco Benchmark by default, displaying exact portfolio KPIs (`2,500` Active Subscribers, `$1,048.32` Average Dynamic CLTV, `$317,126.85` At-Risk Revenue, `32.8%` 12-Mo Projected Churn).
- **AC-PRD-02:** Switching between IBM Telco, Synthetic Weibull Cohort, and Custom CSV updates all 6 tabs, charts, tables, and subscriber dropdowns in-place without stale customer IDs or page crashes.
- **AC-PRD-03:** Running `python main.py --dataset synthetic --model all --stage2-ablation` executes the 60/20/20 train/val/test split, selects the best model strictly on validation $C\text{-Index}$, evaluates 100% of the held-out test set, and prints the filtered vs. unfiltered ablation table.
