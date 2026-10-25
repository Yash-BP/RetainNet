# RetainNet — Complete Engineering Specification Suite

**Project Name:** RetainNet: Deep Survival Analysis Framework for Dynamic Customer Lifetime Value (CLTV) & Proactive Churn Mitigation  
**Version:** 1.0.0 (Production Baseline)  
**Target Codebase:** `retainnet/` Python Package, `main.py` CLI Engine, `dashboard.py` Enterprise Intelligence System, `tests/` (119 Automated Verification Tests)

---

# 2. Software Requirements Specification (SRS)

## 2.1 Functional Requirements

### 2.1.1 Data Ingestion, Validation & Preprocessing (`FR-DATA`)
- **FR-DATA-01 (Benchmark & Synthetic Loaders):** The system shall provide `TelcoDatasetLoader.load_telco_data(filepath, n_samples=2500)` to load either an external Telco CSV or the deterministic 2,500-record Weibull-calibrated IBM Telco benchmark (`seed=42`), and `SyntheticSurvivalDataGenerator(seed).generate(n_samples)` to synthesize cohorts with shape $k=1.2$ and scale $\lambda=40.0$.
- **FR-DATA-02 (Universal CSV Alias Mapping):** The CSV ingestion engine (`validate_customer_csv` and `validate_and_parse_csv`) shall normalize column headers case-insensitively across Telco, Retail Sales, E-Commerce, and SaaS schemas into canonical fields: `customer_id`, `monthly_charges`, `total_charges`, `support_tickets`, `contract_type`, `tech_support`, `online_security`, `paperless_billing`, `duration`, and `event`.
- **FR-DATA-03 (Currency & Numeric Cleaning):** The parser shall strip leading currency symbols (`$`, `€`, `£`, `₹`), whitespace, and thousand-separator commas from string-formatted numeric columns before float conversion.
- **FR-DATA-04 (Labeled vs. Inference-Only Mode Detection):**
  - If both `duration` ($>0$) and `event` ($\in \{0, 1\}$) are present, the cohort shall be processed in **Labeled Survival Mode** (`is_inference_only = False`).
  - If `event` is absent (or both `duration` and `event` are absent), the cohort shall be processed in **Inference-Only Mode** (`is_inference_only = True`), deriving relative tenure in months ($[1.0, 72.0]$) from any detected date column or defaulting to `24.0` months without fabricating ground-truth churn events.
  - If `event` is present without `duration`, validation shall reject the upload with a descriptive error message.
- **FR-DATA-05 (Feature Engineering & Single-Row Consistency):** `DataPreprocessingPipeline` shall engineer `AvgMonthlyCharge`, `ChargesPerService`, and `contract_risk`, fit `MinMaxScaler` and imputation statistics (`numeric_medians`, `categorical_modes`) strictly on the training set, and use `pd.get_dummies(..., drop_first=False)` aligned to `self.feature_names` during `transform()` so that single-row ($N=1$) inference preserves non-reference categorical encodings identically to batch inference.

### 2.1.2 Stage 1 & Stage 2 Hybrid Denoising Pipeline (`FR-PIPE`)
- **FR-PIPE-01 (Stage 1 K-Means Archetype Discovery):** `BehavioralSegmentationStage(n_clusters=4, random_state=42)` shall fit K-Means on scaled covariates, dynamically clamping `effective_k = max(1, min(self.n_clusters, n_samples))` when $N < 4$, and output one-hot cluster matrices padded to width `n_clusters`.
- **FR-PIPE-02 (Stage 2 Tenure-Weighted Propensity Filtering):** `InstanceFilteringStage(confidence_threshold=0.55)` shall assign sample weights $\omega_i = 1.0$ if $E_i = 1$ and $\omega_i = 0.25 + 0.75 \cdot (T_i / T_{\max})$ if $E_i = 0$, fit an L2-regularized logistic classifier (`C=0.5`, `class_weight='balanced'`), and excise training instances where $0.30 \le \hat{p}_i \le 0.70$.
- **FR-PIPE-03 (Underflow & Single-Class Safeguards):** If the training set contains only one event class ($\text{len}(\text{unique}(E)) < 2$) or if boundary filtering leaves fewer than `10` samples or fewer than `2` observed events, `InstanceFilteringStage` shall automatically bypass excision and retain `100%` of the training samples.

### 2.1.3 Stage 3 Survival Models (`FR-MOD`)
- **FR-MOD-01 (DeepSurv PyTorch Engine):** `DeepSurvModel(hidden_dims=[64, 32, 16], dropout=0.2, lr=1e-3, weight_decay=1e-4)` shall train a feed-forward neural network using `RobustBatchNorm1d`, ReLU activations, AdamW optimization, gradient clipping (`max_norm=5.0`), and vectorized Breslow negative partial log-likelihood loss:
  $$\mathcal{L}_{\text{Cox}}(\theta) = -\frac{1}{N_E}\sum_{i: E_i = 1}\left(\hat{h}_\theta(\mathbf{x}_i) - \log \sum_{j: T_j \ge T_i} \exp(\hat{h}_\theta(\mathbf{x}_j))\right) + \lambda_2 \|\theta\|_2^2$$
- **FR-MOD-02 (DeepHit Discrete-Time Engine):** `DeepHitModel(num_bins=20, hidden_dims=[64, 32, 16], lr=1e-3)` shall discretize time into $M$ uniform intervals, output a Softmax probability mass function $\hat{p}_m(\mathbf{x})$, optimize the combined uncensored PMF + right-censored survival log-likelihood, and interpolate continuous survival curves anchored at $S(0 \mid \mathbf{x}) = 1.0$ using bin-center midpoint survival $\hat{S}_{\text{mid}}(t_m \mid \mathbf{x}) = 1 - (\hat{F}_m(\mathbf{x}) - 0.5\hat{p}_m(\mathbf{x}))$.
- **FR-MOD-03 (CoxPH & RSF Baselines):** `CoxPHBaselineModel` shall fit `lifelines.CoxPHFitter` with automatic L2 penalizer escalation (`0.01 -> 0.1 -> 1.0`) on near-collinear matrices. `RandomSurvivalForestBaselineModel` shall wrap `sksurv.ensemble.RandomSurvivalForest` when available and transparently fall back to a constant-hazard baseline (`is_fallback = True`, labeled `Random Survival Forest (Fallback)`) when `scikit-survival` is not installed.

### 2.1.4 Evaluation, Dynamic CLTV, Counterfactuals & MLOps (`FR-EVAL`, `FR-LTV`, `FR-MLOPS`)
- **FR-EVAL-01 (Survival Metrics):** `Evaluator` shall compute Harrell's $C\text{-Index}$, IPCW Brier Score at any horizon $t$ ($\text{BS}(t)$), Integrated Brier Score ($\text{IBS}$) via trapezoidal integration over the time grid, and D-Calibration $\chi^2$ goodness-of-fit across 10 decile bins.
- **FR-LTV-01 (Vectorized Dynamic CLTV):** `DynamicLTVCalculator` shall compute net discounted CLTV for scalar or per-customer vector ARPU arrays via vectorized trapezoidal integration:
  $$\text{CLTV}_i = \sum_{k=1}^{K-1} \frac{1}{2}\left(\frac{\text{ARPU}_i \cdot \hat{S}(t_k \mid \mathbf{x}_i)}{(1+d)^{t_k}} + \frac{\text{ARPU}_i \cdot \hat{S}(t_{k+1} \mid \mathbf{x}_i)}{(1+d)^{t_{k+1}}}\right)(t_{k+1} - t_k) - \text{CAC}$$
- **FR-LTV-02 (Counterfactual Intervention Simulation):** The What-If Simulator shall compute the composite hazard multiplier $\gamma = \gamma_{\text{disc}} \times \gamma_{\text{contract}} \times \gamma_{\text{vip}}$, where $\gamma_{\text{disc}} = \max(0.65, 1.0 - 0.015 \cdot D)$ for monthly discount $D \in [0, 30]$, $\gamma_{\text{contract}} = 0.60$ when 12-Month Lock-in is enabled, and $\gamma_{\text{vip}} = 0.85$ when VIP Concierge (`$25` cost) is enabled, updating $S^*(t \mid \mathbf{x}) = S(t \mid \mathbf{x})^\gamma$, Net LTV Lift, and Campaign ROI %.
- **FR-MLOPS-01 (DriftLens & Conformal Calibration):** `DriftDetector` shall compute 10-bin Population Stability Index ($\text{PSI}$) and Two-Sample Kolmogorov-Smirnov ($D_{\text{KS}}$) statistics per feature, flagging drift when $\text{PSI} \ge 0.20$ or $p_{\text{KS}} < 0.05$. `ConformalSurvivalCalibrator(alpha=0.10)` shall compute censoring-aware non-conformity scores $R_i = |\hat{S}(T_i \mid \mathbf{x}_i) - (1 - E_i)|$ and construct $(1-\alpha)$ prediction bands $[\max(0, \hat{S}-q), \min(1, \hat{S}+q)]$.

## 2.2 User Roles & Permissions
RetainNet v1.0 operates as a local/containerized single-tenant analytical system (`Northstar Production` workspace). Logical role capabilities within the interface and CLI are structured as follows:

| Role | Permitted Operations | Restricted Operations |
| :--- | :--- | :--- |
| **Analytics Viewer / Retention Operator** | View all 6 dashboard tabs, search/select any subscriber, adjust What-If simulator sliders, adjust global horizon/CAC/discount rate, download audit CSVs. | Cannot mutate on-disk benchmark datasets (`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`). |
| **Data Analyst (CSV Uploader)** | Upload custom CSV cohorts ($\le 25\text{ MB}$), remap columns via the interactive Schema Mapper, regenerate synthetic Weibull cohorts (`50–10,000` rows). | Invalid uploads are blocked from overwriting the active session state. |
| **ML / System Administrator (CLI)** | Execute `main.py` training runs, configure `--epochs`, `--n-clusters`, `--stage2-ablation`, `--drop-total-charges`, and run `pytest` / `unittest` suites. | Direct file system access required. |

## 2.3 Business Rules
- **BR-01 (Clinical Tier Classification):** Subscribers are classified into three clinical urgency tiers based strictly on their 12-month survival probability $\hat{S}(12 \mid \mathbf{x})$:
  - **Critical Urgency (`At-Risk Newbies`):** $\hat{S}(12 \mid \mathbf{x}) < 30.0\%$ $\rightarrow$ Prescribed automated `$15/mo` discount linked to 12-month contract lock-in ($\lambda = 0.088/\text{mo}$).
  - **Moderate Priority (`Price-Sensitive`):** $30.0\% \le \hat{S}(12 \mid \mathbf{x}) < 65.0\%$ $\rightarrow$ Prescribed proactive bundle optimization with `$10` bill credit and speed tier boost ($\lambda = 0.035/\text{mo}$).
  - **Healthy / Low Risk (`Loyal Long-Term`):** $\hat{S}(12 \mid \mathbf{x}) \ge 65.0\%$ $\rightarrow$ Prescribed VIP loyalty maintenance and premium cross-sell with zero defensive discounting ($\lambda = 0.009/\text{mo}$).
- **BR-02 (At-Risk Revenue Definition):** Portfolio At-Risk Revenue is defined as the absolute sum of Dynamic CLTV across subscribers falling in the bottom 25th percentile ($\le Q_{0.25}$) of the cohort's CLTV distribution.
- **BR-03 (Priority Queue Ordering):** The Priority Customer Queue in Tab 1 always sorts the active cohort in ascending order of 12-month retention probability (`ret12_num`), surfacing the 6 highest-risk subscribers at the top.

## 2.4 Data Requirements & Validation Rules

| Canonical Field | Data Type | Allowed Range / Domain | Default / Imputation Policy | Validation Rule |
| :--- | :--- | :--- | :--- | :--- |
| `customer_id` | `string` | Non-empty UTF-8 string | Auto-generated `TELCO-0001` / `CUSTOM-0001` | Sanitized via `sanitize_csv_cell()`; strict IDs deduplicated; adapted store/category IDs disambiguated (`ID-0002`). |
| `monthly_charges` | `float64` | Strict: $(0, 100000]$; Adaptive: $[0.01, \infty)$ | Cohort median (`$65.00`) | Rejects negative values or columns with zero valid positive numbers. |
| `total_charges` | `float64` | $[0.0, \infty)$ | `monthly_charges * 12.0` (or `* duration`) | Rejects negative values; dropped when `drop_total_charges=True`. |
| `support_tickets` | `int32` | $[0, \infty)$ | Cohort median (`1`) | Rejects negative values; rounded to nearest integer. |
| `contract_type` | `int32` | $\{0, 1, 2\}$ | Tertile value tiers if absent | Maps `Month-to-Month`$\to 0$, `One Year`$\to 1$, `Two Year`$\to 2$, or modulo-3 category encoding with `contract_label` preservation. |
| `tech_support`, `online_security`, `paperless_billing` | `int32` | $\{0, 1\}$ | Reference median (`0` or `1`) | Maps `Yes/True/1/Y` $\to 1$, `No/False/0/N/No internet service` $\to 0$. |
| `duration` | `float32` | Strict labeled: $(0, \infty)$; Unlabeled: $[1.0, 72.0]$ | `24.0` months (or elapsed months from date column) | Rejects $\le 0$ in labeled mode or $< 0$ in inference-only mode. |
| `event` | `int32` | $\{0, 1\}$ | `0` (in inference-only mode) | Maps `Yes/True/1/Churn/Churned` $\to 1$, `No/False/0/Active/Retained` $\to 0$; rejects non-binary values. |

## 2.5 Authentication, Authorization & Security Requirements
- **SEC-01 (Upload Size & Binary Bomb Protection):** Both Python (`validate_and_parse_csv`, `validate_customer_csv`) and client-side JavaScript (`handleCSVFileSelect`) enforce a strict `25 MB` (`26,214,400` bytes) file size ceiling and immediately reject files containing null bytes (`\x00`).
- **SEC-02 (Spreadsheet Formula Injection Prevention):** Every string cell (`customer_id`, `contract_label`, `Recommended_Action`, and arbitrary text columns) is sanitized before storage and CSV export. Any string starting with `=`, `+`, `-`, `@`, `\t`, or `\r` is prefixed with `'` so spreadsheet applications treat it as literal text.
- **SEC-03 (Benchmark File Isolation):** Custom CSV uploads and synthetic cohort regenerations are processed strictly in memory (`io.StringIO` / `io.BytesIO` / `st.session_state`) and never mutate or overwrite `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- **SEC-04 (Numerical Overflow & Underflow Protection):** Exponential hazard calculations clamp linear predictors to $[-20.0, 20.0]$ (`torch.clamp(log_h, -20.0, 20.0)` and `np.clip(linear_pred, -20.0, 20.0)`), and IPCW censoring survival denominators are floored at $G(t) \ge 10^{-4}$ to prevent division-by-zero or `NaN` propagation.

## 2.6 Error Handling & Edge Cases
- **EDGE-01 (Empty or Header-Only CSV):** Uploading a `0`-byte file, whitespace-only string, or header-only CSV returns `valid=False` with `"Uploaded CSV file is empty"` or `"CSV contains no customer data rows"` and preserves the previously active dataset.
- **EDGE-02 (Malformed RFC-4180 Quotes or Column Mismatch):** Rows with unclosed quotes or column counts differing from the header row are caught by `parseCSVLine` (JS) and `pd.errors.ParserError` (Python), surfacing the exact line number without crashing the UI.
- **EDGE-03 (Single-Row Inference $N=1$):** Passing a 1-row DataFrame through `DataPreprocessingPipeline.transform()`, `BehavioralSegmentationStage`, `DeepSurvModel`, or `DeepHitModel` succeeds without BatchNorm variance errors or dropped dummy columns.
- **EDGE-04 (Zero Observed Events $\sum E_i = 0$):** If a training cohort has zero churn events, `CoxPHBaselineModel` and `DeepSurvModel` set flat baseline survival $\hat{S}_0(t) = 1.0$ across all horizons rather than raising a convergence exception.
- **EDGE-05 (Offline CDN Fallback):** If the client browser has no external internet access to load `Chart.js` from jsDelivr, `window.Chart` automatically falls back to `OfflineChartFallback` so all DOM KPI updates, tables, search filters, and CSV exports continue functioning without JavaScript exceptions.

## 2.7 Performance Requirements
- **PERF-01 (Model Training Time):** Full 3-stage pipeline execution (fitting K-Means, Propensity Filter, CoxPH, RSF, DeepSurv for 60 epochs, and DeepHit for 40 epochs on 2,000 training records) shall complete in $< 8.0\text{ seconds}$ on a standard CPU and be cached via `@st.cache_resource`.
- **PERF-02 (Inference Latency):** Batch survival curve prediction (`predict_survival_function`) across 48 time steps shall execute in $< 1.0\text{ ms}$ per customer (`0.4 ms` verified).
- **PERF-03 (Wire Payload & Client Hydration):** `dashboard.py` strips redundant `customer_db` dictionaries before JSON serialization (`_strip_heavy_customer_db_for_json`) and transfers `compact_customers` arrays, hydrating all 2,500 subscriber objects client-side in $< 5\text{ ms}$.

## 2.8 SRS Acceptance Criteria
- **AC-SRS-01:** `python -m unittest discover -s tests -v` executes all 119 unit and integration tests across `test_audit_verification.py`, `test_comprehensive.py`, `test_dataset_switching.py`, `test_edge_cases.py`, and `test_retainnet.py` with `0` failures and `0` errors.
- **AC-SRS-02:** Uploading a retail sales CSV (`date,store_id,product_category,daily_sales,units_sold,promo` with `"$1,240.50"` currency strings) succeeds in Inference-Only mode, maps `daily_sales -> monthly_charges` and `product_category -> contract_type`, and prevents `date` or `contract_label` from leaking into the numeric feature matrix.
