# RetainNet — Complete Engineering Specification Suite

**Project Name:** RetainNet: Deep Survival Analysis Framework for Dynamic Customer Lifetime Value (CLTV) & Proactive Churn Mitigation  
**Version:** 1.0.0 (Production Baseline)  
**Target Codebase:** `retainnet/` Python Package, `main.py` CLI Engine, `dashboard.py` Enterprise Intelligence System, `tests/` (119 Automated Verification Tests)

---

# 3. System Architecture Document (SAD)

## 3.1 Architectural Overview & Design Philosophy
RetainNet uses a **Modular Monolithic Analytical Architecture** with a **Hybrid Server-Client Execution Model**:
1. **Python Scientific Backend (`retainnet/`):** Encapsulates data ingestion, feature engineering, the 3-stage hybrid pipeline, PyTorch/Lifelines survival models, IPCW evaluation metrics, vectorized CLTV integration, and MLOps telemetry in clean, framework-agnostic Python modules.
2. **Cached Server-Side Orchestrator (`dashboard.py` & `main.py`):** Trains all four survival models once on startup (`@st.cache_resource`), evaluates held-out test splits, and serializes a compact JSON model bundle (`MODEL_BUNDLE`) containing fitted `StandardScaler` parameters, K-Means centroids, baseline survival curves $\hat{S}_0(t)$, exact PyTorch `DeepSurv` MLP weights/biases/BatchNorm running stats, `CoxPH` regression coefficients $\hat{\beta}$, and `AmortizedSurvivalInference` cluster priors.
3. **Zero-Latency Client-Side Inference Engine (Embedded SPA inside `dashboard.py`):** Renders a responsive 6-tab Tailwind CSS + Chart.js Single-Page Application that performs instant tab switching, subscriber filtering, counterfactual simulation, custom CSV parsing, and exact neural forward passes directly in the browser without round-trip Streamlit server reruns.

```
+-----------------------------------------------------------------------------------+
|                        DATA SOURCES & INGESTION LAYER                             |
|  [IBM Telco Benchmark (2,500)]   [Synthetic Weibull Generator]   [Custom CSV <=25M]|
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                   VALIDATION & PREPROCESSING (`retainnet/data/`)                  |
|  validate_customer_csv() -> Alias Mapping -> Currency Cleaning -> Formula Escape  |
|  DataPreprocessingPipeline -> AvgMonthlyCharge / ChargesPerService / MinMaxScaler |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                3-STAGE HYBRID DENOISING PIPELINE (`retainnet/pipeline/`)          |
|  Stage 1: BehavioralSegmentationStage (K-Means k=4 -> One-Hot Archetype Features) |
|  Stage 2: InstanceFilteringStage (Tenure-Weighted L2 Logistic -> Excise 0.30-0.70)|
|           *Applied strictly to X_train; Val & Held-Out Test sets 100% unfiltered* |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                   STAGE 3 SURVIVAL ENGINES (`retainnet/models/`)                  |
|  +------------------+ +------------------+ +------------------+ +---------------+ |
|  | DeepSurv (PyTorch| | DeepHit (PyTorch | | CoxPH Baseline   | | RSF Baseline  | |
|  | [64,32,16] MLP + | | Discrete PMF +   | | (lifelines L2    | | (sksurv /     | |
|  | Breslow Base S0) | | Midpoint Interp) | |  penalizer=0.01) | |  Fallback)    | |
|  +--------+---------+ +--------+---------+ +--------+---------+ +-------+-------+ |
+-----------|--------------------|--------------------|-------------------|---------+
            +--------------------+---------+----------+-------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|             POST-PROCESSING, CLTV & MLOPS (`retainnet/metrics/`, `ltv/`, `adv/`)  |
|  - Evaluator: Harrell's C-Index, IPCW Brier Score, IBS, D-Calibration Chi2        |
|  - DynamicLTVCalculator: Vectorized Trapezoidal Discounted Quadrature             |
|  - ConformalSurvivalCalibrator: Censoring-Aware 90% CiPOT Bounds                  |
|  - DriftDetector (10-Bin PSI & KS) + AmortizedSurvivalInference (Cold-Start Priors)|
+-----------------------------------------+-----------------------------------------+
                                          |
                       +------------------+------------------+
                       |                                     |
                       v                                     v
+--------------------------------------------+ +------------------------------------+
|       CLI RUNNER (`main.py`)               | |   ENTERPRISE UI (`dashboard.py`)   |
| - 60/20/20 Train/Val/Test Split            | | - @st.cache_resource Model Bundle  |
| - Validation-Based Best Model Selection    | | - Client-Side Neural Forward Pass  |
| - `--stage2-ablation` & `--drop-total`     | | - 6 Interactive Tabs + CSV Export  |
+--------------------------------------------+ +------------------------------------+
```

## 3.2 Recommended Technology Stack

| Layer | Technology / Library | Version / Spec | Rationale |
| :--- | :--- | :--- | :--- |
| **Core Language** | Python | $\ge 3.10$ | Standard runtime for scientific computing and PyTorch deep learning. |
| **Deep Learning** | PyTorch (`torch`, `torch.nn`, `torch.optim`) | $\ge 2.0$ (CPU) | Powers `DeepSurvModel` and `DeepHitModel` with custom vectorized Breslow loss and `RobustBatchNorm1d`. |
| **Statistical Survival** | `lifelines` & `scikit-survival` (optional) | `lifelines >= 0.27` | Provides `CoxPHFitter`, `KaplanMeierFitter`, and `concordance_index`. |
| **Machine Learning & Stats** | `scikit-learn`, `scipy`, `numpy`, `pandas` | Standard stack | Powers `KMeans`, `LogisticRegression`, `StandardScaler`, `MinMaxScaler`, `ks_2samp`, `chisquare`, and `np.trapezoid`. |
| **Backend Host & Caching** | Streamlit (`st.cache_resource`, `components.v1.html`) | $\ge 1.28$ | Hosts the application server and caches trained model weights across sessions. |
| **Frontend UI & Styling** | HTML5, Tailwind CSS, Vanilla ES6 JavaScript | Tailwind CDN + Custom Tokens | Eliminates Streamlit widget rerun flicker; enables $< 50\text{ ms}$ tab transitions and live client-side inference. |
| **Data Visualization** | Chart.js + `OfflineChartFallback` | v4.x | Renders interactive histograms, doughnut charts, KM overlays, conformal bands, and counterfactual curves. |

## 3.3 System Components & Module Responsibilities

| Module Path | Key Classes / Functions | Primary Responsibility |
| :--- | :--- | :--- |
| `retainnet/data/synthetic_data.py` | `SyntheticSurvivalDataGenerator` | Generates Weibull time-to-event cohorts with covariate-modulated hazard and administrative censoring. |
| `retainnet/data/telco_data.py` | `TelcoDatasetLoader` | Loads on-disk Telco CSVs or generates the 2,500-subscriber benchmark cohort. |
| `retainnet/data/validator.py` | `validate_customer_csv`, `ValidationResult`, `generate_csv_template` | Universal schema mapper, currency cleaner, range validator, and formula-injection sanitizer. |
| `retainnet/data/pipeline.py` | `DataPreprocessingPipeline` | Feature engineering (`AvgMonthlyCharge`, `ChargesPerService`, `contract_risk`), imputation, scaling, and $N=1$ safe dummy encoding. |
| `retainnet/pipeline/stage1_clustering.py` | `BehavioralSegmentationStage` | K-Means ($k=4$) clustering with micro-cohort clamping and one-hot feature augmentation. |
| `retainnet/pipeline/stage2_filtering.py` | `InstanceFilteringStage` | Tenure-weighted L2 logistic propensity scorer and $[0.30, 0.70]$ ambiguity band filter. |
| `retainnet/pipeline/hybrid_pipeline.py` | `RetainNetHybridPipeline` | Orchestrates Stage 1 + Stage 2 `fit_transform`, `transform`, and `transform_features`. |
| `retainnet/models/cox_baseline.py` | `CoxPHBaselineModel`, `RandomSurvivalForestBaselineModel` | Semiparametric CoxPH with penalizer backoff and RSF with constant-hazard fallback. |
| `retainnet/models/deep_surv.py` | `DeepSurvNetwork`, `DeepSurvModel`, `RobustBatchNorm1d` | Continuous-time neural Cox model with $\mathcal{O}(N)$ vectorized Breslow loss and baseline hazard estimator. |
| `retainnet/models/deep_hit.py` | `DeepHitModel` | Discrete-time PMF neural network with $S(0)=1.0$ anchoring and midpoint survival interpolation. |
| `retainnet/metrics/evaluation.py` | `Evaluator` | Harrell's $C\text{-Index}$, Graf et al. IPCW Brier Score, IBS, and D-Calibration $\chi^2$. |
| `retainnet/ltv/calculator.py` | `DynamicLTVCalculator`, `sanitize_csv_cell` | Vectorized discounted trapezoidal CLTV integration, counterfactual LTV, and CSV cell sanitization. |
| `retainnet/advanced/drift_detection.py` | `DriftDetector` | 10-bin PSI and Two-Sample KS covariate shift detection. |
| `retainnet/advanced/conformal_calibration.py` | `ConformalSurvivalCalibrator` | Censoring-aware CiPOT non-conformity quantile calibration and interval prediction. |
| `retainnet/advanced/amortized_inference.py` | `AmortizedSurvivalInference` | Cluster-specific Breslow baseline survival priors for zero-shot cold-start inference. |

## 3.4 Internal APIs & Data Contracts

### 3.4.1 Python Survival Model Interface Contract
Every Stage 3 model (`CoxPHBaselineModel`, `RandomSurvivalForestBaselineModel`, `DeepSurvModel`, `DeepHitModel`) implements a uniform polymorphic contract:
- `fit(X: np.ndarray, T: np.ndarray, E: np.ndarray, **kwargs) -> Self`
- `predict_risk(X: np.ndarray) -> np.ndarray` — Returns 1D array of shape `(N,)` where higher values indicate higher churn hazard.
- `predict_survival_function(X: np.ndarray, times: np.ndarray) -> np.ndarray` — Returns 2D Individualized Survival Distribution matrix of shape `(N, len(times))` with values strictly in $[0.0, 1.0]$.

### 3.4.2 Server-to-Client JSON Payload Schema (`RAW_SERVER_DATA`)
`dashboard.py` injects a single deterministic JSON payload (`allow_nan=False`) into the frontend containing:
- `datasets`: Dictionary keyed by `"ibm"`, `"weibull"`, and optionally `"custom"`, where each dataset object contains:
  - `kpi`: `{ subscribers, subscribers_num, cltv, cltv_num, atrisk, churn, right_censored, best_cindex, best_ibs, top_model_name, retention_rate, forecast_horizon, forecast_points, latency_ms }`
  - `cltv_histogram`: `{ labels: string[10], counts: int[10] }`
  - `archetypes`: `{ labels: string[4], pcts: float[4] }`
  - `km_overlay`: `{ labels: string[13], km_points: float[13], dsurv_points: float[13] }`
  - `leaderboard`: Array of 4 model evaluation objects `{ name, c_index, ibs, bs12, bs24, bs36, best }`
  - `compact_customers`: Array of $N$ compact 12-tuples `[cid, tenure_months, arpu, contract_code, ret12, ret24, ltv, dsurv_16, cox_16, rsf_16, dhit_16, contract_label]`
  - `priority_queue`: Top 6 highest-risk subscriber summary objects
  - `mlops`: `{ psi, ks_stat, coverage, conformal_q, drift_status, ref_bars: float[5], cohort_bars: float[5] }`
- `model_bundle`: `{ feat_cols, ref_medians, scaler_mean, scaler_scale, kmeans_centers, base_s0_16, time_grid_16, conformal_q, ref_drift_bars, dsurv_mlp, cox_params, amortized_12m_priors }`

## 3.5 Storage, Deployment, Monitoring & Scalability
- **Storage Architecture:** Stateless in-memory execution with on-disk read-only benchmark reference (`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`). User-uploaded CSVs and exported predictions are handled via browser memory blobs (`URL.createObjectURL(blob)`) and session-scoped state (`st.session_state`), ensuring zero persistent PII retention on the server.
- **Deployment Topology:** Single-command execution via `streamlit run dashboard.py --server.port 8501` or CLI batch execution via `python main.py`. Easily containerized in a lightweight `python:3.11-slim` Docker image (CPU-only PyTorch wheel keeps image size $< 900\text{ MB}$).
- **Monitoring & Telemetry:** Built-in MLOps tab tracks live per-subscriber inference latency (`ms`), Population Stability Index (`PSI`), Kolmogorov-Smirnov (`KS`) statistics, D-Calibration $\chi^2$ p-values, and 90% conformal coverage.
- **Scalability:** Vectorized NumPy/PyTorch tensor operations evaluate `2,500` subscribers across `48` horizons in `~1.0 second` total (`0.4 ms/subscriber`) and scale linearly up to `20,000` subscribers per cohort without GPU hardware.
