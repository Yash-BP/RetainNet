# RetainNet — Complete Engineering Specification Suite

**Project Name:** RetainNet: Deep Survival Analysis Framework for Dynamic Customer Lifetime Value (CLTV) & Proactive Churn Mitigation

**Version:** 1.0.0 (Production Baseline)

**Target Codebase:** `retainnet/` Python Package, `main.py` CLI Engine, `dashboard.py` Enterprise Intelligence System, `tests/` (119 Automated Verification Tests)

# 5. Development Plan & Implementation Roadmap

## 5.1 Roadmap Overview & Phased Milestones

| Phase | Milestone Name | Core Deliverables | Dependencies | Priority | Status | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **Phase 1** | **Project Setup & Data Layer** | Package structure (`retainnet/`), Weibull generator, IBM Telco loader, Universal CSV Validator, Feature Engineering pipeline. | Python $\ge 3.10$, NumPy, Pandas, Scikit-Learn | P0 | Completed & Verified | 
| **Phase 2** | **Stage 1 & Stage 2 Hybrid Pipeline** | `BehavioralSegmentationStage` ($k=4$ K-Means), `InstanceFilteringStage` (tenure-weighted L2 logistic filter), `RetainNetHybridPipeline`. | Phase 1 | P0 | Completed & Verified | 
| **Phase 3** | **Stage 3 Survival Models & Evaluator** | `CoxPHBaselineModel`, `RandomSurvivalForestBaselineModel`, `DeepSurvModel` (PyTorch), `DeepHitModel` (PyTorch), IPCW `Evaluator`. | Phase 2, PyTorch, Lifelines | P0 | Completed & Verified | 
| **Phase 4** | **Dynamic CLTV & MLOps Extensions** | `DynamicLTVCalculator` (vectorized trapezoidal integration), `ConformalSurvivalCalibrator`, `DriftDetector`, `AmortizedSurvivalInference`. | Phase 3, SciPy | P0 | Completed & Verified | 
| **Phase 5** | **CLI Runner & Enterprise Dashboard** | `main.py` CLI with `--stage2-ablation` & `--drop-total-charges`; `dashboard.py` 6-tab UI with client-side neural inference and CSV export. | Phases 1–4, Streamlit, Chart.js | P0 | Completed & Verified | 
| **Phase 6** | **Testing, Bug Fixing & Release Audit** | 119 automated unit/integration tests (`tests/`), edge-case hardening ($N=1$, $\sum E=0$, CSV injection), documentation & packaging. | Phases 1–5 | P0 | Completed (119/119 Passing) | 

## 5.2 Detailed Work Breakdown Structure (WBS)

### Phase 1: Environment Setup & Data Ingestion Layer

1. Initialize repository layout (`retainnet/data/`, `retainnet/pipeline/`, `retainnet/models/`, `retainnet/metrics/`, `retainnet/ltv/`, `retainnet/advanced/`, `tests/`, `data/`).

2. Implement `SyntheticSurvivalDataGenerator` in `retainnet/data/synthetic_data.py` with Weibull inverse-CDF sampling and covariate hazard modulation.

3. Implement `TelcoDatasetLoader` in `retainnet/data/telco_data.py` supporting both external CSV paths and the 2,500-subscriber benchmark generator.

4. Build `validate_customer_csv()` in `retainnet/data/validator.py` with universal alias dictionary (`COLUMN_ALIASES`), currency symbol stripping, range validation, duplicate ID disambiguation, and `sanitize_csv_cell()` formula escape.

5. Build `DataPreprocessingPipeline` in `retainnet/data/pipeline.py` with `AvgMonthlyCharge`, `ChargesPerService`, `contract_risk`, `drop_total_charges` support, and $N=1$ safe categorical dummy alignment.

### Phase 2: Stage 1 & Stage 2 Denoising Pipeline

1. Implement `BehavioralSegmentationStage` in `retainnet/pipeline/stage1_clustering.py` with K-Means ($k=4$), dynamic `effective_k` clamping for small cohorts, and one-hot cluster concatenation.

2. Implement `InstanceFilteringStage` in `retainnet/pipeline/stage2_filtering.py` with tenure-scaled reliability weights $\omega_i = 0.25 + 0.75(T_i/T_{\max})$, balanced L2 logistic regression, $[0.30, 0.70]$ boundary excision, and $<10$ sample fallback.

3. Combine both stages in `RetainNetHybridPipeline` (`retainnet/pipeline/hybrid_pipeline.py`), exposing `fit_transform()` for training data and `transform_features()` for unfiltered test evaluation.

### Phase 3: Stage 3 Survival Architectures & IPCW Evaluation

1. Implement `CoxPHBaselineModel` and `RandomSurvivalForestBaselineModel` in `retainnet/models/cox_baseline.py` with L2 penalizer retry and constant-hazard fallback when `scikit-survival` is absent.

2. Implement `RobustBatchNorm1d`, `DeepSurvNetwork`, and `DeepSurvModel` in `retainnet/models/deep_surv.py` with $\mathcal{O}(N)$ reverse-cumulative-sum Breslow partial log-likelihood and Breslow baseline survival estimation.

3. Implement `DeepHitModel` in `retainnet/models/deep_hit.py` with discrete-time PMF softmax output, censored survival likelihood, $S(0)=1.0$ anchoring, and midpoint bin interpolation.

4. Implement `Evaluator` in `retainnet/metrics/evaluation.py` with Harrell's $C\text{-Index}$, Kaplan-Meier censoring distribution $G(t)$ for IPCW Brier Scores at $12\text{M}/24\text{M}/36\text{M}$, Integrated Brier Score ($\text{IBS}$), and D-Calibration $\chi^2$.

### Phase 4: Dynamic CLTV Calculator & MLOps Telemetry

1. Implement `DynamicLTVCalculator` in `retainnet/ltv/calculator.py` using vectorized `np.trapezoid` over $(1+d)^{-t}$, supporting scalar or per-customer vector ARPU, counterfactual $\gamma$ scaling, and `generate_ltv_report()`.

2. Implement `ConformalSurvivalCalibrator` in `retainnet/advanced/conformal_calibration.py` with censoring-aware non-conformity scores $R_i = \vert{}\hat{S}(T_i \mid \mathbf{x}_i) - (1 - E_i)\vert{}$.

3. Implement `DriftDetector` in `retainnet/advanced/drift_detection.py` (10-bin PSI and Two-Sample KS test) and `AmortizedSurvivalInference` in `retainnet/advanced/amortized_inference.py` (per-cluster Breslow survival priors).

### Phase 5: CLI Integration & Enterprise Dashboard Frontend

1. Build `main.py` with `60/20/20` train/validation/test splitting, validation-based best model selection, 100% unfiltered test evaluation, Dynamic CLTV summary, `--stage2-ablation`, and `--drop-total-charges`.

2. Build `dashboard.py` server-side pipeline (`execute_retainnet_pipeline`, `compute_cohort_payload`, `validate_and_parse_csv`, `switch_data_source`, `search_subscribers`) with JSON weight export (`dsurv_mlp`, `cox_params`, `amortized_12m_priors`).

3. Build the embedded 6-tab HTML/Tailwind/Chart.js frontend inside `dashboard.py` with client-side `hydrateCustomerDB()`, `buildCohortFromRecords()`, `validateAndIngestCSVText()`, `reapplyCustomCSVMapping()`, `runSimulation()`, `inferColdStart()`, and `downloadPredictionsCSV()`.

### Phase 6: Verification, Bug Fixing & Deployment Readiness

1. Execute and maintain the 5 test modules in `tests/`:

   * `tests/test_audit_verification.py` (10 tests: schema consistency, vectorized LTV equivalence, priority queue sorting, censoring-aware conformal calibration, amortized priors, $N=1$ transform consistency, DeepHit anchoring, unfiltered test evaluation).

   * `tests/test_dataset_switching.py` (16 tests: IBM default, synthetic count, CSV aliases, empty/malformed rejection, missing columns, negative numeric rejection, duplicate IDs, formula injection defense, full-dataset subscriber search, retail sales adaptive ingestion).

   * `tests/test_comprehensive.py`, `tests/test_edge_cases.py`, `tests/test_retainnet.py` (93 tests covering all models, metrics, pipeline stages, and UI state transitions).

## 5.3 Definition of Done (DoD)

A feature or release candidate of RetainNet is considered **Done** only when all of the following criteria are satisfied:

1. **Zero Placeholder & Schema Integrity:** All modules, CLI outputs, and UI tabs run with real computed tensors and zero hardcoded dummy metrics or bracketed placeholders.

2. **100% Automated Test Pass Rate:** `python -m unittest discover -s tests -v` passes all `119` tests with `0` failures and `0` unhandled warnings.

3. **Strict Split & Leakage Discipline:** Stage 2 propensity filtering is never applied to validation or test splits, model selection uses validation $C\text{-Index}$, and `--drop-total-charges` cleanly removes duration proxy features when requested.

4. **Security & Edge-Case Compliance:** Uploads $> 25\text{ MB}$, binary files, negative charges/durations, and non-binary event flags are safely rejected without mutating active state, and all exported CSV cells are sanitized against formula injection.

5. **Cross-Viewport UI Responsiveness:** All 6 dashboard tabs, interactive charts, search filters, What-If sliders, and CSV exports render cleanly across desktop (`> 1024px`), tablet (`<= 1024px`), and mobile (`<= 640px`) viewports.