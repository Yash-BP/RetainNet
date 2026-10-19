#!/usr/bin/env python3
"""
RetainNet CLI: Primary entry point for the 3-Stage Hybrid Deep Survival Analysis Pipeline.

Usage:
    python main.py --dataset synthetic --model deepsurv --epochs 100
    python main.py --dataset synthetic --model coxph
    python main.py --dataset telco --telco-path data/WA_Fn-UseC_-Telco-Customer-Churn.csv --model all
"""
import argparse
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(
        description="RetainNet: Deep Survival Analysis for Dynamic CLTV & Churn Mitigation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=["synthetic", "telco"], default="synthetic",
                        help="Dataset source (default: synthetic)")
    parser.add_argument("--telco-path", type=str, default=None,
                        help="Path to IBM Telco or custom CSV (required if --dataset telco)")
    parser.add_argument("--model", choices=["coxph", "rsf", "deepsurv", "deephit", "all"], default="all",
                        help="Model to train (default: all)")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of synthetic samples (default: 2000)")
    parser.add_argument("--n-clusters", type=int, default=4,
                        help="Number of K-Means clusters for Stage 1 (default: 4)")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Training epochs for deep models (default: 150)")
    parser.add_argument("--horizon", type=int, default=48,
                        help="LTV forecast horizon in months (default: 48)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--stage2-ablation", action="store_true", default=False,
                        help="Run side-by-side comparative evaluation of models trained with Stage 2 filtering vs unfiltered training")
    parser.add_argument("--drop-total-charges", action="store_true", default=False,
                        help="Drop TotalCharges and duration proxy features to eliminate data leakage")
    args = parser.parse_args()

    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    # ──────────────────────────── BANNER ────────────────────────────
    print("=" * 72)
    print("  RetainNet v1.0 - Deep Survival Analysis Framework")
    print("  3-Stage Hybrid Pipeline: Segmentation -> Filtering -> Survival")
    print("=" * 72)

    # ──────────────────────────── DATA LOADING ────────────────────────────
    from retainnet.data import SyntheticSurvivalDataGenerator, DataPreprocessingPipeline, TelcoDatasetLoader

    if args.dataset == "telco":
        if args.telco_path is None:
            print("[ERROR] --telco-path is required when --dataset is telco")
            sys.exit(1)
        print(f"\n[1/6] Loading dataset from: {args.telco_path}")
        loader = TelcoDatasetLoader(filepath=args.telco_path)
        df = loader.load()
    else:
        print(f"\n[1/6] Generating {args.n_samples} synthetic survival samples (Weibull distribution)...")
        gen = SyntheticSurvivalDataGenerator(seed=args.seed)
        df = gen.generate(n_samples=args.n_samples)

    censoring_rate = 1.0 - df["event"].mean()
    print(f"       Dataset size: {len(df)} records")
    print(f"       Right-censoring rate: {censoring_rate:.2%}")
    print(f"       Event (churn) count: {df['event'].sum()}")
    print(f"       Censored (active) count: {len(df) - df['event'].sum()}")

    # ──────────────────────────── TRAIN/VAL/TEST SPLIT & PREPROCESSING ────────────────────────────
    print("\n[2/7] Train/Validation/Test Split & Preprocessing (Strict Isolation)...")
    from sklearn.model_selection import train_test_split
    events_unique = df["event"].unique()
    stratify_target = df["event"].values if len(events_unique) > 1 else None
    
    # 60% Train, 20% Validation, 20% Held-Out Test
    df_train, df_temp = train_test_split(
        df, test_size=0.4, random_state=args.seed, stratify=stratify_target
    )
    temp_stratify = df_temp["event"].values if len(df_temp["event"].unique()) > 1 else None
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, random_state=args.seed, stratify=temp_stratify
    )
    print(f"       Train: {len(df_train)} (60%) | Val: {len(df_val)} (20%) | Test: {len(df_test)} (20%)")

    pipeline = DataPreprocessingPipeline(drop_total_charges=args.drop_total_charges)
    X_train, T_train, E_train = pipeline.fit_transform(df_train)
    X_val, T_val, E_val = pipeline.transform(df_val)
    X_test, T_test, E_test = pipeline.transform(df_test)
    if args.drop_total_charges:
        print("       [!] Leakage Prevention: TotalCharges and duration proxy features dropped.")
    print(f"       Train feature matrix shape: {X_train.shape}")
    print(f"       Engineered features: {len(pipeline.feature_names)} total")

    # ──────────────────────────── STAGE 1 & 2: HYBRID PIPELINE ────────────────────────────
    print(f"\n[3/6] Stage 1: K-Means Behavioral Segmentation (k={args.n_clusters})...")
    from retainnet.pipeline import RetainNetHybridPipeline
    hybrid = RetainNetHybridPipeline(n_clusters=args.n_clusters, random_state=args.seed)
    result = hybrid.fit_transform(X_train, T_train, E_train)

    X_filtered = result["X_filtered"]
    T_filtered = result["T_filtered"]
    E_filtered = result["E_filtered"]
    cluster_labels = result["cluster_labels"]

    # Ensure validation and test feature matrices match Stage 3 input schema (supports augment_features=True/False)
    X_val_eval = hybrid.transform_features(X_val)
    X_test_eval = hybrid.transform_features(X_test)

    print(f"       Clusters discovered: {args.n_clusters} archetypes")
    print(f"  Stage 2: Propensity Filtering (band: 0.30-0.70)...")
    print(f"       Original: {result['original_count']} -> Filtered: {result['filtered_count']}")
    print(f"       Signal retention: {result['retention_rate']:.1%}")

    # ──────────────────────────── STAGE 3: MODEL TRAINING ────────────────────────────
    print(f"\n[4/6] Stage 3: Training survival models...")
    times = np.linspace(1, args.horizon, args.horizon)
    models_to_train = []

    if args.model == "all":
        models_to_train = ["coxph", "rsf", "deepsurv", "deephit"]
    else:
        models_to_train = [args.model]

    trained_models = {}
    for model_name in models_to_train:
        try:
            if model_name == "coxph":
                from retainnet.models import CoxPHBaselineModel
                print(f"       Training CoxPH Baseline...")
                model = CoxPHBaselineModel()
                model.fit(X_filtered, T_filtered, E_filtered)
                trained_models["CoxPH"] = model

            elif model_name == "rsf":
                from retainnet.models import RandomSurvivalForestBaselineModel
                model = RandomSurvivalForestBaselineModel(random_state=args.seed)
                model.fit(X_filtered, T_filtered, E_filtered)
                display_name = getattr(model, "model_display_name", "RSF")
                if getattr(model, "is_fallback", False):
                    print(f"       [INFO] scikit-survival is not installed. {display_name} running in constant-hazard baseline mode.")
                else:
                    print(f"       Training Random Survival Forest (n=100)...")
                trained_models[display_name] = model

            elif model_name == "deepsurv":
                from retainnet.models import DeepSurvModel
                print(f"       Training DeepSurv (epochs={args.epochs}, arch=[64,32,16])...")
                model = DeepSurvModel(hidden_dims=[64, 32, 16], random_state=args.seed)
                model.fit(X_filtered, T_filtered, E_filtered, epochs=args.epochs)
                trained_models["DeepSurv"] = model

            elif model_name == "deephit":
                from retainnet.models import DeepHitModel
                print(f"       Training DeepHit (epochs={args.epochs}, bins=20)...")
                model = DeepHitModel(num_bins=20, hidden_dims=[64, 32, 16], random_state=args.seed)
                model.fit(X_filtered, T_filtered, E_filtered, epochs=args.epochs)
                trained_models["DeepHit"] = model

        except Exception as e:
            print(f"       [WARNING] {model_name} failed: {e}")

    if not trained_models:
        print("[ERROR] No models were trained successfully.")
        sys.exit(1)

    # ──────────────────────────── VALIDATION & MODEL SELECTION ────────────────────────────
    print(f"\n[5/7] Evaluating candidate models on validation set ({len(X_val_eval)} samples)...")
    from retainnet.metrics import Evaluator

    val_results = []
    for name, model in trained_models.items():
        try:
            val_surv = model.predict_survival_function(X_val_eval, times)
            val_risk = model.predict_risk(X_val_eval)
            val_metrics = Evaluator.evaluate_model(
                name, val_surv, times, T_val, E_val, risk_scores=val_risk
            )
            val_results.append(val_metrics)
        except Exception as e:
            print(f"       [WARNING] Validation evaluation failed for {name}: {e}")

    # Best model selected strictly on validation set to prevent test leakage
    best_model_name = None
    if val_results:
        def _val_score(r):
            c = r.get("c_index", -1.0)
            return c if isinstance(c, (int, float, np.floating)) and np.isfinite(c) else -1.0
        best_entry = max(val_results, key=_val_score)
        best_model_name = best_entry["model_name"]
        print(f"       Selected best model via Validation C-Index: {best_model_name} (val C-index: {best_entry.get('c_index', 0):.4f})")

    # ──────────────────────────── TEST SET EVALUATION ────────────────────────────
    print(f"\n[6/7] Final unbiased evaluation on held-out test set ({len(X_test_eval)} samples, 100% evaluated, 0% filtered)...")
    results_table = []
    for name, model in trained_models.items():
        try:
            surv_matrix = model.predict_survival_function(X_test_eval, times)
            risk_scores = model.predict_risk(X_test_eval)
            metrics = Evaluator.evaluate_model(
                name, surv_matrix, times, T_test, E_test, risk_scores=risk_scores
            )
            results_table.append(metrics)
        except Exception as e:
            print(f"       [WARNING] Evaluation failed for {name}: {e}")

    # Print results table
    print("\n" + "=" * 72)
    print("  FINAL TEST SET EVALUATION RESULTS (HELD-OUT)")
    print("=" * 72)
    header = f"  {'Model':<15} {'C-Index ^':>10} {'IBS v':>10} {'BS@12m':>8} {'BS@24m':>8} {'BS@36m':>8}"
    print(header)
    print("  " + "-" * 67)
    for r in results_table:
        c = r.get("c_index", "N/A")
        ibs = r.get("integrated_brier_score", "N/A")
        b12 = r.get("brier_score_t12", "N/A")
        b24 = r.get("brier_score_t24", "N/A")
        b36 = r.get("brier_score_t36", "N/A")
        c_str = f"{c:.4f}" if isinstance(c, (int, float, np.floating)) else str(c)
        ibs_str = f"{ibs:.4f}" if isinstance(ibs, (int, float, np.floating)) else str(ibs)
        b12_str = f"{b12:.4f}" if isinstance(b12, (int, float, np.floating)) else str(b12)
        b24_str = f"{b24:.4f}" if isinstance(b24, (int, float, np.floating)) else str(b24)
        b36_str = f"{b36:.4f}" if isinstance(b36, (int, float, np.floating)) else str(b36)
        marker = " (Best)" if r["model_name"] == best_model_name else ""
        print(f"  {r['model_name'] + marker:<15} {c_str:>10} {ibs_str:>10} {b12_str:>8} {b24_str:>8} {b36_str:>8}")
    print("=" * 72)

    # ──────────────────────────── LTV REPORT ────────────────────────────
    print(f"\n[7/7] Dynamic CLTV Report ({args.horizon}-month horizon)...")
    from retainnet.ltv import DynamicLTVCalculator
    ltv_calc = DynamicLTVCalculator()

    if best_model_name and best_model_name in trained_models:
        best_model = trained_models[best_model_name]
        surv_test = best_model.predict_survival_function(X_test_eval, times)
        arpu_col = next((c for c in ("monthly_charges", "MonthlyCharges", "daily_sales", "sales") if c in df_test.columns), None)
        test_arpu = pd.to_numeric(df_test[arpu_col], errors="coerce").fillna(65.0).values if arpu_col is not None else None
        ltv_values = ltv_calc.calculate_ltv(surv_test, times, arpu=test_arpu)

        print(f"       Model: {best_model_name}")
        print(f"       Mean LTV:   ${np.mean(ltv_values):,.2f}")
        print(f"       Median LTV: ${np.median(ltv_values):,.2f}")
        print(f"       Min LTV:    ${np.min(ltv_values):,.2f}")
        print(f"       Max LTV:    ${np.max(ltv_values):,.2f}")

        high_risk_mask = ltv_values < np.percentile(ltv_values, 25)
        at_risk_revenue = np.sum(ltv_values[high_risk_mask])
        print(f"\n       [!] At-Risk Revenue (bottom 25%): ${abs(at_risk_revenue):,.2f}")
        print(f"       [!] High-risk customers: {np.sum(high_risk_mask)}")

    # ──────────────────────────── STAGE 2 ABLATION STUDY ────────────────────────────
    if args.stage2_ablation:
        print("\n" + "=" * 80)
        print("  STAGE 2 ABLATION STUDY: FILTERED VS. UNFILTERED TRAINING")
        print("=" * 80)
        print(f"  Unfiltered train cohort : {len(X_train)} samples (100.0% retained, 0 discarded)")
        discarded_count = result['original_count'] - result['filtered_count']
        print(f"  Stage 2 filtered cohort : {len(X_filtered)} samples ({result['retention_rate']:.1%} retained, {discarded_count} discarded)")
        print(f"  Held-out test cohort    : {len(X_test)} samples (identical across both variants, 0% filtered)\n")

        ablation_rows = []
        for model_name in models_to_train:
            # 1. Retrieve filtered metrics from results_table
            if model_name == "coxph":
                key_filter = "CoxPH"
            elif model_name == "rsf":
                key_filter = trained_models.get("RSF", trained_models.get("RSF (Fallback)"))
                key_filter = getattr(key_filter, "model_display_name", "RSF") if key_filter else "RSF"
            elif model_name == "deepsurv":
                key_filter = "DeepSurv"
            elif model_name == "deephit":
                key_filter = "DeepHit"
            else:
                key_filter = model_name

            filt_metric = next((r for r in results_table if r["model_name"] == key_filter), None)
            filt_c = filt_metric.get("c_index", "N/A") if filt_metric else "N/A"
            filt_ibs = filt_metric.get("integrated_brier_score", "N/A") if filt_metric else "N/A"

            ablation_rows.append({
                "model": f"{key_filter} [Stage 2 Filtered]",
                "train_n": len(X_filtered),
                "retention": f"{result['retention_rate']:.1%}",
                "discarded": discarded_count,
                "c_index": filt_c,
                "ibs": filt_ibs,
            })

            # 2. Train unfiltered counterpart on full X_train, T_train, E_train
            unfilt_c = "N/A"
            unfilt_ibs = "N/A"
            try:
                if model_name == "coxph":
                    from retainnet.models import CoxPHBaselineModel
                    m_unfilt = CoxPHBaselineModel()
                    m_unfilt.fit(X_train, T_train, E_train)
                elif model_name == "rsf":
                    from retainnet.models import RandomSurvivalForestBaselineModel
                    m_unfilt = RandomSurvivalForestBaselineModel(random_state=args.seed)
                    m_unfilt.fit(X_train, T_train, E_train)
                elif model_name == "deepsurv":
                    from retainnet.models import DeepSurvModel
                    m_unfilt = DeepSurvModel(hidden_dims=[64, 32, 16], random_state=args.seed)
                    m_unfilt.fit(X_train, T_train, E_train, epochs=args.epochs)
                elif model_name == "deephit":
                    from retainnet.models import DeepHitModel
                    m_unfilt = DeepHitModel(num_bins=20, hidden_dims=[64, 32, 16], random_state=args.seed)
                    m_unfilt.fit(X_train, T_train, E_train, epochs=args.epochs)

                s_unfilt = m_unfilt.predict_survival_function(X_test, times)
                r_unfilt = m_unfilt.predict_risk(X_test)
                m_eval = Evaluator.evaluate_model(
                    f"{key_filter}_unfiltered", s_unfilt, times, T_test, E_test, risk_scores=r_unfilt
                )
                unfilt_c = m_eval.get("c_index", "N/A")
                unfilt_ibs = m_eval.get("integrated_brier_score", "N/A")
            except Exception as e:
                print(f"       [WARNING] Unfiltered ablation training failed for {model_name}: {e}")

            ablation_rows.append({
                "model": f"{key_filter} [Unfiltered / All Data]",
                "train_n": len(X_train),
                "retention": "100.0%",
                "discarded": 0,
                "c_index": unfilt_c,
                "ibs": unfilt_ibs,
            })

        print(f"  {'Model Variant':<32} {'Train N':>8} {'Retention':>10} {'Discarded':>10} {'C-Index ^':>10} {'IBS v':>10}")
        print("  " + "-" * 84)
        for row in ablation_rows:
            c_val = row["c_index"]
            ibs_val = row["ibs"]
            c_str = f"{c_val:.4f}" if isinstance(c_val, (int, float, np.floating)) else str(c_val)
            ibs_str = f"{ibs_val:.4f}" if isinstance(ibs_val, (int, float, np.floating)) else str(ibs_val)
            print(f"  {row['model']:<32} {row['train_n']:>8} {row['retention']:>10} {row['discarded']:>10} {c_str:>10} {ibs_str:>10}")
        print("=" * 80)
        print("  Scientific Note: Retaining 100% of data versus propensity filtering produces")
        print("  differing sample efficiency and calibration tradeoffs depending on signal density.")
        print("=" * 80)

    print("\n" + "=" * 72)
    print("  Pipeline complete. Run `streamlit run dashboard.py` for the UI.")
    print("=" * 72)


if __name__ == "__main__":
    main()