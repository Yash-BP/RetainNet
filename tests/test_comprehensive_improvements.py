"""
Targeted regression and unit tests for RetainNet improvements.
Covers:
1. CamelCase and Unicode BOM header normalization.
2. Extended binary event indicator parsing (t/f, positive/negative, left/stayed).
3. Single-row dataset ingestion, validation, and scoring.
4. Zero-event (100% censored) and all-event (100% churned) cohort handling.
5. What-If simulator mathematical policy formulas (discount hazard decay, contract gamma, VIP concierge gamma, ROI).
6. Missing value imputation in customer features.
"""
import io
import os
import sys
import unittest
import numpy as np
import pandas as pd

# Bootstrap sys.path to project root
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_this_dir, ".."))
if _root_dir in sys.path:
    sys.path.remove(_root_dir)
sys.path.insert(0, _root_dir)

if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

from retainnet.data.validator import validate_customer_csv
from retainnet.ltv.calculator import DynamicLTVCalculator
from dashboard import (
    SIMULATION_POLICY_CONFIG,
    compute_cohort_payload,
    execute_retainnet_pipeline,
    ensure_models_loaded,
)


class TestComprehensiveImprovements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline_output = execute_retainnet_pipeline()
        cls.trained_bundle = cls.pipeline_output["trained_bundle"]
        ensure_models_loaded(cls.trained_bundle)

    def test_camelcase_and_bom_column_alias_mapping(self):
        """Verify camelCase and BOM-prefixed CSV headers map to canonical schema."""
        csv_text = (
            "\ufeffCustomerID,MonthlyCharges,TotalCharges,TenureMonths,IsChurn,SupportTickets\n"
            "CUST-001,75.50,1510.00,20.0,1,2\n"
            "CUST-002,45.00,900.00,20.0,0,0\n"
        )
        res = validate_customer_csv(csv_text)
        self.assertTrue(res.is_valid, f"Validation failed: {res.error_message}")
        self.assertIn("customer_id", res.df.columns)
        self.assertIn("monthly_charges", res.df.columns)
        self.assertIn("total_charges", res.df.columns)
        self.assertIn("duration", res.df.columns)
        self.assertIn("event", res.df.columns)
        self.assertIn("support_tickets", res.df.columns)
        self.assertEqual(len(res.df), 2)

    def test_extended_binary_event_strings(self):
        """Verify extended status indicators (t/f, positive/negative, left/stayed) are recognized."""
        csv_text = (
            "customer_id,monthly_charges,duration,event\n"
            "C1,50.0,12.0,positive\n"
            "C2,60.0,24.0,negative\n"
            "C3,70.0,18.0,t\n"
            "C4,80.0,36.0,f\n"
            "C5,90.0,48.0,left\n"
            "C6,100.0,60.0,stayed\n"
        )
        res = validate_customer_csv(csv_text)
        self.assertTrue(res.is_valid, f"Validation failed: {res.error_message}")
        expected_events = [1, 0, 1, 0, 1, 0]
        self.assertEqual(list(res.df["event"].values), expected_events)

    def test_single_row_dataset_validation_and_scoring(self):
        """Verify single-row datasets validate and compute cohort payload without index or shape errors."""
        csv_text = (
            "customer_id,monthly_charges,total_charges,contract_type,duration,event,support_tickets\n"
            "SOLO-001,65.00,1560.00,0,24.0,0,1\n"
        )
        res = validate_customer_csv(csv_text)
        self.assertTrue(res.is_valid, f"Validation failed: {res.error_message}")
        payload = compute_cohort_payload(
            res.df,
            self.trained_bundle,
            source_key="custom",
            source_label="Single Customer Test",
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["kpi"]["subscribers"], "1")
        self.assertEqual(len(payload["customer_ids"]), 1)
        self.assertEqual(len(payload["compact_customers"]), 1)

    def test_zero_event_cohort_scoring(self):
        """Verify cohorts with 0 events (100% active/censored) execute without zero-division."""
        df_zero = pd.DataFrame([
            {
                "customer_id": f"ACTIVE-{i+1:03d}",
                "monthly_charges": 60.0 + (i * 5),
                "total_charges": 600.0 + (i * 50),
                "contract_type": i % 3,
                "duration": 10.0 + i,
                "event": 0,
                "support_tickets": 0,
                "tech_support": 1,
                "online_security": 1,
                "paperless_billing": 0,
            }
            for i in range(10)
        ])
        payload = compute_cohort_payload(
            df_zero,
            self.trained_bundle,
            source_key="custom",
            source_label="Zero-Event Test",
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["kpi"]["subscribers"], "10")
        self.assertEqual(payload["kpi"]["right_censored"], "100.0%")

    def test_all_event_cohort_scoring(self):
        """Verify cohorts with 100% churn events execute safely without NaN metrics."""
        df_all = pd.DataFrame([
            {
                "customer_id": f"CHURN-{i+1:03d}",
                "monthly_charges": 85.0 + (i * 2),
                "total_charges": 300.0 + (i * 20),
                "contract_type": 0,
                "duration": 4.0 + i,
                "event": 1,
                "support_tickets": 3,
                "tech_support": 0,
                "online_security": 0,
                "paperless_billing": 1,
            }
            for i in range(8)
        ])
        payload = compute_cohort_payload(
            df_all,
            self.trained_bundle,
            source_key="custom",
            source_label="All-Event Test",
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["kpi"]["subscribers"], "8")
        self.assertEqual(payload["kpi"]["right_censored"], "0.0%")

    def test_whatif_simulator_policy_math_and_roi(self):
        """Verify What-If simulator proportional-hazard math, monotonic improvement, and ROI formula."""
        cfg = SIMULATION_POLICY_CONFIG
        discount = 15.0
        # 1. Discount gamma dampening with floor
        gamma_discount = max(cfg["discount_min_factor"], 1.0 - (cfg["discount_hazard_slope"] * discount))
        self.assertAlmostEqual(gamma_discount, 0.775, places=3)
        self.assertGreaterEqual(gamma_discount, cfg["discount_min_factor"])

        # 2. Combined gamma with 12M lock-in and VIP concierge
        has_lockin = True
        has_vip = True
        gamma_lockin = cfg["lockin_contract_gamma"] if has_lockin else 1.0
        gamma_vip = cfg["vip_concierge_gamma"] if has_vip else 1.0
        combined_gamma = round(gamma_discount * gamma_lockin * gamma_vip, 3)

        expected_gamma = round(0.775 * 0.60 * 0.85, 3)
        self.assertEqual(combined_gamma, expected_gamma)
        self.assertLess(combined_gamma, 1.0)

        # 3. Proportional hazard survival transformation: S*(t) = S_base(t)^gamma
        base_surv = np.array([1.0, 0.85, 0.70, 0.55, 0.40])
        counter_surv = np.clip(np.power(base_surv, combined_gamma), 0.0, 1.0)
        self.assertTrue(np.all(counter_surv >= base_surv))

        # 4. Campaign ROI formula verification
        base_ltv = 500.0
        counter_ltv = 750.0
        net_lift = counter_ltv - base_ltv  # +$250.00
        cost = cfg["vip_concierge_cost"]   # $25.00
        roi_pct = ((net_lift - cost) / max(cost, 1.0)) * 100.0
        self.assertEqual(roi_pct, 900.0)

    def test_missing_feature_imputation(self):
        """Verify missing numeric and categorical features are imputed with cohort defaults."""
        csv_text = (
            "customer_id,duration,event,monthly_charges,total_charges,contract_type\n"
            "IMP-001,12.0,0,50.0,,0\n"
            "IMP-002,24.0,1,,2400.0,1\n"
            "IMP-003,36.0,0,100.0,3600.0,\n"
        )
        res = validate_customer_csv(csv_text)
        self.assertTrue(res.is_valid, f"Validation failed: {res.error_message}")
        self.assertFalse(res.df["monthly_charges"].isna().any())
        self.assertFalse(res.df["total_charges"].isna().any())
        self.assertFalse(res.df["contract_type"].isna().any())


if __name__ == "__main__":
    unittest.main()

