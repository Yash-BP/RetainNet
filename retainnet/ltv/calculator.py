import numpy as np
import pandas as pd
from typing import List, Optional

def sanitize_csv_cell(val: any) -> str:
    """
    Sanitizes values against CSV/Spreadsheet Formula Injection.
    Prepends a single quote if the string begins with dangerous formula triggers (=, +, -, @, \\t, \\r).
    """
    s = str(val) if val is not None else ""
    if s and s[0] in ('=', '+', '-', '@', '\t', '\r'):
        return "'" + s
    return s

class DynamicLTVCalculator:
    """Dynamic Customer Lifetime Value (LTV) Engine based on Individualized Survival Distributions (ISD)."""
    def __init__(
        self,
        default_arpu: float = 65.0,
        monthly_discount_rate: float = 0.01,
        acquisition_cost: float = 120.0
    ):
        self.default_arpu = default_arpu
        self.monthly_discount_rate = monthly_discount_rate
        self.acquisition_cost = acquisition_cost

    def calculate_ltv(
        self,
        isd_matrix: np.ndarray,
        time_horizon_months: np.ndarray,
        arpu: Optional[any] = None,
        acquisition_cost: Optional[float] = None
    ) -> np.ndarray:
        arpu_val = arpu if arpu is not None else self.default_arpu
        ac_val = acquisition_cost if acquisition_cost is not None else self.acquisition_cost
        d = self.monthly_discount_rate

        isd_arr = np.asarray(isd_matrix, dtype=float)
        if isd_arr.ndim == 1:
            isd_arr = isd_arr.reshape(1, -1)
        if len(isd_arr) == 0:
            return np.array([], dtype=float)

        # Numerical safety: sanitize NaNs/infinities and bound survival probabilities to [0, 1]
        isd_arr = np.clip(np.nan_to_num(isd_arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        time_arr = np.asarray(time_horizon_months, dtype=float)
        if len(time_arr) == 0:
            return np.zeros(len(isd_arr)) - ac_val

        discount_factors = 1.0 / np.power(1.0 + d, time_arr)

        # Support scalar or 1D array of customer ARPUs
        if isinstance(arpu_val, (int, float, np.floating)):
            arpu_matrix = float(arpu_val) * discount_factors[np.newaxis, :]
        else:
            arpu_arr = np.asarray(arpu_val, dtype=float)
            if arpu_arr.ndim == 1:
                arpu_matrix = arpu_arr[:, np.newaxis] * discount_factors[np.newaxis, :]
            else:
                arpu_matrix = arpu_arr * discount_factors[np.newaxis, :]

        discounted_monthly_rev = isd_arr * arpu_matrix

        if len(time_arr) == 1:
            integral = discounted_monthly_rev[:, 0]
        else:
            if hasattr(np, "trapezoid"):
                integral = np.trapezoid(discounted_monthly_rev, time_arr, axis=1)
            else:
                integral = np.trapz(discounted_monthly_rev, time_arr, axis=1)

        ltv = integral - ac_val
        return ltv

    def generate_ltv_report(
        self,
        customer_ids: List[str],
        isd_matrix: np.ndarray,
        time_horizon_months: np.ndarray,
        monthly_charges: Optional[np.ndarray] = None,
        sanitize_ids: bool = True
    ) -> pd.DataFrame:
        n_customers = len(customer_ids)
        time_arr = np.asarray(time_horizon_months, dtype=float)
        isd_arr = np.asarray(isd_matrix, dtype=float)
        if isd_arr.ndim == 1:
            isd_arr = isd_arr.reshape(1, -1)

        safe_ids = [sanitize_csv_cell(cid) if sanitize_ids else cid for cid in customer_ids]

        if n_customers == 0:
            return pd.DataFrame({
                "customer_id": [],
                "calculated_ltv_$": [],
                "12_month_retention_prob_%": [],
                "24_month_retention_prob_%": [],
                "36_month_retention_prob_%": []
            })

        # Vectorized LTV calculation for all customers simultaneously
        if monthly_charges is not None:
            charges_arr = np.asarray(monthly_charges, dtype=float)
            if len(charges_arr) < n_customers:
                padded = np.full(n_customers, self.default_arpu, dtype=float)
                padded[:len(charges_arr)] = charges_arr
                charges_arr = padded
            elif len(charges_arr) > n_customers:
                charges_arr = charges_arr[:n_customers]
        else:
            charges_arr = self.default_arpu

        ltv_all = self.calculate_ltv(isd_arr[:n_customers], time_arr, arpu=charges_arr)
        ltv_values = [float(np.round(val, 2)) for val in ltv_all]

        if len(time_arr) > 0:
            idx_12 = int(np.argmin(np.abs(time_arr - 12)))
            idx_24 = int(np.argmin(np.abs(time_arr - 24)))
            idx_36 = int(np.argmin(np.abs(time_arr - 36)))
        else:
            idx_12 = idx_24 = idx_36 = None

        if idx_12 is not None and idx_12 < isd_arr.shape[1]:
            months_12_surv = [float(np.round(v, 1)) for v in isd_arr[:n_customers, idx_12] * 100.0]
            months_24_surv = [float(np.round(v, 1)) for v in isd_arr[:n_customers, idx_24] * 100.0]
            months_36_surv = [float(np.round(v, 1)) for v in isd_arr[:n_customers, idx_36] * 100.0]
        else:
            months_12_surv = [100.0] * n_customers
            months_24_surv = [100.0] * n_customers
            months_36_surv = [100.0] * n_customers

        df_report = pd.DataFrame({
            "customer_id": safe_ids,
            "calculated_ltv_$": ltv_values,
            "12_month_retention_prob_%": months_12_surv,
            "24_month_retention_prob_%": months_24_surv,
            "36_month_retention_prob_%": months_36_surv
        })

        return df_report
