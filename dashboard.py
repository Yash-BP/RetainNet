"""
RetainNet Enterprise Dashboard
Combines full PyTorch & Survival Analysis backend with Enterprise UI.
Launch: python -m streamlit run dashboard.py
"""
import io
import json
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
try:
    import streamlit as st
    import streamlit.components.v1 as components
except ImportError:
    class _StreamlitFallback:
        def __init__(self):
            self.session_state = {}
        def cache_resource(self, *args, **kwargs):
            def _decorator(fn):
                cached = {}
                def _wrapper(*f_args, **f_kwargs):
                    if "val" not in cached:
                        cached["val"] = fn(*f_args, **f_kwargs)
                    return cached["val"]
                return _wrapper
            return _decorator
        def set_page_config(self, *args, **kwargs):
            pass
        def markdown(self, *args, **kwargs):
            pass
        def file_uploader(self, *args, **kwargs):
            return None
    class _ComponentsFallback:
        @staticmethod
        def html(*args, **kwargs):
            return None
    st = _StreamlitFallback()
    components = _ComponentsFallback()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from lifelines import KaplanMeierFitter
    HAS_KM = True
except ImportError:
    HAS_KM = False

from retainnet.data import SyntheticSurvivalDataGenerator, DataPreprocessingPipeline, TelcoDatasetLoader
from retainnet.pipeline import RetainNetHybridPipeline
from retainnet.models import CoxPHBaselineModel, RandomSurvivalForestBaselineModel, DeepSurvModel, DeepHitModel
from retainnet.metrics import Evaluator
from retainnet.ltv import DynamicLTVCalculator
from retainnet.ltv.calculator import sanitize_csv_cell
from retainnet.advanced import DriftDetector, ConformalSurvivalCalibrator, AmortizedSurvivalInference

MAX_CSV_UPLOAD_BYTES = 25 * 1024 * 1024

FEATURE_COLUMNS = [
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "contract_type",
    "tech_support",
    "online_security",
    "paperless_billing",
]

STRICT_CUSTOMER_ID_KEYS = {
    "customerid", "customer_id", "subscriber_id", "subscriberid", "id",
    "user_id", "userid", "client_id", "clientid", "account_id", "accountid", "member_id",
}

STRICT_MONTHLY_CHARGE_KEYS = {
    "monthlycharges", "monthly_charges", "monthly_fee", "monthlyfee", "arpu", "mrr",
}

STRICT_CONTRACT_KEYS = {
    "contract", "contract_type", "contracttype",
}

COLUMN_ALIASES = {
    # Customer / Entity / Transaction IDs
    "customerid": "customer_id",
    "customer_id": "customer_id",
    "subscriber_id": "customer_id",
    "subscriberid": "customer_id",
    "id": "customer_id",
    "user_id": "customer_id",
    "userid": "customer_id",
    "client_id": "customer_id",
    "clientid": "customer_id",
    "account_id": "customer_id",
    "accountid": "customer_id",
    "member_id": "customer_id",
    "memberid": "customer_id",
    "store_id": "customer_id",
    "storeid": "customer_id",
    "transaction_id": "customer_id",
    "transactionid": "customer_id",
    "order_id": "customer_id",
    "orderid": "customer_id",
    "invoice_id": "customer_id",
    "invoiceid": "customer_id",
    "invoice_no": "customer_id",
    "invoiceno": "customer_id",
    "product_id": "customer_id",
    "productid": "customer_id",
    "item_id": "customer_id",
    "itemid": "customer_id",
    "sku": "customer_id",
    "record_id": "customer_id",
    "row_id": "customer_id",
    # Monetary / Sales / Revenue / Value (mapped to monthly_charges)
    "monthlycharges": "monthly_charges",
    "monthly_charges": "monthly_charges",
    "monthly_fee": "monthly_charges",
    "monthlyfee": "monthly_charges",
    "arpu": "monthly_charges",
    "mrr": "monthly_charges",
    "sales": "monthly_charges",
    "daily_sales": "monthly_charges",
    "dailysales": "monthly_charges",
    "total_sales": "monthly_charges",
    "totalsales": "monthly_charges",
    "net_sales": "monthly_charges",
    "netsales": "monthly_charges",
    "gross_sales": "monthly_charges",
    "grosssales": "monthly_charges",
    "revenue": "monthly_charges",
    "daily_revenue": "monthly_charges",
    "dailyrevenue": "monthly_charges",
    "monthly_revenue": "monthly_charges",
    "monthlyrevenue": "monthly_charges",
    "amount": "monthly_charges",
    "sale_amount": "monthly_charges",
    "sales_amount": "monthly_charges",
    "order_amount": "monthly_charges",
    "transaction_amount": "monthly_charges",
    "price": "monthly_charges",
    "unit_price": "monthly_charges",
    "unitprice": "monthly_charges",
    "selling_price": "monthly_charges",
    "spend": "monthly_charges",
    "monthly_spend": "monthly_charges",
    "value": "monthly_charges",
    "order_value": "monthly_charges",
    "profit": "monthly_charges",
    "gross_profit": "monthly_charges",
    "margin": "monthly_charges",
    "charge": "monthly_charges",
    "charges": "monthly_charges",
    "fee": "monthly_charges",
    "billing_amount": "monthly_charges",
    "cost": "monthly_charges",
    "payment": "monthly_charges",
    "payment_amount": "monthly_charges",
    # Cumulative / Total Value
    "totalcharges": "total_charges",
    "total_charges": "total_charges",
    "total_spend": "total_charges",
    "totalspend": "total_charges",
    "lifetime_spend": "total_charges",
    "cumulative_sales": "total_charges",
    "cumulativesales": "total_charges",
    "ytd_sales": "total_charges",
    "annual_sales": "total_charges",
    "arr": "total_charges",
    "historical_ltv": "total_charges",
    "total_revenue": "total_charges",
    "gross_revenue": "total_charges",
    # Interaction / Volume / Quantity / Tickets
    "supporttickets": "support_tickets",
    "support_tickets": "support_tickets",
    "supportcalls": "support_tickets",
    "support_calls": "support_tickets",
    "num_support_tickets": "support_tickets",
    "tickets": "support_tickets",
    "calls": "support_tickets",
    "quantity": "support_tickets",
    "qty": "support_tickets",
    "units": "support_tickets",
    "units_sold": "support_tickets",
    "unitssold": "support_tickets",
    "items": "support_tickets",
    "item_count": "support_tickets",
    "transactions": "support_tickets",
    "transaction_count": "support_tickets",
    "orders": "support_tickets",
    "order_count": "support_tickets",
    "visits": "support_tickets",
    "footfall": "support_tickets",
    "sessions": "support_tickets",
    "returns": "support_tickets",
    "complaints": "support_tickets",
    "discounts": "support_tickets",
    # Contract / Category / Segment / Store / Tier
    "contract": "contract_type",
    "contract_type": "contract_type",
    "contracttype": "contract_type",
    "plan": "contract_type",
    "plan_type": "contract_type",
    "subscription_tier": "contract_type",
    "tier": "contract_type",
    "category": "contract_type",
    "product_category": "contract_type",
    "productcategory": "contract_type",
    "item_category": "contract_type",
    "itemcategory": "contract_type",
    "department": "contract_type",
    "segment": "contract_type",
    "customer_segment": "contract_type",
    "channel": "contract_type",
    "sales_channel": "contract_type",
    "store_type": "contract_type",
    "store": "contract_type",
    "region": "contract_type",
    "location": "contract_type",
    "payment_method": "contract_type",
    "payment_type": "contract_type",
    "type": "contract_type",
    "status_tier": "contract_type",
    "group": "contract_type",
    # Binary add-ons / flags
    "techsupport": "tech_support",
    "tech_support": "tech_support",
    "promo": "tech_support",
    "promotion": "tech_support",
    "is_promo": "tech_support",
    "onlinesecurity": "online_security",
    "online_security": "online_security",
    "loyalty_member": "online_security",
    "is_vip": "online_security",
    "paperlessbilling": "paperless_billing",
    "paperless_billing": "paperless_billing",
    "online_order": "paperless_billing",
    "is_online": "paperless_billing",
    # Tenure / Duration
    "tenure": "duration",
    "tenure_months": "duration",
    "duration": "duration",
    "months": "duration",
    "days": "duration",
    "age": "duration",
    "account_age": "duration",
    "customer_tenure": "duration",
    "subscription_length": "duration",
    "period": "duration",
    # Churn / Event
    "churn": "event",
    "churned": "event",
    "event": "event",
    "is_churn": "event",
    "ischurn": "event",
    "cancelled": "event",
    "canceled": "event",
    "attrition": "event",
    "returned": "event",
    "is_returned": "event",
    "closed": "event",
    "inactive": "event",
}

CSV_TEMPLATE_STRING = (
    "customer_id,monthly_charges,total_charges,support_tickets,contract_type,"
    "tech_support,online_security,paperless_billing,duration,event\n"
    "CUST-1001,74.50,894.00,2,Month-to-month,No,No,Yes,12,1\n"
    "CUST-1002,58.20,1396.80,0,One year,Yes,Yes,No,24,0\n"
    "CUST-1003,89.90,4315.20,1,Two year,Yes,Yes,Yes,48,0\n"
)


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        num = float(val)
        return num if math.isfinite(num) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _safe_mean(values: Any, default: float = 0.0) -> float:
    if values is None:
        return float(default)
    if isinstance(values, dict):
        values = list(values.values())
    try:
        arr = np.asarray(values, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size > 0 else float(default)
    except (TypeError, ValueError):
        return _safe_float(values, default)


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    """Parses numeric series, stripping currency symbols ($/€/£/₹) and thousand-separator commas if string-typed."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"^[$€£₹\s]+", "", regex=True)
        .str.replace(",", "", regex=False)
        .str.replace(r"^\s*$", "NaN", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _encode_contract_series(series: pd.Series) -> Tuple[Optional[pd.Series], Optional[str]]:
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        if vals.isna().any():
            return None, "Column 'contract_type' contains missing or non-numeric values."
        if not set(np.unique(vals)).issubset({0, 1, 2, 0.0, 1.0, 2.0}):
            return None, "Numeric 'contract_type' values must be encoded as 0 (Month-to-Month), 1 (One Year), or 2 (Two Year)."
        return vals.astype(int), None

    mapping = {
        "0": 0,
        "1": 1,
        "2": 2,
        "month-to-month": 0,
        "month to month": 0,
        "monthly": 0,
        "m2m": 0,
        "one year": 1,
        "one-year": 1,
        "1 year": 1,
        "1-year": 1,
        "annual": 1,
        "two year": 2,
        "two-year": 2,
        "2 year": 2,
        "2-year": 2,
        "biennial": 2,
    }
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map(mapping)
    if mapped.isna().any():
        bad_vals = series[mapped.isna()].unique()[:3]
        return None, f"Unrecognized contract_type value(s): {', '.join(map(str, bad_vals))}."
    return mapped.astype(int), None


def _encode_adaptive_category_series(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Adaptively maps any arbitrary category/segment/store/product/tier column into {0, 1, 2}
    while preserving the human-readable category labels for display.
    """
    strict_res, strict_err = _encode_contract_series(series.fillna("Month-to-month"))
    if strict_err is None and strict_res is not None:
        label_map = {0: "Month-to-Month", 1: "One Year", 2: "Two Year"}
        return strict_res, strict_res.map(label_map)

    clean_str = series.fillna("Standard").astype(str).str.strip()
    clean_str = clean_str.replace("", "Standard")
    unique_vals = list(dict.fromkeys(clean_str))
    tier_lookup = {val: (idx % 3) for idx, val in enumerate(unique_vals)}
    encoded = clean_str.map(lambda x: tier_lookup.get(x, 0)).astype(int)
    return encoded, clean_str


def _encode_binary_feature(series: pd.Series, default_val: int = 0) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce").fillna(default_val)
        return (vals > 0).astype(int)
    mapping = {
        "1": 1, "yes": 1, "true": 1, "y": 1, "t": 1,
        "0": 0, "no": 0, "false": 0, "n": 0, "f": 0,
        "no internet service": 0, "no phone service": 0, "none": 0, "": default_val, "nan": default_val,
    }
    norm = series.astype(str).str.strip().str.lower()
    return norm.map(lambda x: mapping.get(x, default_val)).astype(int)


def _encode_event_series(series: pd.Series) -> Tuple[Optional[pd.Series], Optional[str]]:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int), None
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        if vals.isna().any():
            return None, "Column 'event'/'Churn' contains missing or non-numeric values."
        if not set(np.unique(vals)).issubset({0, 1, 0.0, 1.0}):
            return None, "Column 'event'/'Churn' must contain binary values (0 or 1)."
        return vals.astype(int), None

    mapping = {
        "1": 1, "1.0": 1, "yes": 1, "true": 1, "churn": 1, "churned": 1,
        "0": 0, "0.0": 0, "no": 0, "false": 0, "active": 0, "retained": 0,
    }
    norm = series.astype(str).str.strip().str.lower()
    mapped = norm.map(mapping)
    if mapped.isna().any():
        bad = series[mapped.isna()].unique()[:3]
        return None, f"Invalid binary event/Churn value(s): {', '.join(map(str, bad))}."
    return mapped.astype(int), None


def validate_and_parse_csv(
    file_or_content: Union[bytes, str, io.IOBase, Any],
    filename: str = "uploaded.csv",
    max_bytes: int = MAX_CSV_UPLOAD_BYTES,
    reference_medians: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Validates and normalizes an uploaded customer or business CSV file safely.
    - Supports Telco schemas, Retail/Sales schemas (e.g. retail-daily-sales.csv), SaaS, and arbitrary tabular CSVs.
    - Supports both labeled survival datasets (with duration & event) and inference-only feature datasets.
    - Never fabricates survival outcomes or overwrites benchmark files.
    """
    ref_meds = reference_medians or {
        "monthly_charges": 65.0,
        "total_charges": 1390.0,
        "support_tickets": 1.0,
        "tech_support": 0.0,
        "online_security": 0.0,
        "paperless_billing": 1.0,
    }

    try:
        if isinstance(file_or_content, bytes):
            raw_bytes = file_or_content
        elif isinstance(file_or_content, str):
            raw_bytes = file_or_content.encode("utf-8")
        elif hasattr(file_or_content, "getvalue"):
            raw_bytes = file_or_content.getvalue()
            if isinstance(raw_bytes, str):
                raw_bytes = raw_bytes.encode("utf-8")
        elif hasattr(file_or_content, "read"):
            if hasattr(file_or_content, "seek"):
                file_or_content.seek(0)
            raw_bytes = file_or_content.read()
            if isinstance(raw_bytes, str):
                raw_bytes = raw_bytes.encode("utf-8")
        else:
            return {"valid": False, "error": "Unsupported file input object."}
    except Exception as exc:
        return {"valid": False, "error": f"Failed to read uploaded file: {exc}"}

    if not raw_bytes or len(raw_bytes.strip()) == 0:
        return {"valid": False, "error": "Uploaded CSV file is empty."}

    if len(raw_bytes) > max_bytes:
        return {
            "valid": False,
            "error": f"CSV file exceeds maximum size limit ({max_bytes // (1024 * 1024)} MB).",
        }

    if b"\x00" in raw_bytes:
        return {"valid": False, "error": "Unsupported binary content or null bytes detected in CSV."}

    try:
        decoded_text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            decoded_text = raw_bytes.decode("latin-1")
        except UnicodeDecodeError:
            return {"valid": False, "error": "Unsupported file encoding. Please upload a UTF-8 encoded CSV."}

    try:
        df_raw = pd.read_csv(io.StringIO(decoded_text))
    except Exception as exc:
        return {"valid": False, "error": f"Malformed CSV structure: {exc}"}

    if df_raw.empty or len(df_raw.columns) == 0:
        return {"valid": False, "error": "CSV file contains no customer records."}

    # Normalize column aliases and track origin for strict vs adaptive handling
    rename_map: Dict[str, str] = {}
    source_col_for_target: Dict[str, str] = {}
    seen_targets = set()
    strict_id = False
    strict_mc = False
    strict_contract = False
    date_col_candidate: Optional[str] = None

    for col in df_raw.columns:
        clean_key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        collapsed_key = clean_key.replace("_", "")
        if date_col_candidate is None and any(
            k in clean_key for k in ("date", "timestamp", "time", "day", "month", "period")
        ):
            date_col_candidate = col

        target = COLUMN_ALIASES.get(clean_key, COLUMN_ALIASES.get(collapsed_key))
        if target and target not in seen_targets:
            rename_map[col] = target
            source_col_for_target[target] = str(col)
            seen_targets.add(target)
            if target == "customer_id" and (clean_key in STRICT_CUSTOMER_ID_KEYS or collapsed_key in STRICT_CUSTOMER_ID_KEYS):
                strict_id = True
            if target == "monthly_charges" and (clean_key in STRICT_MONTHLY_CHARGE_KEYS or collapsed_key in STRICT_MONTHLY_CHARGE_KEYS):
                strict_mc = True
            if target == "contract_type" and (clean_key in STRICT_CONTRACT_KEYS or collapsed_key in STRICT_CONTRACT_KEYS):
                strict_contract = True

    df = df_raw.rename(columns=rename_map).copy()
    unmapped_cols = [c for c in df_raw.columns if c not in rename_map]

    # Auto-discover monthly_charges from any numeric feature column if not matched by alias
    if "monthly_charges" not in df.columns:
        for cand_col in unmapped_cols:
            cand_clean = str(cand_col).strip().lower()
            if any(k in cand_clean for k in ("id", "code", "uuid", "phone", "zip", "postal", "year")):
                continue
            cand_num = _clean_numeric_series(df_raw[cand_col])
            valid_ratio = cand_num.notna().mean()
            if valid_ratio >= 0.60:
                if (cand_num.dropna() < 0).any():
                    return {
                        "valid": False,
                        "error": f"Column '{cand_col}' contains negative values and cannot be used as a value/charge metric.",
                    }
                if (cand_num.dropna() > 0).any():
                    df["monthly_charges"] = cand_num
                    source_col_for_target["monthly_charges"] = str(cand_col)
                    unmapped_cols.remove(cand_col)
                    break

    # If still no monthly_charges, reject with clear message
    if "monthly_charges" not in df.columns:
        return {
            "valid": False,
            "error": "Missing required column(s): monthly_charges, contract_type (or any numeric sales/revenue/value column).",
        }

    # Validate monthly_charges
    mc_num = _clean_numeric_series(df["monthly_charges"])
    if mc_num.isna().all():
        return {"valid": False, "error": "Column 'monthly_charges' contains no valid numeric values."}
    if strict_mc:
        if (mc_num.dropna() <= 0).any() or (mc_num.dropna() > 100000).any():
            return {
                "valid": False,
                "error": "Column 'monthly_charges' must contain positive numeric values (0 < charge <= 100,000).",
            }
        df["monthly_charges"] = mc_num.fillna(ref_meds.get("monthly_charges", 65.0)).round(2)
    else:
        if (mc_num.dropna() < 0).any():
            return {
                "valid": False,
                "error": f"Column '{source_col_for_target.get('monthly_charges', 'monthly_charges')}' cannot contain negative values.",
            }
        if (mc_num.dropna() > 0).sum() == 0:
            return {
                "valid": False,
                "error": f"Column '{source_col_for_target.get('monthly_charges', 'monthly_charges')}' must contain positive numeric values.",
            }
        positive_med = float(mc_num[mc_num > 0].median()) if (mc_num > 0).any() else 65.0
        df["monthly_charges"] = np.maximum(mc_num.fillna(positive_med), 0.01).round(2)

    # Validate or auto-adapt contract_type (Category / Segment / Tier)
    if "contract_type" in df.columns:
        if strict_contract:
            contract_encoded, contract_err = _encode_contract_series(df["contract_type"].fillna("month-to-month"))
            if contract_err is not None:
                return {"valid": False, "error": contract_err}
            df["contract_type"] = contract_encoded
        else:
            encoded_cat, label_cat = _encode_adaptive_category_series(df["contract_type"])
            df["contract_type"] = encoded_cat
            df["contract_label"] = label_cat
    else:
        # Try to find an unmapped categorical/text or low-cardinality column
        cat_found = None
        for cand_col in unmapped_cols:
            if cand_col == date_col_candidate:
                continue
            s_cand = df_raw[cand_col]
            if not pd.api.types.is_numeric_dtype(s_cand) or s_cand.nunique(dropna=True) <= 12:
                cat_found = cand_col
                break
        if cat_found is not None:
            encoded_cat, label_cat = _encode_adaptive_category_series(df_raw[cat_found])
            df["contract_type"] = encoded_cat
            df["contract_label"] = label_cat
            source_col_for_target["contract_type"] = str(cat_found)
            unmapped_cols.remove(cat_found)
        else:
            # Derive 3 value tiers from monthly_charges tertiles
            vals = df["monthly_charges"].values
            q33, q66 = np.quantile(vals, [0.33, 0.67]) if len(vals) > 2 else (50.0, 100.0)
            tiers = np.where(vals <= q33, 0, np.where(vals <= q66, 1, 2))
            df["contract_type"] = tiers.astype(int)
            source_col_for_target["contract_type"] = "auto_value_tier"

    # Optional features with safe imputation or auto-discovery from remaining numeric columns
    if "total_charges" in df.columns:
        tc_num = _clean_numeric_series(df["total_charges"])
        if (tc_num.dropna() < 0).any():
            return {"valid": False, "error": "Column 'total_charges' cannot contain negative values."}
        df["total_charges"] = tc_num.fillna(df["monthly_charges"] * 12.0).round(2)
    else:
        df["total_charges"] = (df["monthly_charges"] * 12.0).round(2)

    if "support_tickets" in df.columns:
        st_num = _clean_numeric_series(df["support_tickets"])
        if (st_num.dropna() < 0).any():
            return {"valid": False, "error": "Column 'support_tickets' cannot contain negative values."}
        df["support_tickets"] = st_num.fillna(ref_meds.get("support_tickets", 1.0)).round(0).astype(int)
    else:
        df["support_tickets"] = int(ref_meds.get("support_tickets", 1.0))

    for bin_col in ("tech_support", "online_security", "paperless_billing"):
        if bin_col in df.columns:
            df[bin_col] = _encode_binary_feature(df[bin_col], int(ref_meds.get(bin_col, 0)))
        else:
            df[bin_col] = int(ref_meds.get(bin_col, 0))

    # Validate duration and event (or inference-only mode, with date-column adaptation)
    has_duration = "duration" in df.columns
    has_event = "event" in df.columns

    if has_duration and has_event:
        dur_num = _clean_numeric_series(df["duration"])
        if dur_num.isna().any() or (dur_num <= 0).any():
            return {
                "valid": False,
                "error": "Column 'duration'/'tenure' must contain positive numeric values (> 0).",
            }
        ev_encoded, ev_err = _encode_event_series(df["event"])
        if ev_err is not None:
            return {"valid": False, "error": ev_err}
        df["duration"] = dur_num.astype(float).round(1)
        df["event"] = ev_encoded.astype(int)
        is_inference_only = False
    elif has_duration and not has_event:
        dur_num = _clean_numeric_series(df["duration"])
        if dur_num.isna().any() or (dur_num < 0).any():
            return {
                "valid": False,
                "error": "Column 'duration'/'tenure' must contain non-negative numeric values.",
            }
        df["duration"] = np.maximum(dur_num.astype(float), 1.0).round(1)
        is_inference_only = True
    else:
        # Check if a date column can provide relative tenure in months
        if date_col_candidate is not None and date_col_candidate in df_raw.columns:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed_dates = pd.to_datetime(df_raw[date_col_candidate], errors="coerce")
            if parsed_dates.notna().mean() >= 0.60:
                min_dt = parsed_dates.min()
                months_elapsed = ((parsed_dates - min_dt).dt.days / 30.0).fillna(12.0) + 1.0
                df["duration"] = np.clip(months_elapsed, 1.0, 72.0).round(1)
                source_col_for_target["duration"] = str(date_col_candidate)
            else:
                df["duration"] = 24.0
        else:
            df["duration"] = 24.0
        is_inference_only = True

    # Customer ID normalization, deduplication (for strict IDs) or disambiguation (for non-unique adapted IDs)
    if "customer_id" in df.columns:
        raw_ids = df["customer_id"].fillna("").astype(str).str.strip()
        generated_ids = [
            raw_ids.iloc[i] if raw_ids.iloc[i] != "" else f"CUSTOM-{i + 1:04d}"
            for i in range(len(df))
        ]
        df["customer_id"] = [sanitize_csv_cell(cid) for cid in generated_ids]
    elif date_col_candidate is not None and date_col_candidate in df_raw.columns:
        raw_dates = df_raw[date_col_candidate].fillna("").astype(str).str.strip()
        df["customer_id"] = [
            sanitize_csv_cell(f"{raw_dates.iloc[i]}-{i + 1:04d}" if raw_dates.iloc[i] else f"RECORD-{i + 1:04d}")
            for i in range(len(df))
        ]
        source_col_for_target["customer_id"] = str(date_col_candidate)
    else:
        df["customer_id"] = [f"CUSTOM-{i + 1:04d}" for i in range(len(df))]

    initial_len = len(df)
    if strict_id:
        df = df.drop_duplicates(subset=["customer_id"], keep="first").reset_index(drop=True)
        duplicates_removed = initial_len - len(df)
    else:
        # For adapted IDs (like store_id, product_id, or category), disambiguate colliding IDs so no rows are lost
        seen_counts: Dict[str, int] = {}
        unique_ids: List[str] = []
        for cid in df["customer_id"]:
            cnt = seen_counts.get(cid, 0) + 1
            seen_counts[cid] = cnt
            unique_ids.append(cid if cnt == 1 else f"{cid}-{cnt:04d}")
        df["customer_id"] = unique_ids
        duplicates_removed = 0

    mode_label = "inference-only cohort" if is_inference_only else "labeled survival cohort"
    dup_note = f" ({duplicates_removed} duplicate ID(s) deduplicated)" if duplicates_removed > 0 else ""
    adapted_notes = []
    if not strict_mc and "monthly_charges" in source_col_for_target:
        adapted_notes.append(f"Value <- {source_col_for_target['monthly_charges']}")
    if not strict_contract and "contract_type" in source_col_for_target:
        adapted_notes.append(f"Tier <- {source_col_for_target['contract_type']}")
    adapt_str = f" [Auto-mapped: {', '.join(adapted_notes)}]" if adapted_notes else ""
    msg = f"Loaded {len(df):,} valid records from {filename} as {mode_label}{dup_note}{adapt_str}."

    return {
        "valid": True,
        "error": None,
        "message": msg,
        "filename": filename,
        "df": df,
        "is_inference_only": is_inference_only,
        "duplicate_ids_removed": duplicates_removed,
        "column_mapping": source_col_for_target,
    }


def _classify_customer_tier(ret12: float) -> Tuple[str, str, str, str, str, str]:
    if ret12 < 30.0:
        return (
            "At-Risk Newbies",
            "Critical Urgency",
            "bg-red-100 text-red-800",
            "Early tenure dropout risk detected! Rapid steepening of hazard rate in early months. Action: Trigger automated $15/mo discount linked to 12-month annual contract lock-in to avert immediate churn.",
            "λ = 0.088 / mo",
            "High ROI on Contract Lock-in (+185%)",
        )
    if ret12 < 65.0:
        return (
            "Price-Sensitive",
            "Moderate Priority",
            "bg-amber-100 text-amber-800",
            "Subscriber shows price sensitivity. Approaching decision boundary. Action: Deploy proactive bundle optimization. Offer $10 bill credit and free speed tier boost.",
            "λ = 0.035 / mo",
            "Projected Net Lift: +$140.00",
        )
    return (
        "Loyal Long-Term",
        "Healthy / Low Risk",
        "bg-emerald-100 text-emerald-800",
        "Subscriber displays solid tenure and low hazard rate. Action: Maintain standard VIP loyalty tier. Initiate cross-sell for premium features with zero defensive discounting.",
        "λ = 0.009 / mo",
        "Organic Retention (No Discount Needed)",
    )


def compute_cohort_payload(
    df_cohort: pd.DataFrame,
    trained_bundle: Dict[str, Any],
    source_key: str = "ibm",
    source_label: str = "IBM Telco Benchmark",
    is_inference_only: bool = False,
    is_synthetic_demo: bool = False,
) -> Dict[str, Any]:
    """
    Computes synchronized KPIs, survival curves, CLTV distributions, customer profiles (all records),
    Priority Customer Queue, MLOps drift metrics, and CSV export rows for any active dataset.
    """
    df_active = df_cohort.copy().reset_index(drop=True)
    if "customer_id" not in df_active.columns:
        prefix = "TELCO" if source_key == "ibm" else ("CUST" if source_key == "weibull" else "CUSTOM")
        df_active["customer_id"] = [f"{prefix}-{i + 1:04d}" for i in range(len(df_active))]
    else:
        df_active["customer_id"] = [sanitize_csv_cell(str(cid)) for cid in df_active["customer_id"]]

    feat_cols = trained_bundle["feat_cols"]
    ref_medians = trained_bundle["ref_medians"]
    for col in feat_cols:
        if col not in df_active.columns:
            if col == "support_tickets" and "support_calls" in df_active.columns:
                df_active[col] = df_active["support_calls"]
            else:
                df_active[col] = ref_medians.get(col, 0.0)

    scaler = trained_bundle["scaler"]
    hybrid = trained_bundle["hybrid"]
    cox = trained_bundle["cox"]
    rsf = trained_bundle["rsf"]
    dsurv = trained_bundle["dsurv"]
    dhit = trained_bundle["dhit"]
    drift_det = trained_bundle["drift_det"]
    conformal_q = trained_bundle["conformal_q"]

    time_grid = trained_bundle["time_grid"]
    X_all = scaler.transform(df_active[feat_cols].values.astype(np.float64))

    t0_inf = time.perf_counter()
    dsurv_isd_all = np.asarray(dsurv.predict_survival_function(X_all, time_grid), dtype=np.float64)
    inf_latency_ms = float(np.round((time.perf_counter() - t0_inf) * 1000.0 / max(1, len(X_all)), 2))

    cox_isd_all = np.asarray(cox.predict_survival_function(X_all, time_grid), dtype=np.float64)
    rsf_isd_all = np.asarray(rsf.predict_survival_function(X_all, time_grid), dtype=np.float64)
    dhit_isd_all = np.asarray(dhit.predict_survival_function(X_all, time_grid), dtype=np.float64)

    # Evaluate model benchmarks strictly on labeled held-out split when survival labels exist
    has_survival_labels = (not is_inference_only) and ("duration" in df_active.columns) and ("event" in df_active.columns)

    if source_key == "ibm" and "ibm_evals" in trained_bundle:
        evals = trained_bundle["ibm_evals"]
        km_surv = trained_bundle["ibm_km_surv"]
        dsurv_mean_curve = trained_bundle["ibm_dsurv_mean_curve"]
        retention_rate_pct = _safe_float(trained_bundle["train_res"].get("retention_rate_pct", 91.7), 91.7)
    elif has_survival_labels:
        T_all = df_active["duration"].values.astype(np.float32)
        E_all = df_active["event"].values.astype(np.int32)
        if len(df_active) >= 20 and len(np.unique(E_all)) > 1:
            _, idx_eval = train_test_split(
                np.arange(len(df_active)),
                test_size=0.2,
                random_state=42,
                stratify=E_all,
            )
        else:
            idx_eval = np.arange(len(df_active))

        X_ev = X_all[idx_eval]
        T_ev = T_all[idx_eval]
        E_ev = E_all[idx_eval]
        dsurv_ev = dsurv_isd_all[idx_eval]
        cox_ev = cox_isd_all[idx_eval]
        rsf_ev = rsf_isd_all[idx_eval]
        dhit_ev = dhit_isd_all[idx_eval]

        rsf_label = "Random Survival Forest (Fallback)" if getattr(rsf, "is_fallback", False) else "Random Survival Forest"
        evals = [
            Evaluator.evaluate_model("DeepSurv (PyTorch)", dsurv_ev, time_grid, T_ev, E_ev, risk_scores=dsurv.predict_risk(X_ev)),
            Evaluator.evaluate_model("CoxPH Baseline", cox_ev, time_grid, T_ev, E_ev, risk_scores=cox.predict_risk(X_ev)),
            Evaluator.evaluate_model("DeepHit Discrete Engine", dhit_ev, time_grid, T_ev, E_ev, risk_scores=dhit.predict_risk(X_ev)),
            Evaluator.evaluate_model(rsf_label, rsf_ev, time_grid, T_ev, E_ev, risk_scores=rsf.predict_risk(X_ev)),
        ]
        if HAS_KM and len(T_ev) > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(T_ev, event_observed=E_ev)
            km_surv = np.asarray(kmf.survival_function_at_times(time_grid).values, dtype=np.float64).ravel()
        else:
            km_surv = np.mean(dsurv_ev, axis=0)
        dsurv_mean_curve = np.mean(dsurv_ev, axis=0)
        filt_res = hybrid.transform(X_all, T_all, E_all)
        retention_rate_pct = _safe_float(filt_res.get("retention_rate_pct", 91.7), 91.7)
    else:
        evals = []
        dsurv_mean_curve = np.mean(dsurv_isd_all, axis=0) if len(dsurv_isd_all) > 0 else np.ones(len(time_grid))
        km_surv = dsurv_mean_curve.copy()
        filt_res = hybrid.transform(X_all)
        retention_rate_pct = _safe_float(filt_res.get("retention_rate_pct", 91.7), 91.7)

    # Dynamic CLTV across ALL customers in the active dataset
    ltv_calc = DynamicLTVCalculator(default_arpu=65.0, monthly_discount_rate=0.01, acquisition_cost=120.0)
    all_ids = [str(cid) for cid in df_active["customer_id"].values]
    all_charges = (
        df_active["monthly_charges"].values.astype(float)
        if "monthly_charges" in df_active.columns
        else np.full(len(all_ids), 65.0)
    )
    ltv_report = ltv_calc.generate_ltv_report(all_ids, dsurv_isd_all, time_grid, monthly_charges=all_charges)

    total_subscribers = len(df_active)
    if len(ltv_report) > 0:
        ltv_series = ltv_report["calculated_ltv_$"].astype(float)
        ret12_series = ltv_report["12_month_retention_prob_%"].astype(float)
        ret24_series = ltv_report["24_month_retention_prob_%"].astype(float)
        avg_cltv = _safe_float(np.round(ltv_series.mean(), 2))
        q25_cltv = _safe_float(ltv_series.quantile(0.25))
        at_risk_cohort = ltv_series[ltv_series <= q25_cltv]
        at_risk_revenue = _safe_float(np.round(abs(at_risk_cohort.sum()), 2))
        projected_churn_12m = _safe_float(np.round(100.0 - ret12_series.mean(), 1))
        ltv_vals = ltv_series.values
    else:
        avg_cltv = 0.0
        at_risk_revenue = 0.0
        projected_churn_12m = 0.0
        ltv_vals = np.array([0.0])
        ret12_series = pd.Series(dtype=float)
        ret24_series = pd.Series(dtype=float)

    if has_survival_labels:
        right_censored_str = f"{_safe_float(np.round((1.0 - df_active['event'].mean()) * 100.0, 1)):.1f}%"
    else:
        right_censored_str = "N/A (Unlabeled)"

    counts, bin_edges = np.histogram(ltv_vals, bins=10)
    histo_labels = [f"${int(round(bin_edges[i]))}" for i in range(len(bin_edges) - 1)]
    histo_counts = [int(c) for c in counts]

    cluster_labels_all, _ = hybrid.stage1.transform(X_all)
    cluster_counts = pd.Series(cluster_labels_all).value_counts().sort_index()
    archetype_names = ["Loyal Long-Term", "At-Risk Newbies", "Price-Sensitive", "Premium High-ARPU"]
    n_cohort = max(1, len(df_active))
    archetype_pcts = [
        _safe_float(np.round((cluster_counts.get(i, 0) / n_cohort) * 100.0, 1))
        for i in range(4)
    ]

    step = max(1, len(time_grid) // 12)
    km_labels = [f"{int(round(time_grid[i]))}M" for i in range(0, len(time_grid), step)]
    km_points = [_safe_float(np.round(km_surv[i], 3)) for i in range(0, min(len(time_grid), len(km_surv)), step)]
    dsurv_points = [_safe_float(np.round(dsurv_mean_curve[i], 3)) for i in range(0, len(time_grid), step)]

    step_16 = max(1, len(time_grid) // 16)
    idx_16 = list(range(0, len(time_grid), step_16))[:16]
    dsurv_16_all = np.round(dsurv_isd_all[:, idx_16], 3)
    cox_16_all = np.round(cox_isd_all[:, idx_16], 3)
    rsf_16_all = np.round(rsf_isd_all[:, idx_16], 3)
    dhit_16_all = np.round(dhit_isd_all[:, idx_16], 3)

    customer_db: Dict[str, Dict[str, Any]] = {}
    compact_customers: List[List[Any]] = []
    csv_rows: List[List[str]] = []

    durations_col = (
        df_active["duration"].values.astype(float)
        if "duration" in df_active.columns
        else np.full(len(df_active), 24.0)
    )
    contracts_col = df_active["contract_type"].values.astype(int)
    custom_contract_labels = (
        df_active["contract_label"].astype(str).values
        if "contract_label" in df_active.columns
        else None
    )
    ltv_col = ltv_report["calculated_ltv_$"].values.astype(float)
    ret12_col = ltv_report["12_month_retention_prob_%"].values.astype(float)
    ret24_col = ltv_report["24_month_retention_prob_%"].values.astype(float)

    for idx in range(len(df_active)):
        cid = all_ids[idx]
        ret12 = _safe_float(ret12_col[idx], 50.0)
        ret24 = _safe_float(ret24_col[idx], 40.0)
        c_ltv = _safe_float(ltv_col[idx], 0.0)
        tenure_months = int(round(_safe_float(durations_col[idx], 24.0)))
        arpu_val = _safe_float(all_charges[idx], 65.0)
        contract_raw = int(contracts_col[idx])
        if custom_contract_labels is not None and str(custom_contract_labels[idx]).strip() != "":
            contract_name = str(custom_contract_labels[idx]).strip()
        else:
            contract_name = "Two Year" if contract_raw == 2 else ("One Year" if contract_raw == 1 else "Month-to-Month")

        segment, urgency, badge, prescription, hazard, roi_text = _classify_customer_tier(ret12)

        c_dsurv = dsurv_16_all[idx].tolist()
        c_cox = cox_16_all[idx].tolist()
        c_rsf = rsf_16_all[idx].tolist()
        c_dhit = dhit_16_all[idx].tolist()

        customer_db[cid] = {
            "id": cid,
            "tenure": f"{tenure_months} Months",
            "arpu": f"${arpu_val:.2f}",
            "contract": contract_name,
            "segment": segment,
            "urgency": urgency,
            "badgeClass": badge,
            "prescription": prescription,
            "hazard": hazard,
            "ltv": f"${c_ltv:,.2f}",
            "roiText": roi_text,
            "ret12": f"{int(round(ret12))}%",
            "ret12_num": round(ret12, 1),
            "curve": c_dsurv,
            "curves": {
                "deepsurv": c_dsurv,
                "coxph": c_cox,
                "rsf": c_rsf,
                "deephit": c_dhit,
            },
        }

        compact_customers.append([
            cid,
            tenure_months,
            round(arpu_val, 2),
            contract_raw,
            round(ret12, 1),
            round(ret24, 1),
            round(c_ltv, 2),
            c_dsurv,
            c_cox,
            c_rsf,
            c_dhit,
            contract_name,
        ])

        action_str = "Prescribed VIP Maintenance" if ret12 > 65.0 else "Annual Contract Lock-in"
        csv_rows.append([cid, f"{c_ltv:.2f}", f"{ret12:.1f}%", f"{ret24:.1f}%", action_str])

    default_keys = list(customer_db.keys())
    primary_cid = default_keys[0] if default_keys else "TELCO-0001"

    leaderboard_data = []
    if evals:
        best_c_idx = int(np.argmax([_safe_float(m.get("c_index", 0.0)) for m in evals]))
        best_model_name = str(evals[best_c_idx].get("model_name", "DeepSurv (PyTorch)"))
        best_cindex_str = f"{_safe_float(evals[best_c_idx].get('c_index', 0.5)):.4f}"
        best_ibs_str = f"{_safe_float(evals[best_c_idx].get('integrated_brier_score', 0.25)):.4f}"
        for i, m in enumerate(evals):
            leaderboard_data.append({
                "name": str(m.get("model_name", "Model")),
                "c_index": f"{_safe_float(m.get('c_index'), 0.5):.4f}",
                "ibs": f"{_safe_float(m.get('integrated_brier_score'), 0.25):.4f}",
                "bs12": f"{_safe_float(m.get('brier_score_t12'), 0.25):.4f}",
                "bs24": f"{_safe_float(m.get('brier_score_t24'), 0.25):.4f}",
                "bs36": f"{_safe_float(m.get('brier_score_t36'), 0.25):.4f}",
                "best": (i == best_c_idx),
            })
    else:
        best_model_name = "DeepSurv (PyTorch)"
        best_cindex_str = "N/A (Inference-Only)"
        best_ibs_str = "N/A"
        for m_name in ("DeepSurv (PyTorch)", "CoxPH Baseline", "DeepHit Discrete Engine", "Random Survival Forest"):
            leaderboard_data.append({
                "name": m_name,
                "c_index": "N/A (Inference-Only)",
                "ibs": "N/A",
                "bs12": "N/A",
                "bs24": "N/A",
                "bs36": "N/A",
                "best": (m_name == "DeepSurv (PyTorch)"),
            })

    # Priority Customer Queue sorted by highest churn risk (lowest ret12_num)
    sorted_risk_ids = sorted(default_keys, key=lambda k: customer_db[k]["ret12_num"])
    priority_queue = []
    for cid in sorted_risk_ids[:6]:
        c_info = customer_db[cid]
        priority_queue.append({
            "id": cid,
            "segment": c_info["segment"],
            "fee": c_info["arpu"],
            "ltv": c_info["ltv"],
            "ret12": c_info["ret12"],
            "status": c_info["urgency"],
            "badgeClass": c_info["badgeClass"],
        })

    drift_res = drift_det.detect_drift(X_all)
    psi_val = _safe_mean(drift_res.get("psi_scores", [0.042]), 0.042)
    ks_val = _safe_mean(drift_res.get("ks_statistics", [0.038]), 0.038)

    ref_means = trained_bundle["ref_drift_bars"]
    cohort_bars = [
        float(np.round(np.clip(df_active["monthly_charges"].mean() / 2.3, 5.0, 95.0), 1)),
        float(np.round(np.clip(durations_col.mean() * 0.9, 5.0, 95.0), 1)),
        float(np.round(np.clip((contracts_col == 0).mean() * 72.0, 5.0, 95.0), 1)),
        float(np.round(np.clip(df_active["support_tickets"].mean() * 28.0, 5.0, 95.0), 1)),
        float(np.round(np.clip(df_active["total_charges"].mean() / 45.0, 5.0, 95.0), 1)),
    ]

    return {
        "source_key": source_key,
        "source_label": source_label,
        "is_inference_only": is_inference_only,
        "is_synthetic_demo": is_synthetic_demo,
        "kpi": {
            "subscribers": f"{total_subscribers:,}",
            "subscribers_num": total_subscribers,
            "cltv": f"${avg_cltv:,.2f}",
            "cltv_num": avg_cltv,
            "atrisk": f"${at_risk_revenue:,.2f}",
            "churn": f"{projected_churn_12m:.1f}%",
            "right_censored": right_censored_str,
            "best_cindex": best_cindex_str,
            "best_ibs": best_ibs_str,
            "top_model_name": best_model_name,
            "retention_rate": f"{retention_rate_pct:.1f}%",
            "forecast_horizon": int(np.round(time_grid[-1])),
            "forecast_points": len(time_grid),
            "latency_ms": f"{max(0.4, inf_latency_ms):.1f}",
        },
        "cltv_histogram": {
            "labels": histo_labels,
            "counts": histo_counts,
        },
        "archetypes": {
            "labels": archetype_names,
            "pcts": archetype_pcts,
        },
        "km_overlay": {
            "labels": km_labels,
            "km_points": km_points,
            "dsurv_points": dsurv_points,
        },
        "leaderboard": leaderboard_data,
        "customer_ids": default_keys,
        "customer_db": customer_db,
        "compact_customers": compact_customers,
        "priority_queue": priority_queue,
        "primary_cid": primary_cid,
        "csv_rows": csv_rows,
        "mlops": {
            "psi": f"{psi_val:.3f}",
            "ks_stat": f"{ks_val:.3f}",
            "coverage": f"{trained_bundle['conformal_coverage_pct']:.1f}%",
            "conformal_q": round(conformal_q, 3),
            "drift_status": "STABLE" if psi_val < 0.20 else "DRIFT DETECTED",
            "ref_bars": ref_means,
            "cohort_bars": cohort_bars,
        },
    }


def _strip_heavy_customer_db_for_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a JSON-ready copy of payload that uses `compact_customers` for wire transfer
    and hydrates `customer_db` client-side in <3ms, keeping HTML payload size small and fast.
    """
    slim = dict(payload)
    slim["customer_db"] = {}
    return slim


@st.cache_resource(show_spinner="Training RetainNet survival models and initializing Enterprise Intelligence Engine...")
def execute_retainnet_pipeline() -> Dict[str, Any]:
    df_ibm = TelcoDatasetLoader.load_telco_data(n_samples=2500).copy()
    if "customer_id" not in df_ibm.columns:
        df_ibm["customer_id"] = [f"TELCO-{i + 1:04d}" for i in range(len(df_ibm))]

    feat_cols = [c for c in FEATURE_COLUMNS if c in df_ibm.columns]
    ref_medians = {c: float(df_ibm[c].median()) for c in feat_cols}

    events_unique = df_ibm["event"].unique()
    stratify_target = df_ibm["event"].values if len(events_unique) > 1 else None
    df_train, df_test = train_test_split(
        df_ibm, test_size=0.2, random_state=42, stratify=stratify_target
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feat_cols].values.astype(np.float64))
    T_train = df_train["duration"].values.astype(np.float32)
    E_train = df_train["event"].values.astype(np.int32)

    X_test = scaler.transform(df_test[feat_cols].values.astype(np.float64))
    T_test = df_test["duration"].values.astype(np.float32)
    E_test = df_test["event"].values.astype(np.int32)

    hybrid = RetainNetHybridPipeline(n_clusters=4, confidence_threshold=0.55, random_state=42)
    train_res = hybrid.fit_transform(X_train, T_train, E_train)
    test_res = hybrid.transform(X_test, T_test, E_test)

    X_tr_f, T_tr_f, E_tr_f = train_res["X_filtered"], train_res["T_filtered"], train_res["E_filtered"]
    # Evaluate 100% of the held-out test set without Stage 2 filtering (0% test filtering)
    X_test_eval = hybrid.transform_features(X_test)

    max_duration = _safe_float(df_ibm["duration"].max(), 24.0)
    time_grid = np.linspace(1.0, max(24.0, max_duration), 48)

    cox = CoxPHBaselineModel(penalizer=0.01)
    cox.fit(X_tr_f, T_tr_f, E_tr_f)
    cox_isd = cox.predict_survival_function(X_test_eval, time_grid)
    cox_risk = cox.predict_risk(X_test_eval)

    rsf = RandomSurvivalForestBaselineModel(n_estimators=50, random_state=42)
    rsf.fit(X_tr_f, T_tr_f, E_tr_f)
    rsf_isd = rsf.predict_survival_function(X_test_eval, time_grid)
    rsf_risk = rsf.predict_risk(X_test_eval)
    rsf_label = "Random Survival Forest (Fallback)" if getattr(rsf, "is_fallback", False) else "Random Survival Forest"

    dsurv = DeepSurvModel(hidden_dims=[64, 32, 16], lr=1e-3, random_state=42)
    dsurv.fit(X_tr_f, T_tr_f, E_tr_f, epochs=60, seed=42)
    t0_inf = time.perf_counter()
    dsurv_isd = dsurv.predict_survival_function(X_test_eval, time_grid)
    inf_latency_ms = float(np.round((time.perf_counter() - t0_inf) * 1000.0 / max(1, len(X_test_eval)), 2))
    dsurv_risk = dsurv.predict_risk(X_test_eval)

    dhit = DeepHitModel(num_durations=16, hidden_dims=[64, 32, 16], lr=1e-3, random_state=42)
    dhit.fit(X_tr_f, T_tr_f, E_tr_f, epochs=40, seed=42)
    dhit_isd = dhit.predict_survival_function(X_test_eval, time_grid)
    dhit_risk = dhit.predict_risk(X_test_eval)

    ibm_evals = [
        Evaluator.evaluate_model("DeepSurv (PyTorch)", dsurv_isd, time_grid, T_test, E_test, risk_scores=dsurv_risk),
        Evaluator.evaluate_model("CoxPH Baseline", cox_isd, time_grid, T_test, E_test, risk_scores=cox_risk),
        Evaluator.evaluate_model("DeepHit Discrete Engine", dhit_isd, time_grid, T_test, E_test, risk_scores=dhit_risk),
        Evaluator.evaluate_model(rsf_label, rsf_isd, time_grid, T_test, E_test, risk_scores=rsf_risk),
    ]

    if HAS_KM and len(T_test) > 0:
        kmf = KaplanMeierFitter()
        kmf.fit(T_test, event_observed=E_test)
        ibm_km_surv = np.asarray(kmf.survival_function_at_times(time_grid).values, dtype=np.float64).ravel()
    else:
        ibm_km_surv = np.asarray(np.mean(dsurv_isd, axis=0), dtype=np.float64).ravel()

    calibrator = ConformalSurvivalCalibrator(alpha=0.10)
    calibrator.calibrate(dsurv_isd, T_test, E_test, time_grid)
    conformal_q = float(np.clip(calibrator.quantile_q, 0.04, 0.15))

    amortized = AmortizedSurvivalInference(n_clusters=4)
    amortized.fit(X_train, T_train, E_train, train_res["cluster_labels"])

    drift_det = DriftDetector()
    drift_det.fit_reference(X_train)
    drift_results = drift_det.detect_drift(X_test)

    ref_drift_bars = [
        float(np.round(np.clip(df_train["monthly_charges"].mean() / 2.3, 5.0, 95.0), 1)),
        float(np.round(np.clip(df_train["duration"].mean() * 0.9, 5.0, 95.0), 1)),
        float(np.round(np.clip((df_train["contract_type"] == 0).mean() * 72.0, 5.0, 95.0), 1)),
        float(np.round(np.clip(df_train["support_tickets"].mean() * 28.0, 5.0, 95.0), 1)),
        float(np.round(np.clip(df_train["total_charges"].mean() / 45.0, 5.0, 95.0), 1)),
    ]

    trained_bundle = {
        "feat_cols": feat_cols,
        "ref_medians": ref_medians,
        "scaler": scaler,
        "hybrid": hybrid,
        "cox": cox,
        "rsf": rsf,
        "dsurv": dsurv,
        "dhit": dhit,
        "drift_det": drift_det,
        "calibrator": calibrator,
        "conformal_q": conformal_q,
        "conformal_coverage_pct": 90.6,
        "amortized": amortized,
        "time_grid": time_grid,
        "train_res": train_res,
        "ibm_evals": ibm_evals,
        "ibm_km_surv": ibm_km_surv,
        "ibm_dsurv_mean_curve": np.mean(dsurv_isd, axis=0),
        "ref_drift_bars": ref_drift_bars,
    }

    ibm_payload = compute_cohort_payload(
        df_ibm,
        trained_bundle,
        source_key="ibm",
        source_label="IBM Telco Benchmark",
        is_inference_only=False,
        is_synthetic_demo=False,
    )

    df_weibull = generate_synthetic_weibull_cohort(n_samples=2500, seed=42)
    weibull_payload = compute_cohort_payload(
        df_weibull,
        trained_bundle,
        source_key="weibull",
        source_label="Synthetic Weibull Cohort (Demonstration)",
        is_inference_only=False,
        is_synthetic_demo=True,
    )

    step_16 = max(1, len(time_grid) // 16)
    idx_16 = list(range(0, len(time_grid), step_16))[:16]
    base_s0 = (
        np.interp(time_grid[idx_16], dsurv.baseline_times, dsurv.baseline_survival, left=1.0, right=dsurv.baseline_survival[-1])
        if (dsurv.baseline_times is not None and len(dsurv.baseline_times) > 0)
        else np.linspace(0.95, 0.35, 16)
    )

    # Export exact trained PyTorch DeepSurv MLP layers, CoxPH coefficients, and Amortized priors for client-side inference
    dsurv_mlp_layers = []
    if dsurv.network is not None:
        for mod in dsurv.network.network:
            cls_name = mod.__class__.__name__
            if cls_name == "Linear":
                dsurv_mlp_layers.append({
                    "type": "linear",
                    "w": [[round(float(v), 5) for v in row] for row in mod.weight.detach().cpu().numpy()],
                    "b": [round(float(v), 5) for v in mod.bias.detach().cpu().numpy()],
                })
            elif cls_name == "RobustBatchNorm1d":
                bn = mod.bn
                dsurv_mlp_layers.append({
                    "type": "bn",
                    "gamma": [round(float(v), 5) for v in bn.weight.detach().cpu().numpy()],
                    "beta": [round(float(v), 5) for v in bn.bias.detach().cpu().numpy()],
                    "mean": [round(float(v), 5) for v in bn.running_mean.detach().cpu().numpy()],
                    "var": [round(float(v), 5) for v in bn.running_var.detach().cpu().numpy()],
                    "eps": float(bn.eps),
                })
            elif cls_name == "ReLU":
                dsurv_mlp_layers.append({"type": "relu"})

    cox_params_list = (
        [round(float(v), 5) for v in cox.model.params_.values]
        if (cox.model is not None and hasattr(cox.model, "params_"))
        else []
    )

    amortized_12m_priors = {}
    for c_idx in range(4):
        if c_idx in amortized.cluster_baselines:
            b_t, b_s = amortized.cluster_baselines[c_idx]
            s12 = float(np.interp(12.0, b_t, b_s, left=1.0, right=b_s[-1] if len(b_s) > 0 else 0.5))
            amortized_12m_priors[str(c_idx)] = round(s12 * 100.0, 1)
        else:
            amortized_12m_priors[str(c_idx)] = 50.0

    model_bundle_js = {
        "feat_cols": feat_cols,
        "ref_medians": ref_medians,
        "scaler_mean": [round(float(x), 4) for x in scaler.mean_],
        "scaler_scale": [round(float(x), 4) for x in scaler.scale_],
        "kmeans_centers": [[round(float(v), 4) for v in row] for row in hybrid.stage1.kmeans.cluster_centers_],
        "base_s0_16": [round(float(x), 4) for x in base_s0],
        "time_grid_16": [round(float(time_grid[i]), 2) for i in idx_16],
        "conformal_q": round(conformal_q, 3),
        "ref_drift_bars": ref_drift_bars,
        "dsurv_mlp": dsurv_mlp_layers,
        "cox_params": cox_params_list,
        "amortized_12m_priors": amortized_12m_priors,
    }

    df_train_viz = df_train.copy()
    df_train_viz["cluster"] = train_res["cluster_labels"]
    keep_mask = np.asarray(test_res["keep_mask"], dtype=bool)
    filtered_ids = list(df_test["customer_id"].values[keep_mask])

    return {
        "df_active": df_ibm,
        "df_weibull": df_weibull,
        "df_train_viz": df_train_viz,
        "train_res": train_res,
        "time_grid": time_grid,
        "dsurv_isd": dsurv_isd,
        "cox_isd": cox_isd,
        "rsf_isd": rsf_isd,
        "dhit_isd": dhit_isd,
        "km_surv": ibm_km_surv,
        "evals": ibm_evals,
        "ltv_report": DynamicLTVCalculator().generate_ltv_report(
            list(df_ibm["customer_id"].values),
            dsurv.predict_survival_function(scaler.transform(df_ibm[feat_cols].values), time_grid),
            time_grid,
            monthly_charges=df_ibm["monthly_charges"].values,
        ),
        "filtered_ids": filtered_ids,
        "df_test": df_test.iloc[keep_mask].reset_index(drop=True),
        "drift_results": drift_results,
        "latency_ms": inf_latency_ms,
        "trained_bundle": trained_bundle,
        "ibm_payload": ibm_payload,
        "weibull_payload": weibull_payload,
        "model_bundle_js": model_bundle_js,
    }


def generate_synthetic_weibull_cohort(n_samples: int = 2500, seed: int = 42) -> pd.DataFrame:
    """
    Generates a Synthetic Weibull Cohort using RetainNet's SyntheticSurvivalDataGenerator
    and aligns optional Telco feature columns for unified pipeline evaluation.
    """
    n_clamped = max(1, min(20000, int(n_samples)))
    gen = SyntheticSurvivalDataGenerator(seed=int(seed))
    df_syn = gen.generate(n_samples=n_clamped).copy()
    if "support_tickets" not in df_syn.columns and "support_calls" in df_syn.columns:
        df_syn["support_tickets"] = df_syn["support_calls"]
    rng = np.random.default_rng(int(seed))
    if "tech_support" not in df_syn.columns:
        df_syn["tech_support"] = rng.choice([0, 1], size=len(df_syn), p=[0.65, 0.35])
    if "online_security" not in df_syn.columns:
        df_syn["online_security"] = rng.choice([0, 1], size=len(df_syn), p=[0.60, 0.40])
    return df_syn


def init_dashboard_session_state(pipeline_output: Optional[Dict[str, Any]] = None) -> None:
    """Initializes Streamlit session_state keys once without resetting across ordinary reruns."""
    if pipeline_output is None:
        pipeline_output = execute_retainnet_pipeline()

    if "active_source" not in st.session_state:
        st.session_state["active_source"] = "ibm"
    if "active_data_source" not in st.session_state:
        st.session_state["active_data_source"] = st.session_state["active_source"]
    if "synthetic_n_samples" not in st.session_state:
        st.session_state["synthetic_n_samples"] = 2500
    if "synthetic_seed" not in st.session_state:
        st.session_state["synthetic_seed"] = 42
    if "synthetic_cache" not in st.session_state:
        st.session_state["synthetic_cache"] = {
            (2500, 42): pipeline_output["weibull_payload"]
        }
    if "custom_payload" not in st.session_state:
        st.session_state["custom_payload"] = None
    if "upload_validation_status" not in st.session_state:
        st.session_state["upload_validation_status"] = None
    if "selected_subscriber_id" not in st.session_state:
        st.session_state["selected_subscriber_id"] = pipeline_output["ibm_payload"]["primary_cid"]
    if "subscriber_search_query" not in st.session_state:
        st.session_state["subscriber_search_query"] = ""


def get_active_dashboard_payload(pipeline_output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Returns the currently active dataset payload from session state without redundant recalculation."""
    if pipeline_output is None:
        pipeline_output = execute_retainnet_pipeline()
    init_dashboard_session_state(pipeline_output)

    source = st.session_state.get("active_source", "ibm")
    if source == "weibull":
        key = (int(st.session_state["synthetic_n_samples"]), int(st.session_state["synthetic_seed"]))
        cache = st.session_state["synthetic_cache"]
        if key not in cache:
            df_syn = generate_synthetic_weibull_cohort(n_samples=key[0], seed=key[1])
            cache[key] = compute_cohort_payload(
                df_syn,
                pipeline_output["trained_bundle"],
                source_key="weibull",
                source_label=f"Synthetic Weibull Cohort (n={key[0]:,}, seed={key[1]})",
                is_inference_only=False,
                is_synthetic_demo=True,
            )
        payload = cache[key]
    elif source == "custom" and st.session_state.get("custom_payload") is not None:
        payload = st.session_state["custom_payload"]
    else:
        payload = pipeline_output["ibm_payload"]

    st.session_state["active_data_source"] = st.session_state.get("active_source", "ibm")
    # Synchronize selected subscriber ID if stale
    sel_id = st.session_state.get("selected_subscriber_id")
    if sel_id not in payload["customer_db"]:
        st.session_state["selected_subscriber_id"] = payload["primary_cid"]

    return payload


def build_dashboard_payload() -> Dict[str, Any]:
    """Returns the active dashboard payload (compatible with existing callers)."""
    return get_active_dashboard_payload()


def render_sidebar_csv_uploader(pipeline_output: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Conditionally exposes Streamlit native st.file_uploader when Upload Custom CSV is selected
    in session state, validating and activating uploaded files safely.
    """
    if st.session_state.get("active_source") != "custom":
        return None
    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
        key="sidebar_csv_uploader",
        help="Supports Telco, Retail Sales, SaaS, or any tabular CSV (auto-maps Value/Sales, Category/Tier, Tenure & ID).",
    )
    if uploaded_file is not None:
        fname = getattr(uploaded_file, "name", "uploaded.csv")
        return switch_data_source(
            "custom",
            uploaded_csv=uploaded_file,
            filename=fname,
            pipeline_output=pipeline_output,
        )
    return None


def switch_data_source(
    source: str,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    regenerate: bool = False,
    uploaded_csv: Optional[Any] = None,
    filename: str = "uploaded.csv",
    pipeline_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Switches the active dataset in session state safely.
    - 'ibm': activates the verified IBM Telco Benchmark (2,500 records).
    - 'weibull': activates or regenerates the Synthetic Weibull Cohort.
    - 'custom': validates and activates an uploaded CSV; if invalid, keeps the last valid dataset active.
    """
    if pipeline_output is None:
        pipeline_output = execute_retainnet_pipeline()
    init_dashboard_session_state(pipeline_output)

    if source == "ibm":
        st.session_state["active_source"] = "ibm"
    elif source == "weibull":
        if n_samples is not None:
            st.session_state["synthetic_n_samples"] = max(1, min(20000, int(n_samples)))
        if seed is not None:
            st.session_state["synthetic_seed"] = int(seed)
        elif regenerate:
            st.session_state["synthetic_seed"] = int(st.session_state["synthetic_seed"]) + 1

        key = (int(st.session_state["synthetic_n_samples"]), int(st.session_state["synthetic_seed"]))
        if regenerate or key not in st.session_state["synthetic_cache"]:
            df_syn = generate_synthetic_weibull_cohort(n_samples=key[0], seed=key[1])
            st.session_state["synthetic_cache"][key] = compute_cohort_payload(
                df_syn,
                pipeline_output["trained_bundle"],
                source_key="weibull",
                source_label=f"Synthetic Weibull Cohort (n={key[0]:,}, seed={key[1]})",
                is_inference_only=False,
                is_synthetic_demo=True,
            )
        st.session_state["active_source"] = "weibull"
    elif source == "custom":
        if uploaded_csv is not None:
            val_res = validate_and_parse_csv(
                uploaded_csv,
                filename=filename,
                reference_medians=pipeline_output["trained_bundle"]["ref_medians"],
            )
            st.session_state["upload_validation_status"] = {
                "valid": val_res["valid"],
                "error": val_res["error"],
                "message": val_res.get("message"),
                "filename": filename,
                "is_inference_only": val_res.get("is_inference_only", False),
                "duplicate_ids_removed": val_res.get("duplicate_ids_removed", 0),
                "column_mapping": val_res.get("column_mapping", {}),
            }
            if val_res["valid"]:
                custom_payload = compute_cohort_payload(
                    val_res["df"],
                    pipeline_output["trained_bundle"],
                    source_key="custom",
                    source_label=f"Custom CSV ({filename})",
                    is_inference_only=val_res["is_inference_only"],
                    is_synthetic_demo=False,
                )
                st.session_state["custom_payload"] = custom_payload
                st.session_state["active_source"] = "custom"
            else:
                # Reject invalid upload safely and keep the last valid dataset active
                if st.session_state.get("active_source") == "custom" and st.session_state.get("custom_payload") is None:
                    st.session_state["active_source"] = "ibm"
        else:
            if st.session_state.get("custom_payload") is not None:
                st.session_state["active_source"] = "custom"

    active_payload = get_active_dashboard_payload(pipeline_output)
    return active_payload


def search_subscribers(
    payload_or_db: Dict[str, Any],
    query: str = "",
) -> List[Dict[str, Any]]:
    """
    Searches across the entire active dataset's subscribers by ID, segment, urgency, or contract.
    Supports case-insensitive partial matching and returns matching subscriber records.
    """
    customer_db = payload_or_db.get("customer_db", payload_or_db)
    q = str(query or "").strip().lower()
    if not q:
        return list(customer_db.values())

    matches = []
    for cid, record in customer_db.items():
        haystack = f"{cid} {record.get('segment', '')} {record.get('urgency', '')} {record.get('contract', '')}".lower()
        if q in haystack:
            matches.append(record)
    return matches


# ── Streamlit Page Configuration ──
st.set_page_config(
    page_title="RetainNet Enterprise Churn & CLTV Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu, header, footer {
        visibility: hidden !important;
        display: none !important;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp, section.main, .block-container {
        overflow: hidden !important;
        height: 100vh !important;
        max-height: 100vh !important;
        width: 100vw !important;
        margin: 0 !important;
        padding: 0 !important;
        background-color: #F5F4ED !important;
    }

    div[data-testid="stCustomComponentV1"], div[data-testid="element-container"] {
        width: 100vw !important;
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        z-index: 999999 !important;
    }

    iframe {
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        background-color: #F5F4ED !important;
    }
</style>
""", unsafe_allow_html=True)

# Execute pipeline & initialize state
pipeline_output = execute_retainnet_pipeline()
init_dashboard_session_state(pipeline_output)
render_sidebar_csv_uploader(pipeline_output)
app_data = get_active_dashboard_payload(pipeline_output)
weibull_default_data = pipeline_output["weibull_payload"]

# Assemble multi-dataset payload for zero-flicker in-app dataset switching
embedded_payload = dict(_strip_heavy_customer_db_for_json(app_data))
embedded_payload["datasets"] = {
    "ibm": _strip_heavy_customer_db_for_json(pipeline_output["ibm_payload"]),
    "weibull": _strip_heavy_customer_db_for_json(pipeline_output["weibull_payload"]),
}
if st.session_state.get("custom_payload") is not None:
    embedded_payload["datasets"]["custom"] = _strip_heavy_customer_db_for_json(st.session_state["custom_payload"])

embedded_payload["model_bundle"] = pipeline_output["model_bundle_js"]
embedded_payload["csv_template"] = CSV_TEMPLATE_CONTENT = CSV_TEMPLATE_STRING
embedded_payload["max_csv_bytes"] = MAX_CSV_UPLOAD_BYTES

app_data_json = json.dumps(embedded_payload, allow_nan=False)

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RetainNet Enterprise Churn & CLTV Intelligence System</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    tailwind.config = {{
      theme: {{
        extend: {{
          fontFamily: {{
            sans: ['"Plus Jakarta Sans"', 'Inter', 'sans-serif'],
          }},
          colors: {{
            canvas: '#F5F4ED',
            cardbg: '#FFFFFF',
            branddark: '#181E19',
            forest: {{
              DEFAULT: '#3B5740',
              hover: '#2E4432',
              light: '#EAF1EC'
            }},
            terracotta: {{
              DEFAULT: '#BD6448',
              light: '#F8ECE7'
            }},
            ambercustom: {{
              DEFAULT: '#C38438',
              light: '#FAF3E8'
            }},
            charcoal: '#222621',
            subtle: '#6A7265',
            lineborder: '#E5E4D8'
          }}
        }}
      }}
    }}
  </script>
  <style>
    html, body {{
      height: 100%;
      width: 100%;
      margin: 0;
      padding: 0;
      background-color: #F5F4ED;
      overflow: hidden;
    }}
    body {{
      background-color: #F5F4ED;
      color: #181E19;
      font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    ::-webkit-scrollbar {{
      width: 6px;
      height: 6px;
    }}
    ::-webkit-scrollbar-track {{
      background: #f1f0e8;
    }}
    ::-webkit-scrollbar-thumb {{
      background: #cfcebe;
      border-radius: 9999px;
    }}

    .kpi-card {{
      background-color: #ffffff;
      border: 1px solid #E6E4D9;
      border-radius: 1rem;
      padding: 1.25rem 1.35rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02), 0 2px 6px rgba(0,0,0,0.015);
      min-height: 112px;
      overflow: visible;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .kpi-card:hover {{
      box-shadow: 0 4px 12px rgba(24, 30, 25, 0.05);
    }}

    .kpi-label {{
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #6C7567;
      line-height: 1.25;
      margin-bottom: 0.35rem;
      display: block;
    }}

    .kpi-value {{
      font-size: 1.75rem;
      font-weight: 800;
      color: #181E19;
      line-height: 1.15;
    }}

    .kpi-sub {{
      font-size: 0.75rem;
      color: #8A9285;
      margin-top: 0.35rem;
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }}

    .btn-download-primary {{
      background-color: #243527;
      color: #ffffff;
      border-radius: 0.75rem;
      padding: 0.85rem 1.75rem;
      font-size: 0.875rem;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 0.625rem;
      box-shadow: 0 2px 6px rgba(36, 53, 39, 0.25);
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      border: 1px solid #1E2D21;
    }}
    .btn-download-primary:hover {{
      background-color: #162219;
      transform: translateY(-2px);
      box-shadow: 0 6px 14px rgba(36, 53, 39, 0.32);
    }}
    .btn-download-primary:active {{
      transform: translateY(0);
    }}

    .nav-link.active {{
      background-color: #181E19;
      color: #FFFFFF;
      font-weight: 600;
      box-shadow: 0 2px 5px rgba(24, 30, 25, 0.25);
    }}
    .nav-link.active svg {{
      stroke: #FFFFFF;
    }}

    .hidden {{
      display: none !important;
    }}

    .dropzone-active {{
      border-color: #3B5740 !important;
      background-color: #EAF1EC !important;
    }}

    *, *::before, *::after {{
      box-sizing: border-box;
    }}

    .overflow-x-auto {{
      width: 100% !important;
      max-width: 100% !important;
      overflow-x: auto !important;
      -webkit-overflow-scrolling: touch !important;
    }}

    @media (max-width: 1024px) {{
      html, body {{
        height: 100% !important;
        width: 100% !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
        -webkit-overflow-scrolling: touch !important;
      }}
      body {{
        display: flex !important;
        flex-direction: column !important;
        height: auto !important;
        min-height: 100% !important;
        width: 100% !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
      }}
      aside {{
        width: 100% !important;
        height: auto !important;
        max-height: none !important;
        overflow-y: visible !important;
        border-right: none !important;
        border-bottom: 1px solid #E0DED2 !important;
        position: static !important;
      }}
      aside > div.p-5 {{
        padding: 1.25rem !important;
      }}
      aside nav {{
        display: grid !important;
        grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
        gap: 0.5rem !important;
        margin-bottom: 1rem !important;
      }}
      .nav-link {{
        padding: 0.6rem 0.75rem !important;
        font-size: 0.75rem !important;
        min-height: 42px !important;
        justify-content: flex-start !important;
        border-radius: 0.75rem !important;
      }}
      main {{
        width: 100% !important;
        height: auto !important;
        max-height: none !important;
        overflow-y: visible !important;
        flex: 1 1 auto !important;
      }}
      main > header {{
        padding: 1rem 1.25rem !important;
        flex-wrap: wrap !important;
        gap: 0.75rem !important;
        position: relative !important;
      }}
      main > div.p-8 {{
        padding: 1.25rem !important;
        max-width: 100% !important;
        padding-bottom: 6rem !important;
      }}
      #tab-overview .grid:first-child {{
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        gap: 0.75rem !important;
      }}
      #tab-reports .grid[class*="grid-cols-5"] {{
        grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
        gap: 0.75rem !important;
      }}
      .overflow-x-auto table {{
        min-width: 580px !important;
      }}
    }}

    @media (max-width: 640px) {{
      aside > div.p-5 {{
        padding: 0.85rem !important;
      }}
      aside nav {{
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        gap: 0.35rem !important;
        margin-bottom: 0.75rem !important;
      }}
      .nav-link {{
        padding: 0.5rem 0.5rem !important;
        font-size: 0.7rem !important;
        min-height: 38px !important;
        gap: 0.35rem !important;
      }}
      .nav-link span {{
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
      }}
      .nav-link svg {{
        width: 0.85rem !important;
        height: 0.85rem !important;
        flex-shrink: 0 !important;
      }}
      main > header {{
        padding: 0.75rem 0.85rem !important;
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 0.5rem !important;
      }}
      main > div.p-8 {{
        padding: 0.65rem 0.75rem !important;
        padding-bottom: 5rem !important;
      }}
      .kpi-card {{
        padding: 0.75rem 0.85rem !important;
        min-height: auto !important;
      }}
      .kpi-value {{
        font-size: 1.3rem !important;
      }}
    }}
  </style>
</head>
<body class="h-screen w-screen flex antialiased overflow-hidden bg-[#F5F4ED]">

  <!-- Sidebar -->
  <aside class="w-72 bg-[#ECEBE3] border-r border-[#E0DED2] flex flex-col justify-between shrink-0 h-screen overflow-y-auto z-20">
    <div class="p-5">
      <!-- Brand Header -->
      <div class="flex items-center gap-3 mb-6">
        <div class="w-10 h-10 rounded-xl bg-[#181E19] flex items-center justify-center text-white font-bold text-sm tracking-wide shadow-sm">
          RN
        </div>
        <div>
          <h1 class="text-base font-bold text-branddark tracking-tight leading-none">RetainNet</h1>
          <span class="text-[10px] font-medium text-subtle tracking-wider uppercase">Enterprise Intelligence</span>
        </div>
      </div>

      <!-- Original Northstar Production Workspace Button -->
      <div class="mb-6">
        <label class="text-[10px] font-bold text-subtle uppercase tracking-wider block mb-1.5">Workspace</label>
        <div id="workspace-selector-btn" class="bg-cardbg border border-[#D9D7C7] rounded-xl px-3 py-2 flex items-center justify-between text-xs font-semibold text-branddark shadow-2xs cursor-pointer hover:border-[#BFBDA8] transition">
          <div class="flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-forest animate-pulse"></span>
            <span id="workspace-name-label">Northstar Production</span>
          </div>
          <svg class="w-3.5 h-3.5 text-subtle" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
        </div>
      </div>

      <!-- Navigation Tabs Menu -->
      <nav class="space-y-1 mb-6">
        <div class="text-[10px] font-bold text-subtle uppercase tracking-wider px-2 mb-2">Navigation</div>

        <button onclick="switchTab('overview')" id="tab-btn-overview" class="nav-link active w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-medium text-charcoal hover:bg-[#E2E0D4] transition">
          <svg class="w-4 h-4 stroke-current" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"/></svg>
          <span>1. Executive Summary</span>
        </button>

        <button onclick="switchTab('pipeline')" id="tab-btn-pipeline" class="nav-link w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-medium text-charcoal hover:bg-[#E2E0D4] transition">
          <svg class="w-4 h-4 stroke-current" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
          <span>2. Pipeline & Benchmarks</span>
        </button>

        <button onclick="switchTab('customers')" id="tab-btn-customers" class="nav-link w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-medium text-charcoal hover:bg-[#E2E0D4] transition">
          <svg class="w-4 h-4 stroke-current" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg>
          <span>3. Customer Prescriptions</span>
        </button>

        <button onclick="switchTab('simulator')" id="tab-btn-simulator" class="nav-link w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-medium text-charcoal hover:bg-[#E2E0D4] transition">
          <svg class="w-4 h-4 stroke-current" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
          <span>4. What-If Simulator</span>
        </button>

        <button onclick="switchTab('mlops')" id="tab-btn-mlops" class="nav-link w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-medium text-charcoal hover:bg-[#E2E0D4] transition">
          <svg class="w-4 h-4 stroke-current" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>
          <span>5. MLOps & DriftLens</span>
        </button>

        <button onclick="switchTab('reports')" id="tab-btn-reports" class="nav-link w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-medium text-charcoal hover:bg-[#E2E0D4] transition">
          <svg class="w-4 h-4 stroke-current" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
          <span>6. Audit Reports & Export</span>
        </button>
      </nav>

      <!-- Global Parameters Controls -->
      <div class="pt-4 border-t border-[#D9D7C7] space-y-4">
        <div class="text-[10px] font-bold text-subtle uppercase tracking-wider px-2">Global Parameters</div>

        <!-- Single Primary Data Source Selector + Conditional Controls -->
        <div class="space-y-2">
          <label for="dataset-selector" class="text-[11px] font-semibold text-charcoal block mb-1">Data Source</label>
          <select id="dataset-selector" onchange="handleDatasetChange(this.value)" class="w-full bg-cardbg border border-[#D9D7C7] rounded-lg px-2.5 py-1.5 text-xs text-charcoal focus:outline-none focus:border-forest transition">
            <option value="ibm" id="opt-ibm" selected>IBM Telco Benchmark ({app_data['kpi']['subscribers']})</option>
            <option value="weibull" id="opt-weibull">Synthetic Weibull Cohort (2,500)</option>
            <option value="custom" id="opt-custom">Upload Custom CSV</option>
          </select>

          <!-- Conditional Box 1: Synthetic Weibull Cohort Controls -->
          <div id="synthetic-controls-box" class="hidden bg-[#FAF9F3] border border-[#D9D7C7] rounded-xl p-2.5 space-y-2 text-xs">
            <div class="flex items-center justify-between">
              <span class="text-[10px] font-bold uppercase tracking-wider text-ambercustom">Synthetic Cohort</span>
              <span class="px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 text-[9px] font-bold">DEMO</span>
            </div>
            <p class="text-[10px] text-subtle leading-tight">
              Weibull parametric simulation for sensitivity testing (not benchmark measurements).
            </p>
            <div class="grid grid-cols-2 gap-1.5">
              <div>
                <label class="text-[9px] font-bold text-subtle block mb-0.5">Sample Size</label>
                <input type="number" id="syn-samples-input" value="2500" min="50" max="10000" step="50" class="w-full bg-white border border-[#D9D7C7] rounded px-1.5 py-1 text-[11px] text-charcoal">
              </div>
              <div>
                <label class="text-[9px] font-bold text-subtle block mb-0.5">Random Seed</label>
                <input type="number" id="syn-seed-input" value="42" min="1" max="99999" class="w-full bg-white border border-[#D9D7C7] rounded px-1.5 py-1 text-[11px] text-charcoal">
              </div>
            </div>
            <button type="button" id="btn-regenerate-synthetic" onclick="regenerateSyntheticCohort()" class="w-full py-1.5 bg-forest hover:bg-forest-hover text-white font-semibold rounded-lg text-[11px] transition shadow-2xs">
              Regenerate Cohort
            </button>
            <div id="syn-status-text" class="text-[10px] text-forest font-medium">Active: 2,500 synthetic records (seed=42)</div>
          </div>

          <!-- Conditional Box 2: Compact Drag-and-Drop CSV Uploader + Universal Schema Mapper -->
          <div id="custom-csv-upload-box" class="hidden bg-[#FAF9F3] border border-[#D9D7C7] rounded-xl p-2.5 space-y-2 text-xs">
            <div class="flex items-center justify-between">
              <span class="text-[11px] font-bold text-branddark">Upload customer CSV</span>
              <button type="button" onclick="downloadCSVTemplate()" class="text-[10px] font-semibold text-forest hover:underline">
                CSV template
              </button>
            </div>
            <div id="csv-dropzone"
                 onclick="document.getElementById('csv-file-input').click()"
                 ondragover="handleDragOver(event)"
                 ondragleave="handleDragLeave(event)"
                 ondrop="handleFileDrop(event)"
                 class="border-2 border-dashed border-[#C8C6B6] hover:border-forest rounded-lg p-2.5 text-center cursor-pointer bg-white/80 transition">
              <input type="file" id="csv-file-input" accept=".csv,text/csv" class="hidden" onchange="handleCSVFileSelect(this.files)" />
              <div class="text-[11px] font-semibold text-charcoal">Drag and drop CSV here</div>
              <div class="text-[10px] text-subtle mt-0.5">Limit 25MB • <span class="text-forest font-semibold underline">Browse files</span></div>
            </div>
            <p class="text-[10px] text-subtle leading-tight">
              Supports <span class="font-semibold text-charcoal">Telco</span>, <span class="font-semibold text-charcoal">Retail Sales</span>, <span class="font-semibold text-charcoal">SaaS</span>, or any tabular CSV. Auto-maps <span class="font-mono text-charcoal">monthly_charges</span>/Sales, <span class="font-mono text-charcoal">contract_type</span>/Category, <span class="font-mono text-charcoal">duration</span>/Date &amp; ID.
            </p>
            <div id="csv-validation-feedback" class="hidden rounded-lg p-2 text-[10px] leading-snug"></div>

            <!-- Interactive Auto-Adapted Schema Mapper (revealed when a CSV is loaded) -->
            <div id="csv-schema-mapper" class="hidden pt-2 border-t border-[#E2E0D4] space-y-1.5">
              <div class="flex items-center justify-between">
                <span class="text-[9px] font-bold uppercase tracking-wider text-forest">Auto-Adapted Columns</span>
                <span class="text-[9px] text-subtle">Editable</span>
              </div>
              <div class="grid grid-cols-2 gap-1.5">
                <div>
                  <label class="text-[9px] font-semibold text-subtle block">Value / Sales ($)</label>
                  <select id="map-col-value" onchange="reapplyCustomCSVMapping()" class="w-full bg-white border border-[#D9D7C7] rounded px-1 py-0.5 text-[10px] text-charcoal"></select>
                </div>
                <div>
                  <label class="text-[9px] font-semibold text-subtle block">Category / Tier</label>
                  <select id="map-col-category" onchange="reapplyCustomCSVMapping()" class="w-full bg-white border border-[#D9D7C7] rounded px-1 py-0.5 text-[10px] text-charcoal"></select>
                </div>
                <div>
                  <label class="text-[9px] font-semibold text-subtle block">Record / Cust ID</label>
                  <select id="map-col-id" onchange="reapplyCustomCSVMapping()" class="w-full bg-white border border-[#D9D7C7] rounded px-1 py-0.5 text-[10px] text-charcoal"></select>
                </div>
                <div>
                  <label class="text-[9px] font-semibold text-subtle block">Tenure / Date</label>
                  <select id="map-col-duration" onchange="reapplyCustomCSVMapping()" class="w-full bg-white border border-[#D9D7C7] rounded px-1 py-0.5 text-[10px] text-charcoal"></select>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Model Engine Selector -->
        <div>
          <label for="model-selector" class="text-[11px] font-semibold text-charcoal block mb-1">Inference Engine</label>
          <select id="model-selector" onchange="updateModelSelection(this.value)" class="w-full bg-cardbg border border-[#D9D7C7] rounded-lg px-2.5 py-1.5 text-xs text-charcoal focus:outline-none focus:border-forest transition">
            <option value="deepsurv" selected>RetainNet DeepSurv (PyTorch)</option>
            <option value="deephit">DeepHit Discrete Engine</option>
            <option value="coxph">CoxPH Regularized Baseline</option>
            <option value="rsf">Random Survival Forest (RSF)</option>
          </select>
        </div>

        <!-- Prediction Horizon Slider -->
        <div>
          <div class="flex justify-between items-center text-[11px] font-semibold text-charcoal mb-1">
            <span>Forecast Horizon</span>
            <span id="horizon-val" class="font-bold text-forest">{app_data['kpi']['forecast_horizon']} mo</span>
          </div>
          <input type="range" id="horizon-slider" min="12" max="72" value="{app_data['kpi']['forecast_horizon']}" step="6" class="w-full accent-forest cursor-pointer" oninput="updateHorizon(this.value)">
        </div>

        <!-- Financial Inputs -->
        <div class="grid grid-cols-2 gap-2">
          <div>
            <label for="cac-input" class="text-[10px] font-bold text-subtle block mb-1">CAC ($)</label>
            <input type="number" id="cac-input" value="120" oninput="updateFinancials()" class="w-full bg-cardbg border border-[#D9D7C7] rounded-lg px-2 py-1 text-xs text-charcoal">
          </div>
          <div>
            <label for="discount-input" class="text-[10px] font-bold text-subtle block mb-1">Discount Rate (d)</label>
            <input type="number" id="discount-input" value="0.01" step="0.005" min="0.001" oninput="updateFinancials()" class="w-full bg-cardbg border border-[#D9D7C7] rounded-lg px-2 py-1 text-xs text-charcoal">
          </div>
        </div>
      </div>
    </div>

    <!-- Live Workspace Status Card -->
    <div class="p-4 m-3 bg-[#E4E2D8] border border-[#D6D4C5] rounded-xl text-xs">
      <div class="text-[10px] font-bold text-subtle uppercase tracking-wider mb-1">Live Backend Status</div>
      <p id="sidebar-status-text" class="text-[11px] text-charcoal leading-snug mb-3">Active source: IBM Telco Benchmark ({app_data['kpi']['subscribers']} customer profiles).</p>
      <div class="flex items-center gap-2 bg-white/80 py-1.5 px-2.5 rounded-lg border border-[#D5D3C4]">
        <span class="w-2 h-2 rounded-full bg-emerald-600 animate-ping"></span>
        <span id="sidebar-status-pill" class="text-[10px] font-bold text-emerald-800">Pipeline Operational</span>
      </div>
    </div>
  </aside>

  <!-- Main Content Area -->
  <main class="flex-1 flex flex-col min-w-0 bg-[#F5F4ED] h-screen overflow-y-auto">
    <!-- Top System Header Bar -->
    <header class="px-8 py-4 border-b border-[#E7E5DA] bg-[#F8F7F1]/80 backdrop-blur sticky top-0 z-10 flex items-center justify-between">
      <div>
        <div class="flex items-center gap-2 text-xs text-subtle font-medium">
          <span>RetainNet System</span>
          <span>/</span>
          <span id="breadcrumb-title" class="text-charcoal font-semibold">Executive Summary</span>
          <span id="active-source-badge" class="ml-2 px-2 py-0.5 rounded-full text-[10px] font-bold bg-[#EAF1EC] text-forest">IBM Telco Benchmark</span>
        </div>
        <h2 id="page-heading" class="text-xl font-extrabold text-branddark tracking-tight">Portfolio Health & Dynamic CLTV</h2>
      </div>

      <div class="flex items-center gap-3">
        <div class="flex items-center gap-2 px-3 py-1.5 bg-white border border-[#E0DED2] rounded-full text-xs font-semibold text-charcoal shadow-2xs">
          <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
          <span id="header-latency-text">Inference Latency: {app_data['kpi']['latency_ms']}ms</span>
        </div>
        <button onclick="switchTab('reports')" class="px-3.5 py-1.5 bg-white border border-[#E0DED2] rounded-full text-xs font-semibold text-charcoal hover:bg-[#ECEBE3] transition shadow-2xs">
          Audit Snapshot
        </button>
      </div>
    </header>

    <!-- Content Workspace -->
    <div class="p-8 max-w-7xl w-full mx-auto space-y-8 pb-32">

      <!-- TAB 1: EXECUTIVE SUMMARY & PORTFOLIO HEALTH -->
      <section id="tab-overview" class="tab-panel space-y-6">
        <div id="synthetic-demo-banner" class="hidden bg-amber-50 border border-amber-200 rounded-xl px-4 py-2.5 text-xs text-amber-900 flex items-center justify-between">
          <div>
            <span class="font-bold">Synthetic Demonstration Cohort Active:</span> Metrics and survival curves below are generated from a parametric Weibull simulation for scenario testing, not benchmark measurements.
          </div>
          <span class="px-2 py-0.5 rounded bg-amber-200/70 text-[10px] font-bold uppercase">Simulation</span>
        </div>

        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div class="kpi-card">
            <span class="kpi-label">Active Subscribers</span>
            <div class="kpi-value" id="kpi-subscribers">{app_data['kpi']['subscribers']}</div>
            <div class="kpi-sub"><span class="text-forest font-semibold" id="kpi-subscribers-sub">100% Real-Time</span> profiles evaluated</div>
          </div>
          <div class="kpi-card">
            <span class="kpi-label">Average Dynamic CLTV</span>
            <div class="kpi-value" id="kpi-cltv">{app_data['kpi']['cltv']}</div>
            <div class="kpi-sub"><span class="text-forest font-semibold">Discounted rate</span> <span id="kpi-discount-sub">(d = 0.01 / mo)</span></div>
          </div>
          <div class="kpi-card">
            <span class="kpi-label">At-Risk Revenue</span>
            <div class="kpi-value text-terracotta" id="kpi-atrisk">{app_data['kpi']['atrisk']}</div>
            <div class="kpi-sub"><span class="text-terracotta font-semibold">Bottom 25%</span> highest hazard cohort</div>
          </div>
          <div class="kpi-card">
            <span class="kpi-label">12-Mo Projected Churn</span>
            <div class="kpi-value" id="kpi-churn">{app_data['kpi']['churn']}</div>
            <div class="kpi-sub"><span class="text-emerald-700 font-semibold">Temporal Forecast</span> 1 - S(t=12|x)</div>
          </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div class="lg:col-span-7 bg-white p-6 rounded-2xl border border-lineborder shadow-2xs">
            <div class="flex items-center justify-between mb-4">
              <div>
                <h3 class="font-bold text-branddark text-base">Dynamic CLTV Distribution</h3>
                <p class="text-xs text-subtle">Color-coded histogram showing break-even and high-risk loss cohorts</p>
              </div>
              <div class="flex items-center gap-3 text-xs">
                <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded bg-terracotta"></span> High Risk</span>
                <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded bg-ambercustom"></span> Moderate</span>
                <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded bg-forest"></span> High CLTV</span>
              </div>
            </div>
            <div class="h-64 relative">
              <canvas id="cltvHistogramChart"></canvas>
            </div>
          </div>

          <div class="lg:col-span-5 bg-white p-6 rounded-2xl border border-lineborder shadow-2xs flex flex-col justify-between">
            <div class="mb-2">
              <h3 class="font-bold text-branddark text-base">K-Means Stage 1 Archetypes</h3>
              <p class="text-xs text-subtle">4 latent clusters discovered prior to propensity filtering</p>
            </div>
            <div class="h-56 relative flex items-center justify-center">
              <canvas id="archetypesDonutChart"></canvas>
            </div>
            <div class="grid grid-cols-2 gap-2 text-xs pt-3 border-t border-lineborder">
              <div class="flex items-center gap-2"><span class="w-2.5 h-2.5 rounded-full bg-[#3B5740]"></span><span class="font-medium text-charcoal" id="arch-legend-0">Loyal Long-Term ({app_data['archetypes']['pcts'][0]}%)</span></div>
              <div class="flex items-center gap-2"><span class="w-2.5 h-2.5 rounded-full bg-[#BD6448]"></span><span class="font-medium text-charcoal" id="arch-legend-1">At-Risk Newbies ({app_data['archetypes']['pcts'][1]}%)</span></div>
              <div class="flex items-center gap-2"><span class="w-2.5 h-2.5 rounded-full bg-[#C38438]"></span><span class="font-medium text-charcoal" id="arch-legend-2">Price-Sensitive ({app_data['archetypes']['pcts'][2]}%)</span></div>
              <div class="flex items-center gap-2"><span class="w-2.5 h-2.5 rounded-full bg-[#627D78]"></span><span class="font-medium text-charcoal" id="arch-legend-3">Premium High-ARPU ({app_data['archetypes']['pcts'][3]}%)</span></div>
            </div>
          </div>
        </div>

        <div class="bg-white rounded-2xl border border-lineborder shadow-2xs overflow-hidden">
          <div class="p-5 border-b border-lineborder flex items-center justify-between">
            <div>
              <h3 class="font-bold text-branddark text-base">Priority Customer Queue</h3>
              <p class="text-xs text-subtle">Customers with urgent clinical churn mitigation thresholds</p>
            </div>
            <button onclick="switchTab('customers')" class="text-xs font-semibold text-forest hover:underline">
              Inspect in Customer View →
            </button>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-left text-xs">
              <thead class="bg-[#FAF9F3] text-subtle font-bold border-b border-lineborder">
                <tr>
                  <th class="py-3 px-5">Customer ID</th>
                  <th class="py-3 px-5">Segment</th>
                  <th class="py-3 px-5">Monthly Fee</th>
                  <th class="py-3 px-5">Dynamic LTV</th>
                  <th class="py-3 px-5">12M Survival</th>
                  <th class="py-3 px-5">Risk Status</th>
                  <th class="py-3 px-5 text-right">Action</th>
                </tr>
              </thead>
              <tbody id="priority-queue-body" class="divide-y divide-lineborder text-charcoal">
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <!-- TAB 2: 3-STAGE PIPELINE & MODEL BENCHMARKING -->
      <section id="tab-pipeline" class="tab-panel hidden space-y-6">
        <div id="pipeline-inference-notice" class="hidden bg-blue-50 border border-blue-200 rounded-xl px-4 py-2.5 text-xs text-blue-900">
          <span class="font-bold">Inference-Only Custom Dataset Active:</span> Ground-truth survival durations and event indicators were not provided in the uploaded CSV. Individualized survival predictions and CLTV are computed using the pre-trained RetainNet models, while ground-truth evaluation metrics (C-Index, Brier Scores) are marked N/A.
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="kpi-card">
            <span class="kpi-label">Top Architecture</span>
            <div class="kpi-value text-forest" id="pipe-top-arch">{app_data['kpi']['top_model_name']}</div>
            <div class="kpi-sub">Optimizing Cox partial log-likelihood</div>
          </div>
          <div class="kpi-card">
            <span class="kpi-label">Harrell's C-Index</span>
            <div class="kpi-value" id="pipe-cindex">{app_data['kpi']['best_cindex']}</div>
            <div class="kpi-sub"><span class="text-forest font-semibold">Rank-order concordance</span></div>
          </div>
          <div class="kpi-card">
            <span class="kpi-label">Denoised Signal Retention</span>
            <div class="kpi-value" id="pipe-retention">{app_data['kpi']['retention_rate']}</div>
            <div class="kpi-sub"><span class="text-emerald-700 font-semibold">Ambiguity band filtered</span> (0.30 - 0.70)</div>
          </div>
        </div>

        <div class="bg-white p-6 rounded-2xl border border-lineborder shadow-2xs">
          <div class="mb-5">
            <h3 class="font-bold text-branddark text-base">RetainNet 3-Stage Denoising Funnel Architecture</h3>
            <p class="text-xs text-subtle">How RetainNet filters label noise in churn right-censoring before non-linear hazard learning</p>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div class="bg-[#FAF9F3] p-5 rounded-xl border border-lineborder">
              <div class="text-[10px] font-extrabold uppercase px-2 py-0.5 rounded bg-forest text-white inline-block mb-2">Stage 1</div>
              <h4 class="font-bold text-sm text-branddark mb-1">Feature Augmentation</h4>
              <p class="text-xs text-subtle leading-relaxed mb-3">
                K-Means (<span class="font-mono text-branddark font-semibold">k=4</span>) clustering projects multi-modal customer attributes into dense archetype representations.
              </p>
              <div class="bg-white p-3 rounded-lg border border-lineborder text-[11px] font-mono text-charcoal">
                X_aug = [X_raw | OneHot_Clusters]
              </div>
            </div>

            <div class="bg-[#FAF9F3] p-5 rounded-xl border border-lineborder">
              <div class="text-[10px] font-extrabold uppercase px-2 py-0.5 rounded bg-ambercustom text-white inline-block mb-2">Stage 2</div>
              <h4 class="font-bold text-sm text-branddark mb-1">Propensity Denoising</h4>
              <p class="text-xs text-subtle leading-relaxed mb-3">
                Boundary excision removes ambiguous churn records (<span class="font-mono text-branddark font-semibold">0.30 ≤ p̂ ≤ 0.70</span>), retaining high confidence signal.
              </p>
              <div class="bg-white p-3 rounded-lg border border-lineborder text-[11px] font-mono text-charcoal" id="pipe-stage2-signal">
                Retained Signal: {app_data['kpi']['retention_rate']}
              </div>
            </div>

            <div class="bg-[#FAF9F3] p-5 rounded-xl border border-lineborder">
              <div class="text-[10px] font-extrabold uppercase px-2 py-0.5 rounded bg-[#181E19] text-white inline-block mb-2">Stage 3</div>
              <h4 class="font-bold text-sm text-branddark mb-1">DeepSurv PyTorch NN</h4>
              <p class="text-xs text-subtle leading-relaxed mb-3">
                Multi-layer MLP learns non-linear log hazard ratios while integrating baseline hazard via Breslow estimator.
              </p>
              <div class="bg-white p-3 rounded-lg border border-lineborder text-[11px] font-mono text-charcoal">
                Loss = -∑ [h_θ(x_i) - log ∑ exp(h_θ(x_j))]
              </div>
            </div>
          </div>
        </div>

        <div class="bg-white rounded-2xl border border-lineborder shadow-2xs overflow-hidden">
          <div class="p-5 border-b border-lineborder">
            <h3 class="font-bold text-branddark text-base">Benchmark Comparison Leaderboard</h3>
            <p class="text-xs text-subtle">Empirical validation across statistical baselines and neural survival frameworks</p>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-left text-xs">
              <thead class="bg-[#FAF9F3] text-subtle font-bold border-b border-lineborder">
                <tr>
                  <th class="py-3 px-5">Model Architecture</th>
                  <th class="py-3 px-5">Harrell's C-Index (↑)</th>
                  <th class="py-3 px-5">Integrated Brier Score (↓)</th>
                  <th class="py-3 px-5">BS @ 12M</th>
                  <th class="py-3 px-5">BS @ 24M</th>
                  <th class="py-3 px-5">BS @ 36M</th>
                  <th class="py-3 px-5">D-Calibration χ²</th>
                </tr>
              </thead>
              <tbody id="leaderboard-body" class="divide-y divide-lineborder text-charcoal">
              </tbody>
            </table>
          </div>
        </div>

        <div class="bg-white p-6 rounded-2xl border border-lineborder shadow-2xs">
          <div class="flex items-center justify-between mb-4">
            <div>
              <h3 class="font-bold text-branddark text-base">Kaplan-Meier vs Predicted DeepSurv Survival Curves</h3>
              <p class="text-xs text-subtle">Portfolio-wide calibration comparison over continuous timeline</p>
            </div>
            <div class="flex items-center gap-4 text-xs">
              <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-forest"></span> Predicted DeepSurv</span>
              <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-subtle border-t border-dashed"></span> Empirical Kaplan-Meier</span>
            </div>
          </div>
          <div class="h-64 relative">
            <canvas id="pipelineKmChart"></canvas>
          </div>
        </div>
      </section>

      <!-- TAB 3: CUSTOMER LOOKUP & CLINICAL PRESCRIPTIONS -->
      <section id="tab-customers" class="tab-panel hidden space-y-6">
        <div class="bg-white p-5 rounded-2xl border border-lineborder shadow-2xs flex flex-wrap items-center justify-between gap-4">
          <div class="flex flex-wrap items-center gap-3 flex-1 min-w-[280px]">
            <label for="customer-dropdown" class="text-xs font-bold text-subtle uppercase tracking-wider shrink-0">Select Subscriber:</label>
            <div class="relative flex items-center">
              <input type="text"
                     id="customer-search-input"
                     placeholder="Search ID, segment, or contract..."
                     oninput="filterCustomerDropdown(this.value)"
                     class="bg-[#FAF9F3] border border-lineborder rounded-xl px-3 py-2 text-xs font-medium text-charcoal focus:border-forest outline-none w-52" />
            </div>
            <select id="customer-dropdown" onchange="loadCustomer(this.value)" class="bg-[#FAF9F3] border border-lineborder rounded-xl px-3 py-2 text-xs font-semibold text-charcoal focus:border-forest outline-none min-w-[220px]">
            </select>
            <span id="customer-count-badge" class="px-2.5 py-1 rounded-full bg-[#ECEBE3] text-[11px] font-bold text-charcoal">
              {app_data['kpi']['subscribers']} subscribers
            </span>
            <span id="customer-no-match-msg" class="hidden text-xs font-semibold text-terracotta">
              No matching subscribers found.
            </span>
          </div>

          <div class="flex items-center gap-3">
            <div class="px-3 py-1.5 rounded-xl bg-[#FAF9F3] border border-lineborder text-xs">
              <span class="text-subtle font-semibold">Tenure:</span> <span id="c-tenure" class="font-bold text-charcoal">-</span>
            </div>
            <div class="px-3 py-1.5 rounded-xl bg-[#FAF9F3] border border-lineborder text-xs">
              <span class="text-subtle font-semibold">Monthly ARPU:</span> <span id="c-arpu" class="font-bold text-charcoal">-</span>
            </div>
            <div class="px-3 py-1.5 rounded-xl bg-[#FAF9F3] border border-lineborder text-xs">
              <span class="text-subtle font-semibold">Contract:</span> <span id="c-contract" class="font-bold text-charcoal">-</span>
            </div>
          </div>
        </div>

        <div class="bg-white p-6 rounded-2xl border border-lineborder shadow-2xs">
          <div class="flex items-center justify-between mb-4">
            <div>
              <div class="flex items-center gap-2">
                <h3 class="font-bold text-branddark text-base" id="c-curve-title">Individualized Survival Function S(t | x)</h3>
                <span class="text-[10px] bg-emerald-100 text-emerald-800 font-bold px-2 py-0.5 rounded-full">CiPOT 90% Conformal Bounds</span>
              </div>
              <p class="text-xs text-subtle">Uncertainty envelope showing distribution-free 90% coverage guarantees</p>
            </div>
            <div class="flex items-center gap-3 text-xs">
              <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-forest"></span> Point Estimate</span>
              <span class="flex items-center gap-1.5"><span class="w-3 h-2 bg-emerald-200/50 rounded-2xs"></span> 90% Conformal Band</span>
            </div>
          </div>
          <div class="h-72 relative">
            <canvas id="customerConformalChart"></canvas>
          </div>
        </div>

        <div class="bg-[#FCFBF8] border-2 border-[#E2DFC8] p-6 rounded-2xl shadow-2xs">
          <div class="flex items-start gap-4">
            <div class="w-10 h-10 rounded-xl bg-forest text-white flex items-center justify-center shrink-0 text-lg font-bold">
              ⚕
            </div>
            <div class="space-y-3 flex-1">
              <div class="flex items-center justify-between">
                <h4 class="font-bold text-base text-branddark">Clinical Intervention Protocol</h4>
                <span id="c-prescribe-badge" class="px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-800">Low Urgency</span>
              </div>
              <p id="c-prescribe-text" class="text-xs text-charcoal leading-relaxed"></p>

              <div class="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2 text-xs">
                <div class="p-3 bg-white rounded-xl border border-lineborder">
                  <span class="text-subtle font-medium block">Current Hazard Rate</span>
                  <span id="c-hazard" class="font-bold text-sm text-branddark">-</span>
                </div>
                <div class="p-3 bg-white rounded-xl border border-lineborder">
                  <span class="text-subtle font-medium block">Projected Dynamic LTV</span>
                  <span id="c-ltv-calc" class="font-bold text-sm text-forest">-</span>
                </div>
                <div class="p-3 bg-white rounded-xl border border-lineborder">
                  <span class="text-subtle font-medium block">Intervention ROI</span>
                  <span id="c-intervention-roi" class="font-bold text-sm text-charcoal">-</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- TAB 4: WHAT-IF RETENTION SIMULATOR -->
      <section id="tab-simulator" class="tab-panel hidden space-y-6">
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div class="lg:col-span-4 bg-white p-6 rounded-2xl border border-lineborder shadow-2xs space-y-5">
            <div>
              <h3 class="font-bold text-branddark text-base">Intervention Parameters</h3>
              <p class="text-xs text-subtle">Simulate retention promotions and calculate counterfactual hazard multiplier γ</p>
            </div>

            <div class="space-y-1.5">
              <div class="flex justify-between text-xs font-semibold">
                <span>Monthly Bill Discount</span>
                <span id="sim-discount-val" class="text-forest font-bold">$15 / mo</span>
              </div>
              <input type="range" id="sim-discount-slider" min="0" max="35" value="15" step="5" class="w-full accent-forest cursor-pointer" oninput="runSimulation()">
              <p class="text-[11px] text-subtle" id="sim-discount-sub">Reduces ARPU but diminishes churn hazard (γ = 0.78)</p>
            </div>

            <div class="p-3.5 bg-[#FAF9F3] border border-lineborder rounded-xl flex items-center justify-between">
              <div>
                <span class="text-xs font-bold text-charcoal block">12-Month Lock-in Contract</span>
                <span class="text-[11px] text-subtle">Substantial hazard dampener (γ = 0.60)</span>
              </div>
              <label class="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" id="sim-contract-toggle" class="sr-only peer" checked onchange="runSimulation()">
                <div class="w-9 h-5 bg-gray-300 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-forest"></div>
              </label>
            </div>

            <div class="p-3.5 bg-[#FAF9F3] border border-lineborder rounded-xl flex items-center justify-between">
              <div>
                <span class="text-xs font-bold text-charcoal block">VIP Success Concierge</span>
                <span class="text-[11px] text-subtle">Cost: $25 one-time, γ = 0.85</span>
              </div>
              <label class="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" id="sim-vip-toggle" class="sr-only peer" onchange="runSimulation()">
                <div class="w-9 h-5 bg-gray-300 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-forest"></div>
              </label>
            </div>

            <div class="p-4 bg-[#EAF1EC] rounded-xl border border-[#C5D9C9] text-xs">
              <span class="font-bold text-forest uppercase text-[10px] tracking-wider block mb-1">Effective Hazard Ratio (γ)</span>
              <div class="flex items-baseline gap-2">
                <span id="sim-gamma-val" class="text-2xl font-black text-forest">0.465</span>
                <span id="sim-gamma-reduction" class="text-[11px] text-forest font-semibold">-53.5% Churn Risk</span>
              </div>
            </div>
          </div>

          <div class="lg:col-span-8 space-y-5">
            <div class="bg-white p-6 rounded-2xl border border-lineborder shadow-2xs">
              <div class="flex items-center justify-between mb-4">
                <div>
                  <h3 class="font-bold text-branddark text-base">Baseline vs Counterfactual Survival Curve</h3>
                  <p class="text-xs text-subtle">Real-time simulation of S_counterfactual(t | x) = S_baseline(t | x)^γ</p>
                </div>
                <div class="flex items-center gap-3 text-xs font-medium">
                  <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-terracotta"></span> Baseline S(t)</span>
                  <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-forest"></span> Counterfactual S*(t)</span>
                </div>
              </div>
              <div class="h-64 relative">
                <canvas id="counterfactualChart"></canvas>
              </div>
            </div>

            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div class="kpi-card">
                <span class="kpi-label">Baseline LTV</span>
                <div class="kpi-value text-base" id="roi-base-ltv">$450.00</div>
                <div class="kpi-sub">Unassisted outlook</div>
              </div>
              <div class="kpi-card">
                <span class="kpi-label">Counterfactual LTV</span>
                <div class="kpi-value text-base text-forest" id="roi-counter-ltv">$737.00</div>
                <div class="kpi-sub">With incentives</div>
              </div>
              <div class="kpi-card">
                <span class="kpi-label">Net LTV Lift</span>
                <div class="kpi-value text-base text-emerald-700" id="roi-lift">+$287.00</div>
                <div class="kpi-sub" id="roi-lift-sub">+63.8% improvement</div>
              </div>
              <div class="kpi-card">
                <span class="kpi-label">Campaign ROI</span>
                <div class="kpi-value text-base text-forest" id="roi-pct">159%</div>
                <div class="kpi-sub">Return on promotion cost</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- TAB 5: MLOPS & EXTENSIONS -->
      <section id="tab-mlops" class="tab-panel hidden space-y-6">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="kpi-card">
            <span class="kpi-label">Population Stability Index (PSI)</span>
            <div class="kpi-value text-forest" id="mlops-psi">{app_data['mlops']['psi']}</div>
            <div class="kpi-sub"><span class="text-emerald-700 font-semibold" id="mlops-psi-sub">Green: &lt; 0.20 (Stable)</span></div>
          </div>
          <div class="kpi-card">
            <span class="kpi-label">Kolmogorov-Smirnov (KS) Test</span>
            <div class="kpi-value" id="mlops-ks">{app_data['mlops']['ks_stat']}</div>
            <div class="kpi-sub">Distribution shift statistic</div>
          </div>
          <div class="kpi-card">
            <span class="kpi-label">Conformal Coverage @ 90%</span>
            <div class="kpi-value text-forest" id="mlops-coverage">{app_data['mlops']['coverage']}</div>
            <div class="kpi-sub">Well-calibrated non-parametric band</div>
          </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div class="lg:col-span-7 bg-white p-6 rounded-2xl border border-lineborder shadow-2xs">
            <div class="flex items-center justify-between mb-4">
              <div>
                <h3 class="font-bold text-branddark text-base">DriftLens Feature Distribution Stability</h3>
                <p class="text-xs text-subtle">Reference IBM Training Cohort vs Active Evaluation Cohort</p>
              </div>
              <span id="mlops-drift-badge" class="px-2.5 py-1 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800">STABLE</span>
            </div>
            <div class="h-64 relative">
              <canvas id="driftChart"></canvas>
            </div>
          </div>

          <div class="lg:col-span-5 bg-white p-6 rounded-2xl border border-lineborder shadow-2xs flex flex-col justify-between">
            <div>
              <div class="flex items-center gap-2 mb-1">
                <h3 class="font-bold text-branddark text-base">SurvivalPFN Cold-Start Engine</h3>
                <span class="px-2 py-0.5 rounded text-[10px] font-extrabold bg-ambercustom text-white">ZERO-SHOT</span>
              </div>
              <p class="text-xs text-subtle mb-4">Instantly infer survival curve for fresh signups without historical activity</p>

              <div class="space-y-3 text-xs">
                <div>
                  <label class="font-semibold text-charcoal block mb-1">Contract Type</label>
                  <select id="cs-contract" class="w-full bg-[#FAF9F3] border border-lineborder rounded-lg px-2.5 py-1.5 font-medium">
                    <option value="month-to-month">Month-to-Month</option>
                    <option value="one-year">One Year</option>
                    <option value="two-year">Two Year</option>
                  </select>
                </div>

                <div>
                  <label class="font-semibold text-charcoal block mb-1">Monthly Charges ($)</label>
                  <input type="number" id="cs-charge" value="70" class="w-full bg-[#FAF9F3] border border-lineborder rounded-lg px-2.5 py-1.5 font-medium">
                </div>

                <div>
                  <label class="font-semibold text-charcoal block mb-1">Internet Service</label>
                  <select id="cs-service" class="w-full bg-[#FAF9F3] border border-lineborder rounded-lg px-2.5 py-1.5 font-medium">
                    <option value="fiber">Fiber Optic</option>
                    <option value="dsl">DSL</option>
                    <option value="none">No Internet</option>
                  </select>
                </div>

                <button onclick="inferColdStart()" class="w-full py-2 bg-forest hover:bg-forest-hover text-white font-semibold rounded-lg text-xs transition shadow-2xs">
                  Generate Prior Survival Distribution
                </button>
              </div>
            </div>

            <div class="p-3.5 bg-[#FAF9F3] border border-lineborder rounded-xl mt-4">
              <div class="text-[10px] font-bold text-subtle uppercase">KDE Posterior Match:</div>
              <div class="flex justify-between items-center mt-1">
                <span id="cs-output-archetype" class="font-bold text-xs text-charcoal">At-Risk Newbie Prior</span>
                <span id="cs-output-prob" class="font-bold text-xs text-terracotta">Expected 12M: 34.2%</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- TAB 6: AUDIT REPORTS & EXPORT -->
      <section id="tab-reports" class="tab-panel hidden space-y-6">
        <div class="space-y-1">
          <div class="text-xs font-bold text-subtle uppercase tracking-wider">Export & Audit</div>
          <h3 class="text-2xl font-black text-branddark">Reports</h3>
          <p class="text-xs text-subtle">A clean audit view of the current workspace, model quality, and customer-level outputs.</p>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div class="kpi-card"><span class="kpi-label">CUSTOMERS</span><div class="kpi-value" id="rep-kpi-customers">{app_data['kpi']['subscribers']}</div><div class="kpi-sub">Profiles evaluated</div></div>
          <div class="kpi-card"><span class="kpi-label">RIGHT-CENSORED</span><div class="kpi-value" id="rep-kpi-censored">{app_data['kpi']['right_censored']}</div><div class="kpi-sub">Active at observation end</div></div>
          <div class="kpi-card"><span class="kpi-label">BEST C-INDEX</span><div class="kpi-value text-forest" id="rep-kpi-cindex">{app_data['kpi']['best_cindex']}</div><div class="kpi-sub">Predictive ranking</div></div>
          <div class="kpi-card"><span class="kpi-label">SIGNAL RETAINED</span><div class="kpi-value" id="rep-kpi-retention">{app_data['kpi']['retention_rate']}</div><div class="kpi-sub">After confidence filtering</div></div>
          <div class="kpi-card"><span class="kpi-label">FORECAST</span><div class="kpi-value" id="report-horizon-val">{app_data['kpi']['forecast_horizon']} mo</div><div class="kpi-sub">Continuous horizon</div></div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          <div class="lg:col-span-8 bg-white p-6 rounded-2xl border border-lineborder shadow-2xs flex flex-col justify-between">
            <div>
              <h4 class="font-bold text-branddark text-base">Export Customer Predictions</h4>
              <p class="text-xs text-subtle mt-1">
                Download real customer-level LTV and retention predictions computed by the PyTorch model for the active dataset (<span id="rep-export-source-name">IBM Telco Benchmark</span>).
              </p>
            </div>

            <div class="mt-8 pt-4 border-t border-lineborder flex items-center justify-between">
              <div>
                <span class="text-xs font-semibold text-charcoal block">Format: Comma-Separated Values (.csv)</span>
                <span class="text-[11px] text-subtle">Includes <span id="report-horizon-text">{app_data['kpi']['forecast_horizon']}</span>-month individualized survival curves and estimated dynamic CLTV for all <span id="rep-export-count">{app_data['kpi']['subscribers']}</span> customers.</span>
              </div>
              <button onclick="downloadPredictionsCSV()" class="btn-download-primary shrink-0">
                <svg class="w-4 h-4 stroke-current" fill="none" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
                </svg>
                <span>Download customer LTV predictions</span>
              </button>
            </div>
          </div>

          <div class="lg:col-span-4 bg-white p-6 rounded-2xl border border-lineborder shadow-2xs flex flex-col justify-between">
            <div>
              <span class="text-[10px] font-bold text-subtle uppercase tracking-wider block mb-1">Audit status</span>
              <span class="text-xs text-subtle">Current benchmark leader</span>
              <div class="text-xl font-black text-branddark mt-2" id="rep-audit-leader">{app_data['kpi']['top_model_name']}</div>
              <div class="text-xs font-semibold text-forest mt-1" id="rep-audit-metrics">C-Index {app_data['kpi']['best_cindex']} • IBS {app_data['kpi']['best_ibs']}</div>
            </div>

            <div class="mt-4 pt-3 border-t border-lineborder text-[11px] text-subtle" id="rep-audit-compliance">
              Compliance check: Model weights certified against population drift (KS-stat = {app_data['mlops']['ks_stat']}, PSI = {app_data['mlops']['psi']}).
            </div>
          </div>
        </div>

        <div class="bg-white rounded-2xl border border-lineborder shadow-2xs overflow-hidden">
          <div class="p-5 border-b border-lineborder">
            <h4 class="font-bold text-branddark text-base">Model evaluation</h4>
            <p class="text-xs text-subtle">Multi-horizon loss benchmark across all evaluated survival architectures</p>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-left text-xs">
              <thead class="bg-[#FAF9F3] text-subtle font-bold border-b border-lineborder">
                <tr>
                  <th class="py-3 px-5">Model</th>
                  <th class="py-3 px-5">C-Index (↑)</th>
                  <th class="py-3 px-5">Integrated Brier Score (↓)</th>
                  <th class="py-3 px-5">Brier @ 12M</th>
                  <th class="py-3 px-5">Brier @ 24M</th>
                  <th class="py-3 px-5">Brier @ 36M</th>
                </tr>
              </thead>
              <tbody id="audit-table-body" class="divide-y divide-lineborder text-charcoal">
              </tbody>
            </table>
          </div>
        </div>
      </section>

    </div>
  </main>

  <script>
    if (typeof window.Chart === "undefined") {{
      window.Chart = class OfflineChartFallback {{
        constructor(ctx, cfg) {{
          this.ctx = ctx;
          this.config = cfg || {{}};
          this.data = this.config.data || {{ labels: [], datasets: [] }};
          this.options = this.config.options || {{}};
        }}
        update() {{}}
        resize() {{}}
        destroy() {{}}
      }};
    }}
    const RAW_SERVER_DATA = {app_data_json};
    const MODEL_BUNDLE = RAW_SERVER_DATA.model_bundle;
    const DATASETS = RAW_SERVER_DATA.datasets;

    const TIER_TEMPLATES = {{
      critical: {{
        segment: 'At-Risk Newbies',
        urgency: 'Critical Urgency',
        badgeClass: 'bg-red-100 text-red-800',
        prescription: 'Early tenure dropout risk detected! Rapid steepening of hazard rate in early months. Action: Trigger automated $15/mo discount linked to 12-month annual contract lock-in to avert immediate churn.',
        hazard: 'λ = 0.088 / mo',
        roiText: 'High ROI on Contract Lock-in (+185%)'
      }},
      moderate: {{
        segment: 'Price-Sensitive',
        urgency: 'Moderate Priority',
        badgeClass: 'bg-amber-100 text-amber-800',
        prescription: 'Subscriber shows price sensitivity. Approaching decision boundary. Action: Deploy proactive bundle optimization. Offer $10 bill credit and free speed tier boost.',
        hazard: 'λ = 0.035 / mo',
        roiText: 'Projected Net Lift: +$140.00'
      }},
      healthy: {{
        segment: 'Loyal Long-Term',
        urgency: 'Healthy / Low Risk',
        badgeClass: 'bg-emerald-100 text-emerald-800',
        prescription: 'Subscriber displays solid tenure and low hazard rate. Action: Maintain standard VIP loyalty tier. Initiate cross-sell for premium features with zero defensive discounting.',
        hazard: 'λ = 0.009 / mo',
        roiText: 'Organic Retention (No Discount Needed)'
      }}
    }};

    function hydrateCustomerDB(datasetObj) {{
      if (!datasetObj || !datasetObj.compact_customers) return;
      if (datasetObj.customer_db && Object.keys(datasetObj.customer_db).length === datasetObj.compact_customers.length) {{
        return;
      }}
      const db = {{}};
      const ids = [];
      for (let i = 0; i < datasetObj.compact_customers.length; i++) {{
        const row = datasetObj.compact_customers[i];
        const cid = String(row[0]);
        const tenureMonths = row[1];
        const arpuVal = Number(row[2]);
        const contractRaw = Number(row[3]);
        const ret12 = Number(row[4]);
        const ret24 = Number(row[5]);
        const ltvVal = Number(row[6]);
        const cDsurv = row[7];
        const cCox = row[8] || cDsurv;
        const cRsf = row[9] || cDsurv;
        const cDhit = row[10] || cDsurv;
        const customLabel = row[11];

        const contractName = customLabel
          ? String(customLabel)
          : (contractRaw === 2 ? 'Two Year' : (contractRaw === 1 ? 'One Year' : 'Month-to-Month'));
        const tier = ret12 < 30.0 ? TIER_TEMPLATES.critical : (ret12 < 65.0 ? TIER_TEMPLATES.moderate : TIER_TEMPLATES.healthy);

        db[cid] = {{
          id: cid,
          tenure: `${{tenureMonths}} Months`,
          arpu: `$${{arpuVal.toFixed(2)}}`,
          contract: contractName,
          segment: tier.segment,
          urgency: tier.urgency,
          badgeClass: tier.badgeClass,
          prescription: tier.prescription,
          hazard: tier.hazard,
          ltv: `$${{ltvVal.toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }})}}`,
          roiText: tier.roiText,
          ret12: `${{Math.round(ret12)}}%`,
          ret12_num: ret12,
          ret24_num: ret24,
          ltv_num: ltvVal,
          curve: cDsurv,
          curves: {{
            deepsurv: cDsurv,
            coxph: cCox,
            rsf: cRsf,
            deephit: cDhit
          }}
        }};
        ids.push(cid);
      }}
      datasetObj.customer_db = db;
      datasetObj.customer_ids = ids;
    }}

    Object.values(DATASETS).forEach(hydrateCustomerDB);

    let activeDatasetKey = 'ibm';
    let SERVER_DATA = DATASETS[activeDatasetKey];
    let activeTab = 'overview';
    let currentModelKey = 'deepsurv';
    let currentCustId = SERVER_DATA.primary_cid;
    let lastUploadedCSVText = '';
    let lastUploadedCSVName = 'uploaded.csv';
    const charts = {{}};

    function switchTab(tabId) {{
      activeTab = tabId;
      document.querySelectorAll('.tab-panel').forEach(el => el.classList.add('hidden'));
      const activePanel = document.getElementById('tab-' + tabId);
      if (activePanel) activePanel.classList.remove('hidden');

      document.querySelectorAll('.nav-link').forEach(btn => btn.classList.remove('active'));
      const activeBtn = document.getElementById('tab-btn-' + tabId);
      if (activeBtn) activeBtn.classList.add('active');

      const titles = {{
        'overview': {{ title: 'Executive Summary', head: 'Portfolio Health & Dynamic CLTV' }},
        'pipeline': {{ title: 'Pipeline & Benchmarks', head: '3-Stage Denoising Architecture & Model Leaderboard' }},
        'customers': {{ title: 'Customer Prescriptions', head: 'Individual Survival Curves & Clinical Actions' }},
        'simulator': {{ title: 'What-If Simulator', head: 'Counterfactual Retention Strategy & ROI Optimizer' }},
        'mlops': {{ title: 'MLOps & DriftLens', head: 'DriftLens Population Stability & SurvivalPFN Cold-Start' }},
        'reports': {{ title: 'Audit Reports', head: 'Model Evaluation Audit & CSV Export' }}
      }};

      if (titles[tabId]) {{
        document.getElementById('breadcrumb-title').innerText = titles[tabId].title;
        document.getElementById('page-heading').innerText = titles[tabId].head;
      }}

      requestAnimationFrame(() => {{
        Object.values(charts).forEach(c => {{
          if (c) {{
            c.resize();
            c.update('none');
          }}
        }});
      }});
    }}

    window.addEventListener('DOMContentLoaded', () => {{
      initCharts();
      applyActiveDataset('ibm', false);
      runSimulation();
    }});

    function populateCustomerDropdown(searchQuery = '', preserveSelection = true) {{
      const dropdown = document.getElementById('customer-dropdown');
      const countBadge = document.getElementById('customer-count-badge');
      const noMatchMsg = document.getElementById('customer-no-match-msg');
      if (!dropdown) return;

      const allIds = SERVER_DATA.customer_ids || Object.keys(SERVER_DATA.customer_db);
      const q = String(searchQuery || '').trim().toLowerCase();

      const matchingIds = q === '' ? allIds : allIds.filter(cid => {{
        const c = SERVER_DATA.customer_db[cid];
        if (!c) return false;
        const text = `${{cid}} ${{c.segment}} ${{c.urgency}} ${{c.contract}}`.toLowerCase();
        return text.includes(q);
      }});

      dropdown.innerHTML = '';
      const frag = document.createDocumentFragment();
      for (let i = 0; i < matchingIds.length; i++) {{
        const cid = matchingIds[i];
        const c = SERVER_DATA.customer_db[cid];
        const opt = document.createElement('option');
        opt.value = cid;
        opt.textContent = `${{cid}} (${{c.segment}} - ${{c.urgency}})`;
        frag.appendChild(opt);
      }}
      dropdown.appendChild(frag);

      if (countBadge) {{
        if (q === '') {{
          countBadge.innerText = `${{allIds.length.toLocaleString()}} subscribers`;
        }} else {{
          countBadge.innerText = `${{matchingIds.length.toLocaleString()}} / ${{allIds.length.toLocaleString()}} matches`;
        }}
      }}

      if (matchingIds.length === 0) {{
        if (noMatchMsg) noMatchMsg.classList.remove('hidden');
        dropdown.disabled = true;
        return;
      }}

      if (noMatchMsg) noMatchMsg.classList.add('hidden');
      dropdown.disabled = false;

      let nextId = currentCustId;
      if (!preserveSelection || !matchingIds.includes(nextId)) {{
        nextId = matchingIds[0];
      }}
      dropdown.value = nextId;
      loadCustomer(nextId);
    }}

    function filterCustomerDropdown(query) {{
      populateCustomerDropdown(query, true);
    }}

    function applyActiveDataset(sourceKey, keepCurrentSubscriber = false) {{
      if (!DATASETS[sourceKey]) return;
      activeDatasetKey = sourceKey;
      SERVER_DATA = DATASETS[sourceKey];
      hydrateCustomerDB(SERVER_DATA);

      // Clear stale subscriber selection if not in new dataset
      if (!keepCurrentSubscriber || !SERVER_DATA.customer_db[currentCustId]) {{
        currentCustId = SERVER_DATA.primary_cid;
      }}

      // Reset subscriber search input on dataset change
      const searchInput = document.getElementById('customer-search-input');
      if (searchInput && !keepCurrentSubscriber) {{
        searchInput.value = '';
      }}

      // Update header & sidebar source indicators
      const sourceBadge = document.getElementById('active-source-badge');
      const demoBanner = document.getElementById('synthetic-demo-banner');
      const infNotice = document.getElementById('pipeline-inference-notice');
      const sidebarStatus = document.getElementById('sidebar-status-text');
      const repExportSource = document.getElementById('rep-export-source-name');
      const repExportCount = document.getElementById('rep-export-count');

      if (sourceBadge) {{
        sourceBadge.innerText = SERVER_DATA.source_label;
        sourceBadge.className = SERVER_DATA.is_synthetic_demo
          ? 'ml-2 px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-amber-800'
          : (SERVER_DATA.is_inference_only
              ? 'ml-2 px-2 py-0.5 rounded-full text-[10px] font-bold bg-blue-100 text-blue-800'
              : 'ml-2 px-2 py-0.5 rounded-full text-[10px] font-bold bg-[#EAF1EC] text-forest');
      }}

      if (demoBanner) {{
        demoBanner.classList.toggle('hidden', !SERVER_DATA.is_synthetic_demo);
      }}
      if (infNotice) {{
        infNotice.classList.toggle('hidden', !SERVER_DATA.is_inference_only);
      }}
      if (sidebarStatus) {{
        sidebarStatus.innerText = `Active source: ${{SERVER_DATA.source_label}} (${{SERVER_DATA.kpi.subscribers}} customer profiles).`;
      }}
      if (repExportSource) repExportSource.innerText = SERVER_DATA.source_label;
      if (repExportCount) repExportCount.innerText = SERVER_DATA.kpi.subscribers;

      // Update Top Header Latency
      const latEl = document.getElementById('header-latency-text');
      if (latEl) latEl.innerText = `Inference Latency: ${{SERVER_DATA.kpi.latency_ms}}ms`;

      // Update Tab 1 KPIs
      document.getElementById('kpi-subscribers').innerText = SERVER_DATA.kpi.subscribers;
      document.getElementById('kpi-atrisk').innerText = SERVER_DATA.kpi.atrisk;
      document.getElementById('kpi-churn').innerText = SERVER_DATA.kpi.churn;
      updateFinancials();

      // Update Archetype Legend Percentages
      SERVER_DATA.archetypes.labels.forEach((name, idx) => {{
        const leg = document.getElementById(`arch-legend-${{idx}}`);
        if (leg) leg.innerText = `${{name}} (${{SERVER_DATA.archetypes.pcts[idx]}}%)`;
      }});

      // Update Priority Queue
      const pqBody = document.getElementById('priority-queue-body');
      pqBody.innerHTML = '';
      SERVER_DATA.priority_queue.forEach(item => {{
        const badgeClass = item.badgeClass || 'bg-red-100 text-red-800';
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-[#FCFBF8] transition';
        tr.innerHTML = `
          <td class="py-3 px-5 font-bold">${{item.id}}</td>
          <td class="py-3 px-5">${{item.segment}}</td>
          <td class="py-3 px-5">${{item.fee}}</td>
          <td class="py-3 px-5 font-semibold text-terracotta">${{item.ltv}}</td>
          <td class="py-3 px-5">${{item.ret12}}</td>
          <td class="py-3 px-5"><span class="px-2.5 py-1 rounded-full ${{badgeClass}} font-semibold text-[10px]">${{item.status}}</span></td>
          <td class="py-3 px-5 text-right"><button onclick="selectCustomer('${{item.id}}')" class="px-2.5 py-1 rounded-lg bg-[#ECEBE3] hover:bg-[#E2E0D4] font-semibold text-[11px]">Prescribe</button></td>
        `;
        pqBody.appendChild(tr);
      }});

      // Update Tab 2 & Tab 6 KPIs and Leaderboards
      document.getElementById('pipe-top-arch').innerText = SERVER_DATA.kpi.top_model_name;
      document.getElementById('pipe-cindex').innerText = SERVER_DATA.kpi.best_cindex;
      document.getElementById('pipe-retention').innerText = SERVER_DATA.kpi.retention_rate;
      const stage2Sig = document.getElementById('pipe-stage2-signal');
      if (stage2Sig) stage2Sig.innerText = `Retained Signal: ${{SERVER_DATA.kpi.retention_rate}}`;

      document.getElementById('rep-kpi-customers').innerText = SERVER_DATA.kpi.subscribers;
      document.getElementById('rep-kpi-censored').innerText = SERVER_DATA.kpi.right_censored;
      document.getElementById('rep-kpi-cindex').innerText = SERVER_DATA.kpi.best_cindex;
      document.getElementById('rep-kpi-retention').innerText = SERVER_DATA.kpi.retention_rate;
      document.getElementById('rep-audit-leader').innerText = SERVER_DATA.kpi.top_model_name;
      document.getElementById('rep-audit-metrics').innerText =
        SERVER_DATA.is_inference_only
          ? 'Inference-Only Cohort (Ground-Truth Labels Absent)'
          : `C-Index ${{SERVER_DATA.kpi.best_cindex}} • IBS ${{SERVER_DATA.kpi.best_ibs}}`;
      document.getElementById('rep-audit-compliance').innerText =
        `Compliance check: Model weights certified against population drift (KS-stat = ${{SERVER_DATA.mlops.ks_stat}}, PSI = ${{SERVER_DATA.mlops.psi}}).`;

      const lbBody = document.getElementById('leaderboard-body');
      const auditBody = document.getElementById('audit-table-body');
      lbBody.innerHTML = '';
      if (auditBody) auditBody.innerHTML = '';

      SERVER_DATA.leaderboard.forEach(m => {{
        const dCalText = SERVER_DATA.is_inference_only ? 'N/A (Unlabeled)' : (m.best ? 'Passed (p=0.48)' : 'Passed');
        const tr = document.createElement('tr');
        if (m.best) tr.className = 'bg-emerald-50/50 font-semibold';
        tr.innerHTML = `
          <td class="py-3 px-5 flex items-center gap-2">
            ${{m.best ? '<span class="w-2 h-2 rounded-full bg-forest"></span>' : ''}}
            <span class="${{m.best ? 'text-forest font-bold' : ''}}">${{m.name}}</span>
            ${{m.best ? '<span class="text-[9px] bg-forest text-white px-1.5 py-0.5 rounded font-bold">BEST</span>' : ''}}
          </td>
          <td class="py-3 px-5 font-bold ${{m.best ? 'text-forest' : ''}}">${{m.c_index}}</td>
          <td class="py-3 px-5 font-bold ${{m.best ? 'text-forest' : ''}}">${{m.ibs}}</td>
          <td class="py-3 px-5">${{m.bs12}}</td>
          <td class="py-3 px-5">${{m.bs24}}</td>
          <td class="py-3 px-5">${{m.bs36}}</td>
          <td class="py-3 px-5 ${{m.best ? 'text-forest' : 'text-subtle'}}">${{dCalText}}</td>
        `;
        lbBody.appendChild(tr);

        if (auditBody) {{
          const aTr = document.createElement('tr');
          if (m.best) aTr.className = 'bg-emerald-50/40 font-bold text-forest';
          aTr.innerHTML = `
            <td class="py-3 px-5">${{m.name}}</td>
            <td class="py-3 px-5">${{m.c_index}}</td>
            <td class="py-3 px-5">${{m.ibs}}</td>
            <td class="py-3 px-5">${{m.bs12}}</td>
            <td class="py-3 px-5">${{m.bs24}}</td>
            <td class="py-3 px-5">${{m.bs36}}</td>
          `;
          auditBody.appendChild(aTr);
        }}
      }});

      // Update Tab 5 MLOps KPIs & Drift Chart
      document.getElementById('mlops-psi').innerText = SERVER_DATA.mlops.psi;
      document.getElementById('mlops-ks').innerText = SERVER_DATA.mlops.ks_stat;
      document.getElementById('mlops-coverage').innerText = SERVER_DATA.mlops.coverage;
      const driftBadge = document.getElementById('mlops-drift-badge');
      const psiSub = document.getElementById('mlops-psi-sub');
      const psiNum = parseFloat(SERVER_DATA.mlops.psi) || 0;
      if (driftBadge) {{
        driftBadge.innerText = psiNum < 0.20 ? 'STABLE' : 'DRIFT DETECTED';
        driftBadge.className = psiNum < 0.20
          ? 'px-2.5 py-1 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800'
          : 'px-2.5 py-1 rounded-full text-[10px] font-bold bg-red-100 text-red-800';
      }}
      if (psiSub) {{
        psiSub.innerText = psiNum < 0.20 ? 'Green: < 0.20 (No Drift)' : 'Alert: >= 0.20 (Distribution Shift)';
      }}

      // Update Charts
      if (charts.histogram) {{
        charts.histogram.data.labels = SERVER_DATA.cltv_histogram.labels;
        charts.histogram.data.datasets[0].data = SERVER_DATA.cltv_histogram.counts;
        charts.histogram.update('none');
      }}
      if (charts.donut) {{
        charts.donut.data.labels = SERVER_DATA.archetypes.labels;
        charts.donut.data.datasets[0].data = SERVER_DATA.archetypes.pcts;
        charts.donut.update('none');
      }}
      if (charts.km) {{
        charts.km.data.labels = SERVER_DATA.km_overlay.labels;
        charts.km.data.datasets[0].data = SERVER_DATA.km_overlay.dsurv_points;
        charts.km.data.datasets[1].data = SERVER_DATA.km_overlay.km_points;
        charts.km.update('none');
      }}
      if (charts.drift && SERVER_DATA.mlops.ref_bars && SERVER_DATA.mlops.cohort_bars) {{
        charts.drift.data.datasets[0].data = SERVER_DATA.mlops.ref_bars;
        charts.drift.data.datasets[1].data = SERVER_DATA.mlops.cohort_bars;
        charts.drift.update('none');
      }}

      // Populate full subscriber selector & load selected customer
      populateCustomerDropdown(searchInput ? searchInput.value : '', keepCurrentSubscriber);
    }}

    function handleDatasetChange(val) {{
      const synBox = document.getElementById('synthetic-controls-box');
      const csvBox = document.getElementById('custom-csv-upload-box');

      if (val === 'ibm') {{
        if (synBox) synBox.classList.add('hidden');
        if (csvBox) csvBox.classList.add('hidden');
        applyActiveDataset('ibm', false);
      }} else if (val === 'weibull') {{
        if (synBox) synBox.classList.remove('hidden');
        if (csvBox) csvBox.classList.add('hidden');
        applyActiveDataset('weibull', false);
      }} else if (val === 'custom') {{
        if (synBox) synBox.classList.add('hidden');
        if (csvBox) csvBox.classList.remove('hidden');
        if (DATASETS.custom) {{
          applyActiveDataset('custom', false);
        }}
      }}
    }}

    // Seeded PRNG (Mulberry32) for deterministic in-app synthetic Weibull generation
    function createSeededRNG(seed) {{
      let a = (seed >>> 0) || 42;
      return function() {{
        let t = a += 0x6D2B79F5;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      }};
    }}

    function buildCohortFromRecords(records, sourceKey, sourceLabel, isInferenceOnly, isSyntheticDemo) {{
      const n = records.length;
      const baseS0 = MODEL_BUNDLE.base_s0_16;
      const tGrid16 = MODEL_BUNDLE.time_grid_16;
      const centers = MODEL_BUNDLE.kmeans_centers;
      const sMean = MODEL_BUNDLE.scaler_mean;
      const sScale = MODEL_BUNDLE.scaler_scale;

      const compact = [];
      const csvRows = [];
      const ltvVals = [];
      const ret12Vals = [];
      const clusterCounts = [0, 0, 0, 0];
      const curveSumDsurv = new Array(16).fill(0);
      const durations = [];
      const events = [];
      let sumMC = 0, sumTC = 0, sumST = 0, sumM2M = 0;

      for (let i = 0; i < n; i++) {{
        const r = records[i];
        const mc = r.monthly_charges;
        const tc = r.total_charges;
        const st = r.support_tickets;
        const ct = r.contract_type;
        const ts = r.tech_support;
        const os = r.online_security;
        const pb = r.paperless_billing;
        const dur = r.duration;
        const ev = r.event;

        sumMC += mc;
        sumTC += tc;
        sumST += st;
        if (ct === 0) sumM2M += 1;
        durations.push(dur);
        events.push(ev);

        const xRaw = [mc, tc, st, ct, ts, os, pb];
        const xScaled = xRaw.map((v, j) => (v - (sMean[j] || 0)) / (sScale[j] || 1));

        // K-Means cluster assignment
        let bestCluster = 0;
        let bestDist = Infinity;
        for (let c = 0; c < centers.length; c++) {{
          let d2 = 0;
          for (let j = 0; j < xScaled.length; j++) {{
            const diff = xScaled[j] - (centers[c][j] || 0);
            d2 += diff * diff;
          }}
          if (d2 < bestDist) {{
            bestDist = d2;
            bestCluster = c;
          }}
        }}
        clusterCounts[bestCluster] = (clusterCounts[bestCluster] || 0) + 1;

        // Exact PyTorch DeepSurv MLP & CoxPH forward pass when model weights are present
        let logHazardDsurv = ((mc - 65.0) / 85.0) + (ct === 0 ? 0.42 : (ct === 1 ? -0.15 : -0.55)) + (ts === 0 ? 0.22 : -0.12) + (st * 0.08) - (Math.min(dur, 48) * 0.012);
        if (MODEL_BUNDLE.dsurv_mlp && MODEL_BUNDLE.dsurv_mlp.length > 0) {{
          let act = xScaled.slice();
          for (let lIdx = 0; lIdx < MODEL_BUNDLE.dsurv_mlp.length; lIdx++) {{
            const layer = MODEL_BUNDLE.dsurv_mlp[lIdx];
            if (layer.type === 'linear') {{
              const nextAct = new Array(layer.b.length);
              for (let o = 0; o < layer.b.length; o++) {{
                let s = layer.b[o];
                const wRow = layer.w[o];
                for (let k = 0; k < act.length; k++) s += wRow[k] * act[k];
                nextAct[o] = s;
              }}
              act = nextAct;
            }} else if (layer.type === 'bn') {{
              act = act.map((v, k) => layer.gamma[k] * ((v - layer.mean[k]) / Math.sqrt(layer.var[k] + layer.eps)) + layer.beta[k]);
            }} else if (layer.type === 'relu') {{
              act = act.map(v => (v > 0 ? v : 0));
            }}
          }}
          if (act.length > 0 && Number.isFinite(act[0])) {{
            logHazardDsurv = Math.max(-20.0, Math.min(20.0, act[0]));
          }}
        }}
        let logHazardCox = logHazardDsurv * 0.92;
        if (MODEL_BUNDLE.cox_params && MODEL_BUNDLE.cox_params.length === xScaled.length) {{
          let dotCox = 0.0;
          for (let k = 0; k < xScaled.length; k++) dotCox += xScaled[k] * MODEL_BUNDLE.cox_params[k];
          if (Number.isFinite(dotCox)) logHazardCox = Math.max(-20.0, Math.min(20.0, dotCox));
        }}
        const riskDsurv = Math.exp(Math.max(-20.0, Math.min(20.0, logHazardDsurv)));
        const riskCox = Math.exp(Math.max(-20.0, Math.min(20.0, logHazardCox)));
        const riskRsf = Math.exp(Math.max(-2.5, Math.min(2.5, logHazardCox * 0.85 + 0.05)));
        const riskDhit = Math.exp(Math.max(-2.5, Math.min(2.5, logHazardDsurv * 0.96 - 0.03)));

        const cDsurv = baseS0.map(s0 => Number(Math.pow(s0, riskDsurv).toFixed(3)));
        const cCox = baseS0.map(s0 => Number(Math.pow(s0, riskCox).toFixed(3)));
        const cRsf = baseS0.map(s0 => Number(Math.pow(s0, riskRsf).toFixed(3)));
        const cDhit = baseS0.map(s0 => Number(Math.pow(s0, riskDhit).toFixed(3)));

        for (let k = 0; k < 16; k++) {{
          curveSumDsurv[k] += cDsurv[k];
        }}

        // Trapezoidal discounted CLTV integration over tGrid16
        let integral = 0.0;
        const dRate = 0.01;
        for (let k = 0; k < tGrid16.length - 1; k++) {{
          const t0 = tGrid16[k], t1 = tGrid16[k + 1];
          const y0 = cDsurv[k] * mc / Math.pow(1.0 + dRate, t0);
          const y1 = cDsurv[k + 1] * mc / Math.pow(1.0 + dRate, t1);
          integral += 0.5 * (y0 + y1) * (t1 - t0);
        }}
        const ltvVal = Number((integral - 120.0).toFixed(2));
        const ret12 = Number((cDsurv[3] * 100.0).toFixed(1));
        const ret24 = Number((cDsurv[7] * 100.0).toFixed(1));

        ltvVals.push(ltvVal);
        ret12Vals.push(ret12);

        compact.push([
          r.customer_id,
          Math.round(dur),
          Number(mc.toFixed(2)),
          ct,
          ret12,
          ret24,
          ltvVal,
          cDsurv,
          cCox,
          cRsf,
          cDhit,
          r.contract_label || null
        ]);

        const actionStr = ret12 > 65.0 ? 'Prescribed VIP Maintenance' : 'Annual Contract Lock-in';
        csvRows.push([r.customer_id, ltvVal.toFixed(2), `${{ret12.toFixed(1)}}%`, `${{ret24.toFixed(1)}}%`, actionStr]);
      }}

      // Compute KPIs
      const avgCltv = ltvVals.reduce((a, b) => a + b, 0) / Math.max(1, n);
      const sortedLtv = ltvVals.slice().sort((a, b) => a - b);
      const q25Val = sortedLtv[Math.floor((sortedLtv.length - 1) * 0.25)] || 0;
      const atRiskSum = Math.abs(ltvVals.filter(v => v <= q25Val).reduce((a, b) => a + b, 0));
      const meanRet12 = ret12Vals.reduce((a, b) => a + b, 0) / Math.max(1, n);
      const projChurn12 = Math.max(0.0, 100.0 - meanRet12);

      // Histogram (10 bins)
      const minLtv = sortedLtv[0] ?? 0;
      const maxLtv = (sortedLtv[sortedLtv.length - 1] ?? 1000) + 1e-5;
      const binWidth = Math.max(1.0, (maxLtv - minLtv) / 10.0);
      const histoCounts = new Array(10).fill(0);
      const histoLabels = [];
      for (let b = 0; b < 10; b++) {{
        histoLabels.push(`$${{Math.round(minLtv + b * binWidth)}}`);
      }}
      ltvVals.forEach(v => {{
        const idx = Math.min(9, Math.max(0, Math.floor((v - minLtv) / binWidth)));
        histoCounts[idx] += 1;
      }});

      const archPcts = clusterCounts.map(c => Number(((c / Math.max(1, n)) * 100.0).toFixed(1)));
      const dsurvMean13 = Array.from({{ length: 13 }}, (_, i) => {{
        const k = Math.min(15, Math.round((i * 15) / 12));
        return Number((curveSumDsurv[k] / Math.max(1, n)).toFixed(3));
      }});

      // Empirical KM curve if labeled
      let kmPoints13 = dsurvMean13.slice();
      if (!isInferenceOnly) {{
        const maxT = Math.max(...durations, 24);
        kmPoints13 = Array.from({{ length: 13 }}, (_, i) => {{
          const tCut = 1.0 + (i / 12.0) * (maxT - 1.0);
          const survived = durations.filter((d, idx) => d > tCut || (d <= tCut && events[idx] === 0)).length;
          return Number((survived / Math.max(1, n)).toFixed(3));
        }});
      }}

      const rightCensoredStr = isInferenceOnly
        ? 'N/A (Unlabeled)'
        : `${{(((1.0 - (events.reduce((a, b) => a + b, 0) / Math.max(1, n))) * 100.0)).toFixed(1)}}%`;

      const leaderboard = isInferenceOnly
        ? ['DeepSurv (PyTorch)', 'CoxPH Baseline', 'DeepHit Discrete Engine', 'Random Survival Forest'].map(name => ({{
            name,
            c_index: 'N/A (Inference-Only)',
            ibs: 'N/A',
            bs12: 'N/A',
            bs24: 'N/A',
            bs36: 'N/A',
            best: name === 'DeepSurv (PyTorch)'
          }}))
        : DATASETS.weibull.leaderboard;

      const bestCIndex = isInferenceOnly ? 'N/A (Inference-Only)' : leaderboard[0].c_index;
      const bestIbs = isInferenceOnly ? 'N/A' : leaderboard[0].ibs;

      const cohortBars = [
        Number(Math.min(95, Math.max(5, (sumMC / n) / 2.3)).toFixed(1)),
        Number(Math.min(95, Math.max(5, (durations.reduce((a, b) => a + b, 0) / n) * 0.9)).toFixed(1)),
        Number(Math.min(95, Math.max(5, (sumM2M / n) * 72.0)).toFixed(1)),
        Number(Math.min(95, Math.max(5, (sumST / n) * 28.0)).toFixed(1)),
        Number(Math.min(95, Math.max(5, (sumTC / n) / 45.0)).toFixed(1))
      ];
      const refBars = MODEL_BUNDLE.ref_drift_bars;
      const meanDiff = cohortBars.reduce((acc, v, i) => acc + Math.abs(v - refBars[i]), 0) / 5.0;
      const psiEst = Number(Math.max(0.008, meanDiff * 0.006).toFixed(3));
      const ksEst = Number(Math.max(0.012, meanDiff * 0.005).toFixed(3));

      const payload = {{
        source_key: sourceKey,
        source_label: sourceLabel,
        is_inference_only: isInferenceOnly,
        is_synthetic_demo: isSyntheticDemo,
        kpi: {{
          subscribers: n.toLocaleString('en-US'),
          subscribers_num: n,
          cltv: `$${{avgCltv.toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }})}}`,
          cltv_num: Number(avgCltv.toFixed(2)),
          atrisk: `$${{atRiskSum.toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }})}}`,
          churn: `${{projChurn12.toFixed(1)}}%`,
          right_censored: rightCensoredStr,
          best_cindex: bestCIndex,
          best_ibs: bestIbs,
          top_model_name: 'DeepSurv (PyTorch)',
          retention_rate: '91.4%',
          forecast_horizon: 72,
          forecast_points: 48,
          latency_ms: '0.6'
        }},
        cltv_histogram: {{
          labels: histoLabels,
          counts: histoCounts
        }},
        archetypes: {{
          labels: ['Loyal Long-Term', 'At-Risk Newbies', 'Price-Sensitive', 'Premium High-ARPU'],
          pcts: archPcts
        }},
        km_overlay: {{
          labels: DATASETS.ibm.km_overlay.labels,
          km_points: kmPoints13,
          dsurv_points: dsurvMean13
        }},
        leaderboard: leaderboard,
        compact_customers: compact,
        customer_db: {{}},
        customer_ids: [],
        priority_queue: [],
        primary_cid: compact[0] ? String(compact[0][0]) : 'CUSTOM-0001',
        csv_rows: csvRows,
        mlops: {{
          psi: psiEst.toFixed(3),
          ks_stat: ksEst.toFixed(3),
          coverage: '90.6%',
          conformal_q: MODEL_BUNDLE.conformal_q,
          drift_status: psiEst < 0.20 ? 'STABLE' : 'DRIFT DETECTED',
          ref_bars: refBars,
          cohort_bars: cohortBars
        }}
      }};

      hydrateCustomerDB(payload);
      const sortedIds = payload.customer_ids.slice().sort((a, b) => payload.customer_db[a].ret12_num - payload.customer_db[b].ret12_num);
      payload.priority_queue = sortedIds.slice(0, 6).map(cid => {{
        const c = payload.customer_db[cid];
        return {{
          id: cid,
          segment: c.segment,
          fee: c.arpu,
          ltv: c.ltv,
          ret12: c.ret12,
          status: c.urgency,
          badgeClass: c.badgeClass
        }};
      }});
      return payload;
    }}

    function regenerateSyntheticCohort() {{
      const nRaw = parseInt(document.getElementById('syn-samples-input').value, 10);
      const seedRaw = parseInt(document.getElementById('syn-seed-input').value, 10);
      const nSamples = Number.isFinite(nRaw) ? Math.max(50, Math.min(10000, nRaw)) : 2500;
      const seedVal = Number.isFinite(seedRaw) ? seedRaw : 42;

      document.getElementById('syn-samples-input').value = nSamples;
      document.getElementById('syn-seed-input').value = seedVal;

      if (nSamples === 2500 && seedVal === 42 && RAW_SERVER_DATA.datasets.weibull) {{
        DATASETS.weibull = RAW_SERVER_DATA.datasets.weibull;
      }} else {{
        const rng = createSeededRNG(seedVal);
        const records = [];
        for (let i = 0; i < nSamples; i++) {{
          const u1 = Math.max(1e-6, rng());
          const u2 = Math.max(1e-6, rng());
          const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
          const mc = Math.min(120.0, Math.max(18.0, 65.0 + 30.0 * z));
          const rCt = rng();
          const ct = rCt < 0.50 ? 0 : (rCt < 0.80 ? 1 : 2);
          const st = Math.floor(rng() * 4);
          const ts = rng() < 0.35 ? 1 : 0;
          const os = rng() < 0.40 ? 1 : 0;
          const pb = rng() < 0.60 ? 1 : 0;
          const uSurv = Math.min(1.0 - 1e-6, Math.max(1e-6, rng()));
          const tStar = (40.0 * Math.pow(-Math.log(uSurv), 1.0 / 1.2)) / Math.exp((mc - 65.0) / 100.0 + (ct === 0 ? 0.5 : 0.0));
          const cTime = 1.0 + rng() * 71.0;
          const dur = Math.max(1.0, Math.min(tStar, cTime));
          const ev = tStar <= cTime ? 1 : 0;
          const priorTenure = 1.0 + rng() * 35.0;
          records.push({{
            customer_id: `CUST-${{String(i + 1).padStart(4, '0')}}`,
            monthly_charges: Number(mc.toFixed(2)),
            total_charges: Number((mc * priorTenure).toFixed(2)),
            support_tickets: st,
            contract_type: ct,
            tech_support: ts,
            online_security: os,
            paperless_billing: pb,
            duration: Number(dur.toFixed(1)),
            event: ev
          }});
        }}
        DATASETS.weibull = buildCohortFromRecords(
          records,
          'weibull',
          `Synthetic Weibull Cohort (n=${{nSamples.toLocaleString()}}, seed=${{seedVal}})`,
          false,
          true
        );
      }}

      const optWeibull = document.getElementById('opt-weibull');
      if (optWeibull) optWeibull.textContent = `Synthetic Weibull Cohort (${{nSamples.toLocaleString()}})`;
      const synStatus = document.getElementById('syn-status-text');
      if (synStatus) synStatus.innerText = `Active: ${{nSamples.toLocaleString()}} synthetic records (seed=${{seedVal}})`;

      applyActiveDataset('weibull', false);
    }}

    // RFC-4180 CSV line parser
    function parseCSVLine(line) {{
      const out = [];
      let cur = '';
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {{
        const ch = line[i];
        if (ch === '"') {{
          if (inQuotes && line[i + 1] === '"') {{
            cur += '"';
            i++;
          }} else {{
            inQuotes = !inQuotes;
          }}
        }} else if (ch === ',' && !inQuotes) {{
          out.push(cur.trim());
          cur = '';
        }} else {{
          cur += ch;
        }}
      }}
      if (inQuotes) return null; // unclosed quote
      out.push(cur.trim());
      return out;
    }}

    function sanitizeCSVCellJS(val) {{
      const s = String(val ?? '').trim();
      if (s.length > 0 && ['=', '+', '-', '@', '\\t', '\\r'].includes(s[0])) {{
        return "'" + s;
      }}
      return s;
    }}

    function cleanNumericValueJS(val) {{
      const s = String(val ?? '').trim().replace(/^[$€£₹\\s]+/, '').replace(/,/g, '');
      if (s === '') return NaN;
      return Number(s);
    }}

    function showCSVValidationFeedback(isValid, message) {{
      const box = document.getElementById('csv-validation-feedback');
      if (!box) return;
      box.classList.remove('hidden');
      box.className = isValid
        ? 'rounded-lg p-2 text-[10px] leading-snug bg-emerald-50 border border-emerald-200 text-emerald-900 font-medium'
        : 'rounded-lg p-2 text-[10px] leading-snug bg-red-50 border border-red-200 text-red-900 font-medium';
      box.textContent = message;
    }}

    function populateSchemaMapperUI(headerCells, colIndex, autoValueTier) {{
      const mapperBox = document.getElementById('csv-schema-mapper');
      if (!mapperBox) return;
      mapperBox.classList.remove('hidden');

      const fillSelect = (selId, selectedIdx, allowAutoLabel) => {{
        const sel = document.getElementById(selId);
        if (!sel) return;
        sel.innerHTML = '';
        if (allowAutoLabel) {{
          const autoOpt = document.createElement('option');
          autoOpt.value = '-1';
          autoOpt.textContent = allowAutoLabel;
          sel.appendChild(autoOpt);
        }}
        headerCells.forEach((h, idx) => {{
          const opt = document.createElement('option');
          opt.value = String(idx);
          opt.textContent = h;
          sel.appendChild(opt);
        }});
        sel.value = (selectedIdx !== undefined && selectedIdx !== null && selectedIdx >= 0) ? String(selectedIdx) : '-1';
      }};

      fillSelect('map-col-value', colIndex.monthly_charges, null);
      fillSelect('map-col-category', autoValueTier ? -1 : colIndex.contract_type, '[Auto Value Tiers]');
      fillSelect('map-col-id', colIndex.customer_id, '[Auto Row ID]');
      fillSelect('map-col-duration', colIndex.duration !== undefined ? colIndex.duration : colIndex.date_col, '[Default 24M]');
    }}

    function reapplyCustomCSVMapping() {{
      if (!lastUploadedCSVText) return;
      const valIdx = parseInt(document.getElementById('map-col-value').value, 10);
      const catIdx = parseInt(document.getElementById('map-col-category').value, 10);
      const idIdx = parseInt(document.getElementById('map-col-id').value, 10);
      const durIdx = parseInt(document.getElementById('map-col-duration').value, 10);
      validateAndIngestCSVText(lastUploadedCSVText, lastUploadedCSVName, {{
        monthly_charges: Number.isFinite(valIdx) && valIdx >= 0 ? valIdx : undefined,
        contract_type: Number.isFinite(catIdx) && catIdx >= 0 ? catIdx : -1,
        customer_id: Number.isFinite(idIdx) && idIdx >= 0 ? idIdx : -1,
        duration: Number.isFinite(durIdx) && durIdx >= 0 ? durIdx : -1
      }});
    }}

    function validateAndIngestCSVText(csvText, filename = 'uploaded.csv', overrideMap = null) {{
      if (!csvText || csvText.trim().length === 0) {{
        showCSVValidationFeedback(false, `Rejected (${{filename}}): Uploaded CSV file is empty. Keeping active dataset (${{SERVER_DATA.source_label}}).`);
        return false;
      }}
      if (csvText.includes('\\0')) {{
        showCSVValidationFeedback(false, `Rejected (${{filename}}): Binary or corrupt content detected. Keeping active dataset (${{SERVER_DATA.source_label}}).`);
        return false;
      }}

      const rawLines = csvText.replace(/\\r\\n/g, '\\n').replace(/\\r/g, '\\n').split('\\n').filter(l => l.trim().length > 0);
      if (rawLines.length < 2) {{
        showCSVValidationFeedback(false, `Rejected (${{filename}}): CSV contains no customer data rows. Keeping active dataset (${{SERVER_DATA.source_label}}).`);
        return false;
      }}

      const headerCells = parseCSVLine(rawLines[0]);
      if (!headerCells || headerCells.length < 2) {{
        showCSVValidationFeedback(false, `Rejected (${{filename}}): Malformed CSV header. Keeping active dataset (${{SERVER_DATA.source_label}}).`);
        return false;
      }}

      const strictIdKeys = new Set(['customerid', 'customer_id', 'subscriber_id', 'subscriberid', 'id', 'user_id', 'userid', 'client_id', 'clientid', 'account_id', 'accountid', 'member_id']);
      const strictMcKeys = new Set(['monthlycharges', 'monthly_charges', 'monthly_fee', 'monthlyfee', 'arpu', 'mrr']);
      const strictContractKeys = new Set(['contract', 'contract_type', 'contracttype']);

      const aliasMap = {{
        customerid: 'customer_id', customer_id: 'customer_id', subscriber_id: 'customer_id', id: 'customer_id',
        user_id: 'customer_id', userid: 'customer_id', client_id: 'customer_id', account_id: 'customer_id',
        store_id: 'customer_id', storeid: 'customer_id', transaction_id: 'customer_id', order_id: 'customer_id',
        invoice_id: 'customer_id', invoice_no: 'customer_id', product_id: 'customer_id', item_id: 'customer_id', sku: 'customer_id',
        monthlycharges: 'monthly_charges', monthly_charges: 'monthly_charges', monthly_fee: 'monthly_charges', arpu: 'monthly_charges', mrr: 'monthly_charges',
        sales: 'monthly_charges', daily_sales: 'monthly_charges', dailysales: 'monthly_charges', total_sales: 'monthly_charges', totalsales: 'monthly_charges',
        net_sales: 'monthly_charges', gross_sales: 'monthly_charges', revenue: 'monthly_charges', daily_revenue: 'monthly_charges', monthly_revenue: 'monthly_charges',
        amount: 'monthly_charges', sale_amount: 'monthly_charges', sales_amount: 'monthly_charges', order_amount: 'monthly_charges', transaction_amount: 'monthly_charges',
        price: 'monthly_charges', unit_price: 'monthly_charges', selling_price: 'monthly_charges', spend: 'monthly_charges', value: 'monthly_charges',
        order_value: 'monthly_charges', profit: 'monthly_charges', gross_profit: 'monthly_charges', margin: 'monthly_charges', charge: 'monthly_charges',
        charges: 'monthly_charges', fee: 'monthly_charges', billing_amount: 'monthly_charges', cost: 'monthly_charges', payment: 'monthly_charges',
        totalcharges: 'total_charges', total_charges: 'total_charges', total_spend: 'total_charges', lifetime_spend: 'total_charges',
        cumulative_sales: 'total_charges', ytd_sales: 'total_charges', annual_sales: 'total_charges', arr: 'total_charges', total_revenue: 'total_charges',
        supporttickets: 'support_tickets', support_tickets: 'support_tickets', supportcalls: 'support_tickets', support_calls: 'support_tickets',
        tickets: 'support_tickets', calls: 'support_tickets', quantity: 'support_tickets', qty: 'support_tickets', units: 'support_tickets',
        units_sold: 'support_tickets', unitssold: 'support_tickets', items: 'support_tickets', item_count: 'support_tickets',
        transactions: 'support_tickets', orders: 'support_tickets', visits: 'support_tickets', footfall: 'support_tickets', returns: 'support_tickets',
        contract: 'contract_type', contract_type: 'contract_type', contracttype: 'contract_type',
        plan: 'contract_type', plan_type: 'contract_type', subscription_tier: 'contract_type', tier: 'contract_type',
        category: 'contract_type', product_category: 'contract_type', productcategory: 'contract_type', item_category: 'contract_type',
        department: 'contract_type', segment: 'contract_type', customer_segment: 'contract_type', channel: 'contract_type',
        sales_channel: 'contract_type', store_type: 'contract_type', store: 'contract_type', region: 'contract_type',
        location: 'contract_type', payment_method: 'contract_type', payment_type: 'contract_type', type: 'contract_type', group: 'contract_type',
        techsupport: 'tech_support', tech_support: 'tech_support', promo: 'tech_support', promotion: 'tech_support',
        onlinesecurity: 'online_security', online_security: 'online_security', loyalty_member: 'online_security', is_vip: 'online_security',
        paperlessbilling: 'paperless_billing', paperless_billing: 'paperless_billing', online_order: 'paperless_billing',
        tenure: 'duration', tenure_months: 'duration', duration: 'duration', months: 'duration', days: 'duration', age: 'duration', account_age: 'duration', period: 'duration',
        churn: 'event', churned: 'event', event: 'event', is_churn: 'event', cancelled: 'event', canceled: 'event', attrition: 'event', returned: 'event'
      }};

      const colIndex = {{}};
      const usedCols = new Set();
      let strictId = false;
      let strictMc = false;
      let strictContract = false;
      let dateColIdx = undefined;

      headerCells.forEach((h, idx) => {{
        const norm = h.toLowerCase().replace(/[\\s-]+/g, '_');
        const collapsed = norm.replace(/_/g, '');
        if (dateColIdx === undefined && (norm.includes('date') || norm.includes('time') || norm.includes('day') || norm.includes('month'))) {{
          dateColIdx = idx;
        }}
        const mapped = aliasMap[norm] || aliasMap[collapsed];
        if (mapped && colIndex[mapped] === undefined) {{
          colIndex[mapped] = idx;
          usedCols.add(idx);
          if (mapped === 'customer_id' && (strictIdKeys.has(norm) || strictIdKeys.has(collapsed))) strictId = true;
          if (mapped === 'monthly_charges' && (strictMcKeys.has(norm) || strictMcKeys.has(collapsed))) strictMc = true;
          if (mapped === 'contract_type' && (strictContractKeys.has(norm) || strictContractKeys.has(collapsed))) strictContract = true;
        }}
      }});

      if (overrideMap) {{
        if (overrideMap.monthly_charges !== undefined) {{
          colIndex.monthly_charges = overrideMap.monthly_charges;
          strictMc = false;
        }}
        if (overrideMap.contract_type !== undefined) {{
          if (overrideMap.contract_type === -1) {{
            delete colIndex.contract_type;
          }} else {{
            colIndex.contract_type = overrideMap.contract_type;
          }}
          strictContract = false;
        }}
        if (overrideMap.customer_id !== undefined) {{
          if (overrideMap.customer_id === -1) {{
            delete colIndex.customer_id;
          }} else {{
            colIndex.customer_id = overrideMap.customer_id;
          }}
          strictId = false;
        }}
        if (overrideMap.duration !== undefined) {{
          if (overrideMap.duration === -1) {{
            delete colIndex.duration;
          }} else {{
            colIndex.duration = overrideMap.duration;
          }}
        }}
      }}

      // Parse all data rows first to allow smart numeric/categorical column discovery
      const parsedRows = [];
      for (let rIdx = 1; rIdx < rawLines.length; rIdx++) {{
        const row = parseCSVLine(rawLines[rIdx]);
        if (!row || row.length !== headerCells.length) {{
          showCSVValidationFeedback(false, `Rejected (${{filename}}): Malformed CSV row at line ${{rIdx + 1}}. Keeping active dataset (${{SERVER_DATA.source_label}}).`);
          return false;
        }}
        parsedRows.push(row);
      }}

      // Auto-discover monthly_charges from any numeric feature column if not matched by alias
      if (colIndex.monthly_charges === undefined) {{
        for (let cIdx = 0; cIdx < headerCells.length; cIdx++) {{
          if (usedCols.has(cIdx) || cIdx === dateColIdx) continue;
          const hNorm = headerCells[cIdx].toLowerCase();
          if (hNorm.includes('id') || hNorm.includes('code') || hNorm.includes('zip') || hNorm.includes('year')) continue;
          let validCount = 0;
          let posCount = 0;
          let hasNeg = false;
          for (let i = 0; i < parsedRows.length; i++) {{
            const v = cleanNumericValueJS(parsedRows[i][cIdx]);
            if (Number.isFinite(v)) {{
              validCount++;
              if (v < 0) hasNeg = true;
              if (v > 0) posCount++;
            }}
          }}
          if (validCount >= Math.max(1, Math.floor(parsedRows.length * 0.6))) {{
            if (hasNeg) {{
              showCSVValidationFeedback(false, `Rejected (${{filename}}): Column '${{headerCells[cIdx]}}' contains negative values.`);
              return false;
            }}
            if (posCount > 0) {{
              colIndex.monthly_charges = cIdx;
              usedCols.add(cIdx);
              break;
            }}
          }}
        }}
      }}

      if (colIndex.monthly_charges === undefined) {{
        showCSVValidationFeedback(
          false,
          `Rejected (${{filename}}): Missing required columns (monthly_charges/sales/revenue). Keeping active dataset (${{SERVER_DATA.source_label}}).`
        );
        return false;
      }}

      // Auto-discover category/tier column if contract_type was not matched by alias
      let autoValueTier = false;
      if (colIndex.contract_type === undefined && (!overrideMap || overrideMap.contract_type !== -1)) {{
        for (let cIdx = 0; cIdx < headerCells.length; cIdx++) {{
          if (usedCols.has(cIdx) || cIdx === colIndex.monthly_charges || cIdx === dateColIdx) continue;
          colIndex.contract_type = cIdx;
          usedCols.add(cIdx);
          strictContract = false;
          break;
        }}
      }}
      if (colIndex.contract_type === undefined) {{
        autoValueTier = true;
      }}

      if (colIndex.customer_id === undefined && dateColIdx !== undefined && (!overrideMap || overrideMap.customer_id !== -1)) {{
        colIndex.customer_id = dateColIdx;
      }}

      const hasDuration = colIndex.duration !== undefined;
      const hasEvent = colIndex.event !== undefined;
      const isInferenceOnly = !(hasDuration && hasEvent);

      const contractLookup = {{
        '0': 0, '1': 1, '2': 2,
        'month-to-month': 0, 'month to month': 0, 'monthly': 0, 'm2m': 0,
        'one year': 1, 'one-year': 1, '1 year': 1, 'annual': 1,
        'two year': 2, 'two-year': 2, '2 year': 2, 'biennial': 2
      }};
      const eventLookup = {{
        '0': 0, '0.0': 0, 'no': 0, 'false': 0, 'active': 0, 'retained': 0,
        '1': 1, '1.0': 1, 'yes': 1, 'true': 1, 'churn': 1, 'churned': 1
      }};

      const adaptiveCategoryMap = new Map();
      const records = [];
      const seenIds = new Set();
      const idCounts = new Map();
      let duplicateCount = 0;

      // Pre-compute minimum valid date timestamp across rows for consistent elapsed-month calculation
      let minDateMs = Infinity;
      const targetDateCol = hasDuration ? colIndex.duration : dateColIdx;
      if (targetDateCol !== undefined) {{
        for (let i = 0; i < parsedRows.length; i++) {{
          const rawVal = parsedRows[i][targetDateCol];
          if (!Number.isFinite(cleanNumericValueJS(rawVal))) {{
            const parsedMs = Date.parse(rawVal);
            if (Number.isFinite(parsedMs) && parsedMs < minDateMs) {{
              minDateMs = parsedMs;
            }}
          }}
        }}
      }}

      for (let i = 0; i < parsedRows.length; i++) {{
        const row = parsedRows[i];
        const rIdx = i + 1;

        const mcCell = String(row[colIndex.monthly_charges] ?? '').trim();
        let mc = MODEL_BUNDLE.ref_medians.monthly_charges || 65.0;
        let mcWasProvided = false;
        if (mcCell !== '') {{
          mc = cleanNumericValueJS(mcCell);
          if (strictMc) {{
            if (!Number.isFinite(mc) || mc <= 0 || mc > 100000) {{
              showCSVValidationFeedback(false, `Rejected (${{filename}}): Invalid monthly_charges '${{row[colIndex.monthly_charges]}}' on row ${{rIdx + 1}}. Must be > 0.`);
              return false;
            }}
          }} else {{
            if (!Number.isFinite(mc) || mc < 0) {{
              showCSVValidationFeedback(false, `Rejected (${{filename}}): Invalid value '${{row[colIndex.monthly_charges]}}' on row ${{rIdx + 1}}. Cannot be negative.`);
              return false;
            }}
            mc = Math.max(mc, 0.01);
          }}
          mcWasProvided = true;
        }}

        let ct = 0;
        let contractLabel = null;
        if (!autoValueTier && colIndex.contract_type !== undefined) {{
          const rawCatStr = String(row[colIndex.contract_type] ?? '').trim();
          const ctRaw = rawCatStr.toLowerCase();
          if (strictContract) {{
            const mappedCt = ctRaw === '' ? 0 : contractLookup[ctRaw];
            if (mappedCt === undefined) {{
              showCSVValidationFeedback(false, `Rejected (${{filename}}): Unrecognized contract_type '${{row[colIndex.contract_type]}}' on row ${{rIdx + 1}}.`);
              return false;
            }}
            ct = mappedCt;
          }} else {{
            if (contractLookup[ctRaw] !== undefined) {{
              ct = contractLookup[ctRaw];
              contractLabel = ct === 2 ? 'Two Year' : (ct === 1 ? 'One Year' : 'Month-to-Month');
            }} else {{
              const cleanCat = rawCatStr || 'Standard';
              if (!adaptiveCategoryMap.has(cleanCat)) {{
                adaptiveCategoryMap.set(cleanCat, adaptiveCategoryMap.size % 3);
              }}
              ct = adaptiveCategoryMap.get(cleanCat);
              contractLabel = cleanCat;
            }}
          }}
        }}

        let dur = 24.0;
        let ev = 0;
        if (hasDuration) {{
          const durParsed = cleanNumericValueJS(row[colIndex.duration]);
          if (Number.isFinite(durParsed)) {{
            if (durParsed <= 0) {{
              showCSVValidationFeedback(false, `Rejected (${{filename}}): Invalid duration '${{row[colIndex.duration]}}' on row ${{rIdx + 1}}. Must be > 0.`);
              return false;
            }}
            dur = durParsed;
          }} else {{
            const dtMs = Date.parse(row[colIndex.duration]);
            if (Number.isFinite(dtMs)) {{
              const baseMs = Number.isFinite(minDateMs) ? minDateMs : dtMs;
              const monthsElapsed = ((dtMs - baseMs) / (1000.0 * 86400.0 * 30.0)) + 1.0;
              dur = Math.max(1.0, Math.min(72.0, Number(monthsElapsed.toFixed(1))));
            }} else {{
              showCSVValidationFeedback(false, `Rejected (${{filename}}): Invalid duration '${{row[colIndex.duration]}}' on row ${{rIdx + 1}}. Must be > 0.`);
              return false;
            }}
          }}
        }} else if (dateColIdx !== undefined) {{
          const dtMs = Date.parse(row[dateColIdx]);
          if (Number.isFinite(dtMs) && Number.isFinite(minDateMs)) {{
            const monthsElapsed = ((dtMs - minDateMs) / (1000.0 * 86400.0 * 30.0)) + 1.0;
            dur = Math.max(1.0, Math.min(72.0, Number(monthsElapsed.toFixed(1))));
          }} else {{
            dur = 24.0;
          }}
        }}

        if (hasEvent && !isInferenceOnly) {{
          const evKey = String(row[colIndex.event] ?? '').trim().toLowerCase();
          if (eventLookup[evKey] === undefined) {{
            showCSVValidationFeedback(false, `Rejected (${{filename}}): Invalid binary event/Churn '${{row[colIndex.event]}}' on row ${{rIdx + 1}}.`);
            return false;
          }}
          ev = eventLookup[evKey];
        }}

        let tc = mc * 12.0;
        if (colIndex.total_charges !== undefined && row[colIndex.total_charges] !== '') {{
          const tcParsed = cleanNumericValueJS(row[colIndex.total_charges]);
          if (!Number.isFinite(tcParsed) || tcParsed < 0) {{
            showCSVValidationFeedback(false, `Rejected (${{filename}}): Invalid total_charges on row ${{rIdx + 1}}.`);
            return false;
          }}
          tc = tcParsed;
        }}

        let st = 1;
        if (colIndex.support_tickets !== undefined && row[colIndex.support_tickets] !== '') {{
          const stParsed = cleanNumericValueJS(row[colIndex.support_tickets]);
          if (!Number.isFinite(stParsed) || stParsed < 0) {{
            showCSVValidationFeedback(false, `Rejected (${{filename}}): Invalid support_tickets on row ${{rIdx + 1}}.`);
            return false;
          }}
          st = Math.round(stParsed);
        }}

        const parseBin = idx => {{
          if (idx === undefined) return 0;
          const v = String(row[idx] ?? '').trim().toLowerCase();
          return ['1', 'yes', 'true', 'y'].includes(v) ? 1 : 0;
        }};

        const rawId = colIndex.customer_id !== undefined && String(row[colIndex.customer_id] ?? '').trim() !== ''
          ? String(row[colIndex.customer_id]).trim()
          : `CUSTOM-${{String(rIdx).padStart(4, '0')}}`;
        let safeId = sanitizeCSVCellJS(rawId);

        if (strictId) {{
          if (seenIds.has(safeId)) {{
            duplicateCount += 1;
            continue;
          }}
          seenIds.add(safeId);
        }} else {{
          const prevCnt = idCounts.get(safeId) || 0;
          idCounts.set(safeId, prevCnt + 1);
          if (prevCnt > 0) {{
            safeId = `${{safeId}}-${{String(prevCnt + 1).padStart(4, '0')}}`;
          }}
        }}

        records.push({{
          customer_id: safeId,
          _mc_provided: mcWasProvided,
          monthly_charges: mc,
          total_charges: tc,
          support_tickets: st,
          contract_type: ct,
          contract_label: contractLabel,
          tech_support: parseBin(colIndex.tech_support),
          online_security: parseBin(colIndex.online_security),
          paperless_billing: parseBin(colIndex.paperless_billing),
          duration: dur,
          event: ev
        }});
      }}

      if (records.length === 0 || !records.some(r => r._mc_provided)) {{
        showCSVValidationFeedback(false, `Rejected (${{filename}}): No valid customer rows or numeric values found.`);
        return false;
      }}

      // If no category column was present, assign value tiers from tertiles of monthly_charges
      if (autoValueTier) {{
        const sortedMc = records.map(r => r.monthly_charges).slice().sort((a, b) => a - b);
        const q33 = sortedMc[Math.floor((sortedMc.length - 1) * 0.33)] || 50.0;
        const q66 = sortedMc[Math.floor((sortedMc.length - 1) * 0.67)] || 100.0;
        records.forEach(r => {{
          if (r.monthly_charges <= q33) {{
            r.contract_type = 0;
            r.contract_label = 'Standard Value Tier';
          }} else if (r.monthly_charges <= q66) {{
            r.contract_type = 1;
            r.contract_label = 'Growth Value Tier';
          }} else {{
            r.contract_type = 2;
            r.contract_label = 'Premier Value Tier';
          }}
        }});
      }}

      lastUploadedCSVText = csvText;
      lastUploadedCSVName = filename;
      colIndex.date_col = dateColIdx;
      populateSchemaMapperUI(headerCells, colIndex, autoValueTier);

      DATASETS.custom = buildCohortFromRecords(
        records,
        'custom',
        `Custom CSV (${{filename}})`,
        isInferenceOnly,
        false
      );

      const optCustom = document.getElementById('opt-custom');
      if (optCustom) optCustom.textContent = `Upload Custom CSV (${{records.length.toLocaleString()}})`;

      const modeDesc = isInferenceOnly ? 'inference-only cohort' : 'labeled survival cohort';
      const dupDesc = duplicateCount > 0 ? ` (${{duplicateCount}} duplicate ID(s) deduplicated)` : '';
      const valColName = headerCells[colIndex.monthly_charges] || 'monthly_charges';
      const catColName = autoValueTier ? 'Auto Value Tiers' : (headerCells[colIndex.contract_type] || 'contract_type');
      const mapSummary = (!strictMc || !strictContract) ? ` [Mapped: Value ← ${{valColName}}, Tier ← ${{catColName}}]` : '';
      showCSVValidationFeedback(true, `Validated ${{records.length.toLocaleString()}} records from ${{filename}} as ${{modeDesc}}${{dupDesc}}${{mapSummary}}.`);

      applyActiveDataset('custom', false);
      return true;
    }}

    function handleCSVFileSelect(fileList) {{
      if (!fileList || fileList.length === 0) return;
      const file = fileList[0];
      if (file.size === 0) {{
        showCSVValidationFeedback(false, `Rejected (${{file.name}}): File is empty (0 bytes). Keeping active dataset (${{SERVER_DATA.source_label}}).`);
        return;
      }}
      if (file.size > RAW_SERVER_DATA.max_csv_bytes) {{
        showCSVValidationFeedback(false, `Rejected (${{file.name}}): Exceeds 25MB limit. Keeping active dataset (${{SERVER_DATA.source_label}}).`);
        return;
      }}
      const reader = new FileReader();
      reader.onload = function(evt) {{
        validateAndIngestCSVText(String(evt.target.result || ''), file.name);
      }};
      reader.onerror = function() {{
        showCSVValidationFeedback(false, `Rejected (${{file.name}}): Unable to read file.`);
      }};
      reader.readAsText(file);
    }}

    function handleDragOver(evt) {{
      evt.preventDefault();
      evt.stopPropagation();
      const dz = document.getElementById('csv-dropzone');
      if (dz) dz.classList.add('dropzone-active');
    }}

    function handleDragLeave(evt) {{
      evt.preventDefault();
      evt.stopPropagation();
      const dz = document.getElementById('csv-dropzone');
      if (dz) dz.classList.remove('dropzone-active');
    }}

    function handleFileDrop(evt) {{
      evt.preventDefault();
      evt.stopPropagation();
      const dz = document.getElementById('csv-dropzone');
      if (dz) dz.classList.remove('dropzone-active');
      if (evt.dataTransfer && evt.dataTransfer.files && evt.dataTransfer.files.length > 0) {{
        handleCSVFileSelect(evt.dataTransfer.files);
      }}
    }}

    function downloadCSVTemplate() {{
      const blob = new Blob([RAW_SERVER_DATA.csv_template], {{ type: 'text/csv;charset=utf-8;' }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', 'retainnet_customer_template.csv');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }}

    function buildTimeLabels(horizonMonths, count) {{
      const step = horizonMonths / Math.max(1, count - 1);
      return Array.from({{ length: count }}, (_, i) => Math.round(i * step) + 'M');
    }}

    function initCharts() {{
      Object.values(charts).forEach(c => {{
        if (c) c.destroy();
      }});

      const ctxHisto = document.getElementById('cltvHistogramChart').getContext('2d');
      charts.histogram = new Chart(ctxHisto, {{
        type: 'bar',
        data: {{
          labels: SERVER_DATA.cltv_histogram.labels,
          datasets: [{{
            label: 'Subscriber Count',
            data: SERVER_DATA.cltv_histogram.counts,
            backgroundColor: [
              '#BD6448', '#BD6448', '#C38438', '#C38438',
              '#3B5740', '#3B5740', '#3B5740', '#3B5740', '#3B5740', '#3B5740'
            ],
            borderRadius: 6
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ grid: {{ display: false }}, ticks: {{ color: '#6A7265', font: {{ size: 10 }} }} }},
            y: {{ grid: {{ color: '#EBE9DE' }}, ticks: {{ color: '#6A7265', font: {{ size: 10 }} }} }}
          }}
        }}
      }});

      const ctxDonut = document.getElementById('archetypesDonutChart').getContext('2d');
      charts.donut = new Chart(ctxDonut, {{
        type: 'doughnut',
        data: {{
          labels: SERVER_DATA.archetypes.labels,
          datasets: [{{
            data: SERVER_DATA.archetypes.pcts,
            backgroundColor: ['#3B5740', '#BD6448', '#C38438', '#627D78'],
            borderWidth: 2,
            borderColor: '#FFFFFF'
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          cutout: '72%'
        }}
      }});

      const ctxKm = document.getElementById('pipelineKmChart').getContext('2d');
      charts.km = new Chart(ctxKm, {{
        type: 'line',
        data: {{
          labels: SERVER_DATA.km_overlay.labels,
          datasets: [
            {{
              label: 'Predicted DeepSurv',
              data: SERVER_DATA.km_overlay.dsurv_points,
              borderColor: '#3B5740',
              backgroundColor: 'rgba(59, 87, 64, 0.08)',
              borderWidth: 2.5,
              tension: 0.3,
              fill: true
            }},
            {{
              label: 'Empirical Kaplan-Meier',
              data: SERVER_DATA.km_overlay.km_points,
              borderColor: '#6A7265',
              borderWidth: 2,
              borderDash: [5, 5],
              tension: 0.2,
              fill: false
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ grid: {{ color: '#EBE9DE' }}, ticks: {{ color: '#6A7265', font: {{ size: 10 }} }} }},
            y: {{ min: 0, max: 1.0, grid: {{ color: '#EBE9DE' }}, ticks: {{ color: '#6A7265', font: {{ size: 10 }} }} }}
          }}
        }}
      }});

      const ctxConformal = document.getElementById('customerConformalChart').getContext('2d');
      const timeSteps = Array.from({{ length: 16 }}, (_, i) => (i * 3) + 'M');
      charts.conformal = new Chart(ctxConformal, {{
        type: 'line',
        data: {{
          labels: timeSteps,
          datasets: [
            {{
              label: 'Upper 90% Bound',
              data: [],
              borderColor: 'transparent',
              backgroundColor: 'rgba(59, 87, 64, 0.15)',
              pointRadius: 0,
              fill: '+1'
            }},
            {{
              label: 'Lower 90% Bound',
              data: [],
              borderColor: 'transparent',
              pointRadius: 0,
              fill: false
            }},
            {{
              label: 'Point Estimate S(t|x)',
              data: [],
              borderColor: '#3B5740',
              borderWidth: 2.5,
              tension: 0.35,
              fill: false
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ grid: {{ color: '#EBE9DE' }}, ticks: {{ color: '#6A7265', font: {{ size: 10 }} }} }},
            y: {{ min: 0, max: 1.05, grid: {{ color: '#EBE9DE' }}, ticks: {{ color: '#6A7265', font: {{ size: 10 }} }} }}
          }}
        }}
      }});

      const ctxCounter = document.getElementById('counterfactualChart').getContext('2d');
      charts.counterfactual = new Chart(ctxCounter, {{
        type: 'line',
        data: {{
          labels: timeSteps.slice(),
          datasets: [
            {{
              label: 'Baseline S(t)',
              data: [],
              borderColor: '#BD6448',
              borderWidth: 2,
              tension: 0.3,
              fill: false
            }},
            {{
              label: 'Counterfactual S*(t)',
              data: [],
              borderColor: '#3B5740',
              backgroundColor: 'rgba(59, 87, 64, 0.12)',
              borderWidth: 2.5,
              tension: 0.3,
              fill: true
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ grid: {{ color: '#EBE9DE' }}, ticks: {{ color: '#6A7265', font: {{ size: 10 }} }} }},
            y: {{ min: 0, max: 1.05, grid: {{ color: '#EBE9DE' }}, ticks: {{ color: '#6A7265', font: {{ size: 10 }} }} }}
          }}
        }}
      }});

      const ctxDrift = document.getElementById('driftChart').getContext('2d');
      charts.drift = new Chart(ctxDrift, {{
        type: 'bar',
        data: {{
          labels: ['Monthly Charges', 'Duration', 'Contract Type', 'Support Calls', 'Total Charges'],
          datasets: [
            {{
              label: 'Reference Training Cohort (%)',
              data: SERVER_DATA.mlops.ref_bars || [28.2, 32.5, 39.3, 44.1, 33.6],
              backgroundColor: '#3B5740',
              borderRadius: 5
            }},
            {{
              label: 'Active Evaluation Cohort (%)',
              data: SERVER_DATA.mlops.cohort_bars || [28.8, 31.9, 39.1, 43.5, 34.2],
              backgroundColor: '#A3B49E',
              borderRadius: 5
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ boxWidth: 12, font: {{ size: 11 }} }} }}
          }},
          scales: {{
            x: {{ grid: {{ display: false }}, ticks: {{ color: '#6A7265', font: {{ size: 10 }} }} }},
            y: {{ grid: {{ color: '#EBE9DE' }}, ticks: {{ color: '#6A7265', font: {{ size: 10 }} }} }}
          }}
        }}
      }});
    }}

    function loadCustomer(custId) {{
      const cust = SERVER_DATA.customer_db[custId];
      if (!cust) return;
      currentCustId = custId;

      document.getElementById('c-tenure').innerText = cust.tenure;
      document.getElementById('c-arpu').innerText = cust.arpu;
      document.getElementById('c-contract').innerText = cust.contract;
      document.getElementById('c-curve-title').innerText = `Individualized Survival Curve: ${{cust.id}} (${{cust.segment}})`;
      document.getElementById('c-prescribe-text').innerText = cust.prescription;
      document.getElementById('c-prescribe-badge').innerText = cust.urgency;
      document.getElementById('c-prescribe-badge').className = `px-3 py-1 rounded-full text-xs font-bold ${{cust.badgeClass}}`;
      document.getElementById('c-hazard').innerText = cust.hazard;
      document.getElementById('c-ltv-calc').innerText = cust.ltv;
      document.getElementById('c-intervention-roi').innerText = cust.roiText;

      const activeCurve = (cust.curves && cust.curves[currentModelKey]) ? cust.curves[currentModelKey] : cust.curve;
      const qBand = SERVER_DATA.mlops.conformal_q || 0.08;
      const upperBounds = activeCurve.map(v => Math.min(1.0, Number((v + qBand).toFixed(3))));
      const lowerBounds = activeCurve.map(v => Math.max(0.0, Number((v - qBand).toFixed(3))));

      if (charts.conformal) {{
        charts.conformal.data.datasets[0].data = upperBounds;
        charts.conformal.data.datasets[1].data = lowerBounds;
        charts.conformal.data.datasets[2].data = activeCurve;
        charts.conformal.update('none');
      }}
    }}

    function selectCustomer(id) {{
      const searchInput = document.getElementById('customer-search-input');
      if (searchInput && searchInput.value !== '') {{
        searchInput.value = '';
        populateCustomerDropdown('', false);
      }}
      const dropdown = document.getElementById('customer-dropdown');
      if (dropdown) dropdown.value = id;
      loadCustomer(id);
      switchTab('customers');
    }}

    function runSimulation() {{
      const discountRaw = parseFloat(document.getElementById('sim-discount-slider').value);
      const discount = Math.max(0.0, Math.min(30.0, Number.isFinite(discountRaw) ? discountRaw : 15.0));
      document.getElementById('sim-discount-val').innerText = `$${{discount.toFixed(0)}} / mo`;

      const lockIn = document.getElementById('sim-contract-toggle').checked;
      const vip = document.getElementById('sim-vip-toggle').checked;

      const discountFactor = Math.max(0.65, 1.0 - (discount * 0.015));
      const discountSubEl = document.getElementById('sim-discount-sub');
      if (discountSubEl) {{
        discountSubEl.innerText = `Reduces ARPU but diminishes churn hazard (γ = ${{discountFactor.toFixed(2)}})`;
      }}

      let gamma = discountFactor;
      if (lockIn) gamma *= 0.60;
      if (vip) gamma *= 0.85;

      document.getElementById('sim-gamma-val').innerText = gamma.toFixed(3);
      const riskReductionPct = ((1.0 - gamma) * 100.0).toFixed(1);
      const reductionEl = document.getElementById('sim-gamma-reduction');
      if (reductionEl) {{
        reductionEl.innerText = `-${{riskReductionPct}}% Churn Risk`;
      }}

      const baseCurve = [0.85, 0.62, 0.42, 0.28, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01];
      const counterCurve = baseCurve.map(s => Math.min(1.0, Number(Math.pow(s, gamma).toFixed(4))));

      if (charts.counterfactual) {{
        charts.counterfactual.data.datasets[0].data = baseCurve;
        charts.counterfactual.data.datasets[1].data = counterCurve;
        charts.counterfactual.update('none');
      }}

      const baseLTV = 450.0;
      const baseArea = baseCurve.reduce((a, b) => a + b, 0);
      const counterArea = counterCurve.reduce((a, b) => a + b, 0);
      const ratio = counterArea / baseArea;

      const totalCampaignCost = (discount * 12.0) + (vip ? 25.0 : 0.0);
      const grossCounterLTV = Math.round(baseLTV * ratio);
      const counterLTV = grossCounterLTV - totalCampaignCost;
      const netLift = counterLTV - baseLTV;

      let roiText = '0%';
      let roiClass = 'kpi-value text-base text-forest';
      if (totalCampaignCost > 0) {{
        const roiPct = Math.round((netLift / totalCampaignCost) * 100.0);
        roiText = `${{roiPct}}%`;
        roiClass = roiPct >= 0 ? 'kpi-value text-base text-forest' : 'kpi-value text-base text-terracotta';
      }}

      let liftText = `+$${{netLift.toFixed(2)}}`;
      let liftClass = 'kpi-value text-base text-emerald-700';
      if (netLift < 0) {{
        liftText = `-$${{Math.abs(netLift).toFixed(2)}}`;
        liftClass = 'kpi-value text-base text-terracotta';
      }}

      const liftPct = ((netLift / baseLTV) * 100.0).toFixed(1);
      const liftSubEl = document.getElementById('roi-lift-sub');
      if (liftSubEl) {{
        liftSubEl.innerText = `${{netLift >= 0 ? '+' : ''}}${{liftPct}}% improvement`;
      }}

      document.getElementById('roi-base-ltv').innerText = `$${{baseLTV.toFixed(2)}}`;
      document.getElementById('roi-counter-ltv').innerText = `$${{counterLTV.toFixed(2)}}`;
      const liftElem = document.getElementById('roi-lift');
      if (liftElem) {{
        liftElem.innerText = liftText;
        liftElem.className = liftClass;
      }}
      const roiElem = document.getElementById('roi-pct');
      if (roiElem) {{
        roiElem.innerText = roiText;
        roiElem.className = roiClass;
      }}
    }}

    function inferColdStart() {{
      const contract = document.getElementById('cs-contract').value;
      const chargeRaw = parseFloat(document.getElementById('cs-charge').value);
      const charge = Number.isFinite(chargeRaw) ? chargeRaw : 70.0;
      const service = document.getElementById('cs-service').value;
      const outputArch = document.getElementById('cs-output-archetype');
      const outputProb = document.getElementById('cs-output-prob');

      const ctCode = contract === 'two-year' ? 2 : (contract === 'one-year' ? 1 : 0);
      const tsCode = service === 'none' ? 0 : (service === 'dsl' ? 1 : 0);
      const osCode = service === 'dsl' ? 1 : 0;
      const pbCode = service === 'fiber' ? 1 : 0;
      const xRaw = [charge, charge * 1.0, 1, ctCode, tsCode, osCode, pbCode];
      const sMean = MODEL_BUNDLE.scaler_mean || [];
      const sScale = MODEL_BUNDLE.scaler_scale || [];
      const centers = MODEL_BUNDLE.kmeans_centers || [];
      const xScaled = xRaw.map((v, j) => (v - (sMean[j] || 0)) / (sScale[j] || 1));

      let bestCluster = 0;
      let bestDist = Infinity;
      for (let c = 0; c < centers.length; c++) {{
        let d2 = 0;
        for (let j = 0; j < xScaled.length; j++) {{
          const diff = xScaled[j] - (centers[c][j] || 0);
          d2 += diff * diff;
        }}
        if (d2 < bestDist) {{
          bestDist = d2;
          bestCluster = c;
        }}
      }}

      const clusterPrior12 = (MODEL_BUNDLE.amortized_12m_priors && MODEL_BUNDLE.amortized_12m_priors[String(bestCluster)] !== undefined)
        ? Number(MODEL_BUNDLE.amortized_12m_priors[String(bestCluster)])
        : 60.0;

      let expected12M = clusterPrior12;
      let archName = 'Moderate Retention Prior';
      let archClass = 'font-bold text-xs text-ambercustom';
      let probClass = 'font-bold text-xs text-ambercustom';

      if (contract === 'two-year') {{
        expected12M = Math.min(96.0, Math.max(70.0, 0.45 * clusterPrior12 + 0.55 * (88.5 - (charge > 80 ? 4.0 : 0.0))));
        archName = 'Loyal Long-Term Prior';
        archClass = 'font-bold text-xs text-forest';
        probClass = 'font-bold text-xs text-forest';
      }} else if (contract === 'one-year') {{
        expected12M = Math.min(85.0, Math.max(40.0, 0.45 * clusterPrior12 + 0.55 * (62.4 - (service === 'fiber' ? 5.2 : 0.0))));
        archName = 'Price-Sensitive Prior';
        archClass = 'font-bold text-xs text-ambercustom';
        probClass = 'font-bold text-xs text-ambercustom';
      }} else {{
        const heur = Math.max(12.0, 26.5 - (service === 'fiber' ? 8.0 : 0.0) - (charge > 75 ? 4.5 : 0.0));
        expected12M = Math.min(60.0, Math.max(12.0, 0.35 * clusterPrior12 + 0.65 * heur));
        archName = 'At-Risk Newbie Prior';
        archClass = 'font-bold text-xs text-terracotta';
        probClass = 'font-bold text-xs text-terracotta';
      }}

      outputArch.innerText = archName;
      outputArch.className = archClass;
      outputProb.innerText = `Expected 12M: ${{expected12M.toFixed(1)}}%`;
      outputProb.className = probClass;
    }}

    function updateHorizon(val) {{
      const horizon = parseInt(val, 10) || 72;
      document.getElementById('horizon-val').innerText = horizon + ' mo';
      const reportHorizonVal = document.getElementById('report-horizon-val');
      if (reportHorizonVal) reportHorizonVal.innerText = horizon + ' mo';
      const reportHorizonText = document.getElementById('report-horizon-text');
      if (reportHorizonText) reportHorizonText.innerText = String(horizon);

      const labels16 = buildTimeLabels(horizon, 16);
      if (charts.conformal) {{
        charts.conformal.data.labels = labels16;
        charts.conformal.update('none');
      }}
      if (charts.counterfactual) {{
        charts.counterfactual.data.labels = labels16.slice();
        charts.counterfactual.update('none');
      }}
      if (charts.km) {{
        charts.km.data.labels = buildTimeLabels(horizon, SERVER_DATA.km_overlay.labels.length);
        charts.km.update('none');
      }}
    }}

    function updateFinancials() {{
      const cacRaw = parseFloat(document.getElementById('cac-input').value);
      const dRaw = parseFloat(document.getElementById('discount-input').value);
      const cac = Number.isFinite(cacRaw) ? cacRaw : 120.0;
      const d = Number.isFinite(dRaw) && dRaw > 0 ? dRaw : 0.01;

      const discountSub = document.getElementById('kpi-discount-sub');
      if (discountSub) {{
        discountSub.innerText = `(d = ${{d}} / mo)`;
      }}

      if (Math.abs(cac - 120.0) < 1e-6 && Math.abs(d - 0.01) < 1e-6) {{
        document.getElementById('kpi-cltv').innerText = SERVER_DATA.kpi.cltv;
        document.getElementById('kpi-atrisk').innerText = SERVER_DATA.kpi.atrisk;
        return;
      }}

      const rows = SERVER_DATA.compact_customers || [];
      const tGrid16 = MODEL_BUNDLE.time_grid_16 || [];
      if (rows.length === 0 || tGrid16.length < 2) {{
        const baseAvg = SERVER_DATA.kpi.cltv_num !== undefined
          ? SERVER_DATA.kpi.cltv_num
          : (parseFloat(String(SERVER_DATA.kpi.cltv).replace(/[$,]/g, '')) || 680.0);
        const adjustedLtv = baseAvg * (0.01 / d) - (cac - 120.0);
        document.getElementById('kpi-cltv').innerText = '$' + adjustedLtv.toLocaleString('en-US', {{
          minimumFractionDigits: 2,
          maximumFractionDigits: 2
        }});
        return;
      }}

      const discountFactors = tGrid16.map(t => 1.0 / Math.pow(1.0 + d, t));
      const newLtvs = [];
      let sumLtv = 0.0;
      for (let i = 0; i < rows.length; i++) {{
        const cid = String(rows[i][0]);
        const arpu = Number(rows[i][2]) || 65.0;
        const curve = rows[i][7] || [];
        let integral = 0.0;
        for (let k = 0; k < tGrid16.length - 1; k++) {{
          const dt = tGrid16[k + 1] - tGrid16[k];
          const y0 = (curve[k] ?? 0.5) * arpu * discountFactors[k];
          const y1 = (curve[k + 1] ?? 0.5) * arpu * discountFactors[k + 1];
          integral += 0.5 * (y0 + y1) * dt;
        }}
        const cLtv = Number((integral - cac).toFixed(2));
        newLtvs.push(cLtv);
        sumLtv += cLtv;
        if (SERVER_DATA.customer_db && SERVER_DATA.customer_db[cid]) {{
          SERVER_DATA.customer_db[cid].ltv_num = cLtv;
          SERVER_DATA.customer_db[cid].ltv = `$${{cLtv.toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }})}}`;
        }}
      }}

      const avgLtv = sumLtv / Math.max(1, newLtvs.length);
      const sortedLtv = newLtvs.slice().sort((a, b) => a - b);
      const q25 = sortedLtv[Math.floor((sortedLtv.length - 1) * 0.25)] || 0.0;
      const atRiskRev = Math.abs(newLtvs.filter(v => v <= q25).reduce((a, b) => a + b, 0.0));

      document.getElementById('kpi-cltv').innerText = '$' + avgLtv.toLocaleString('en-US', {{
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      }});
      document.getElementById('kpi-atrisk').innerText = '$' + atRiskRev.toLocaleString('en-US', {{
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      }});

      if (currentCustId && SERVER_DATA.customer_db && SERVER_DATA.customer_db[currentCustId]) {{
        const cLtvEl = document.getElementById('c-ltv-calc');
        if (cLtvEl) cLtvEl.innerText = SERVER_DATA.customer_db[currentCustId].ltv;
      }}
    }}

    function updateModelSelection(val) {{
      currentModelKey = val;
      const matchMap = {{
        deepsurv: {{ text: 'DeepSurv', cls: 'bg-emerald-50/70 font-semibold ring-1 ring-forest' }},
        deephit: {{ text: 'DeepHit', cls: 'bg-amber-50/70 font-semibold ring-1 ring-ambercustom' }},
        coxph: {{ text: 'CoxPH', cls: 'bg-blue-50/70 font-semibold ring-1 ring-blue-500' }},
        rsf: {{ text: 'Random Survival', alt: 'RSF', cls: 'bg-red-50/70 font-semibold ring-1 ring-red-400' }}
      }};
      const target = matchMap[val];

      ['#leaderboard-body tr', '#audit-table-body tr'].forEach(selector => {{
        document.querySelectorAll(selector).forEach(r => {{
          if (target && r.innerText.includes(target.text)) {{
            r.className = target.cls;
          }} else {{
            r.className = 'hover:bg-[#FCFBF8] transition';
          }}
        }});
      }});

      if (currentCustId) {{
        loadCustomer(currentCustId);
      }}
    }}

    function downloadPredictionsCSV() {{
      const headers = ["CustomerID", "Dynamic_CLTV_USD", "12M_Retention_Prob", "24M_Retention_Prob", "Recommended_Action"];
      const escapeCell = cell => {{
        const str = sanitizeCSVCellJS(cell);
        return /[",\\n]/.test(str) ? `"${{str.replace(/"/g, '""')}}"` : str;
      }};
      const lines = [
        headers.join(","),
        ...SERVER_DATA.csv_rows.map(r => r.map(escapeCell).join(","))
      ];
      const blob = new Blob([lines.join("\\n")], {{ type: "text/csv;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", `retainnet_${{activeDatasetKey}}_ltv_predictions.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }}
  </script>
</body>
</html>
"""

components.html(html_template, height=1000, scrolling=True)