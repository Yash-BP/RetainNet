"""
RetainNet CSV Data Validator & Universal Adaptive Schema Ingestion Engine.
Provides robust schema validation, universal column alias mapping (Telco, Retail Sales,
E-Commerce, SaaS), currency/comma numeric cleaning, formula injection protection,
range checking, duplicate ID handling, and inference-only detection.
"""
import io
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from retainnet.ltv.calculator import sanitize_csv_cell


@dataclass
class ValidationResult:
    """Structured result returned by CSV validation."""
    is_valid: bool
    df: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    is_inference_only: bool = False
    warnings: List[str] = field(default_factory=list)
    record_count: int = 0
    feature_columns: List[str] = field(default_factory=list)
    column_mapping: Dict[str, str] = field(default_factory=dict)


# Column aliases to map Telco, Retail Sales, E-Commerce, and SaaS CSV naming styles to RetainNet canonical columns
COLUMN_ALIASES = {
    'duration': [
        'duration', 'tenure', 'months', 'tenure_months', 'time',
        'days', 'age', 'account_age', 'customer_tenure', 'subscription_length', 'period'
    ],
    'event': [
        'event', 'churn', 'churned', 'status', 'is_churn', 'ischurn', 'churn_label',
        'cancelled', 'canceled', 'attrition', 'returned', 'is_returned', 'closed', 'inactive'
    ],
    'monthly_charges': [
        'monthly_charges', 'monthlycharges', 'monthly_fee', 'monthlyfee', 'arpu', 'monthly_bill', 'mrr',
        'sales', 'daily_sales', 'dailysales', 'total_sales', 'totalsales', 'net_sales', 'netsales',
        'gross_sales', 'grosssales', 'revenue', 'daily_revenue', 'dailyrevenue', 'monthly_revenue',
        'monthlyrevenue', 'amount', 'sale_amount', 'sales_amount', 'order_amount', 'transaction_amount',
        'price', 'unit_price', 'unitprice', 'selling_price', 'spend', 'monthly_spend', 'value',
        'order_value', 'profit', 'gross_profit', 'margin', 'charge', 'charges', 'fee',
        'billing_amount', 'cost', 'payment', 'payment_amount'
    ],
    'total_charges': [
        'total_charges', 'totalcharges', 'total_billed', 'lifetime_charges',
        'total_spend', 'totalspend', 'lifetime_spend', 'cumulative_sales', 'cumulativesales',
        'ytd_sales', 'annual_sales', 'arr', 'historical_ltv', 'total_revenue', 'gross_revenue'
    ],
    'contract_type': [
        'contract_type', 'contract', 'contracttype', 'plan_type', 'plan', 'subscription_tier', 'tier',
        'category', 'product_category', 'productcategory', 'item_category', 'itemcategory',
        'department', 'segment', 'customer_segment', 'channel', 'sales_channel',
        'store_type', 'store', 'region', 'location', 'payment_method', 'payment_type',
        'type', 'status_tier', 'group'
    ],
    'customer_id': [
        'customer_id', 'customerid', 'id', 'user_id', 'userid', 'subscriber_id', 'subscriberid',
        'account_id', 'accountid', 'client_id', 'clientid', 'member_id', 'memberid',
        'store_id', 'storeid', 'transaction_id', 'transactionid', 'order_id', 'orderid',
        'invoice_id', 'invoiceid', 'invoice_no', 'invoiceno', 'product_id', 'productid',
        'item_id', 'itemid', 'sku', 'record_id', 'row_id'
    ],
    'support_tickets': [
        'support_tickets', 'supporttickets', 'support_calls', 'supportcalls', 'tickets',
        'customer_service_calls', 'num_support_tickets', 'calls', 'quantity', 'qty',
        'units', 'units_sold', 'unitssold', 'items', 'item_count', 'transactions',
        'transaction_count', 'orders', 'order_count', 'visits', 'footfall', 'sessions',
        'returns', 'complaints', 'discounts'
    ],
    'tech_support': [
        'tech_support', 'techsupport', 'has_tech_support', 'promo', 'promotion', 'is_promo'
    ],
    'online_security': [
        'online_security', 'onlinesecurity', 'has_online_security', 'loyalty_member', 'is_vip'
    ],
    'paperless_billing': [
        'paperless_billing', 'paperlessbilling', 'paperless', 'online_order', 'is_online'
    ]
}

MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB limit (unified with dashboard.py)


def generate_csv_template() -> str:
    """Generates a downloadable CSV template with standard RetainNet schema and example rows."""
    header = "customer_id,monthly_charges,total_charges,contract_type,duration,event,support_tickets,tech_support,online_security,paperless_billing"
    rows = [
        "CUST-0001,65.50,1572.00,0,24.0,1,2,0,0,1",
        "CUST-0002,89.90,3236.40,1,36.0,0,1,1,1,0",
        "CUST-0003,24.20,290.40,2,12.0,0,0,1,0,0",
        "CUST-0004,105.00,5040.00,0,48.0,1,4,0,1,1",
        "CUST-0005,45.30,815.40,1,18.0,0,1,0,0,1"
    ]
    return header + "\n" + "\n".join(rows) + "\n"


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    """Parses numeric series, stripping currency symbols ($/€/£/₹) and thousand-separator commas if string-typed."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"^[$€£₹\s]+", "", regex=True)
        .str.replace(",", "", regex=False)
        .str.replace(r"^(nan|none|null|\s*)$", "NaN", flags=re.IGNORECASE, regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _encode_contract_or_category(series: pd.Series) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Encodes standard Telco contract strings/numbers into {0, 1, 2}, or adaptively encodes
    arbitrary retail/SaaS categories into {0, 1, 2} while returning human-readable labels.
    """
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce").fillna(0)
        if set(np.unique(vals)).issubset({0, 1, 2, 0.0, 1.0, 2.0}):
            return vals.astype(int), None

    mapping = {
        "0": 0, "1": 1, "2": 2,
        "month-to-month": 0, "month to month": 0, "monthly": 0, "m2m": 0,
        "one year": 1, "one-year": 1, "1 year": 1, "1-year": 1, "annual": 1,
        "two year": 2, "two-year": 2, "2 year": 2, "2-year": 2, "biennial": 2,
    }
    normalized = series.fillna("month-to-month").astype(str).str.strip().str.lower()
    mapped = normalized.map(mapping)
    if not mapped.isna().any():
        return mapped.astype(int), None

    clean_str = series.fillna("Standard").astype(str).str.strip().replace("", "Standard")
    unique_vals = list(dict.fromkeys(clean_str))
    tier_lookup = {val: (idx % 3) for idx, val in enumerate(unique_vals)}
    encoded = clean_str.map(lambda x: tier_lookup.get(x, 0)).astype(int)
    return encoded, clean_str


def _normalize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Map user columns to canonical RetainNet column names based on aliases."""
    rename_map: Dict[str, str] = {}
    col_lower_map: Dict[str, str] = {}
    for c in df.columns:
        norm = str(c).lower().strip().replace(" ", "_").replace("-", "_")
        col_lower_map[norm] = c
        col_lower_map[norm.replace("_", "")] = c

    used_orig_cols = set()
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            used_orig_cols.add(canonical)
            continue
        for alias in aliases:
            norm_alias = alias.lower().strip().replace(" ", "_").replace("-", "_")
            orig_col = col_lower_map.get(norm_alias, col_lower_map.get(norm_alias.replace("_", "")))
            if orig_col is not None and orig_col not in used_orig_cols:
                rename_map[orig_col] = canonical
                used_orig_cols.add(orig_col)
                break

    if rename_map:
        df = df.rename(columns=rename_map)
    return df, rename_map


def validate_customer_csv(
    file_or_data: Union[str, bytes, io.BytesIO, io.StringIO, pd.DataFrame],
    filename: Optional[str] = None,
    max_size_bytes: int = MAX_UPLOAD_SIZE_BYTES
) -> ValidationResult:
    """
    Safely parses, validates, and adapts an uploaded customer or business CSV dataset.

    Validates:
    - File size constraints (up to 25 MB)
    - Non-empty content and valid CSV structure
    - Schema requirements (duration, event for survival analysis; features for inference;
      supports Telco, Retail Sales, E-Commerce, and SaaS schemas)
    - Currency/thousand-separator cleaning (e.g. "$1,240.50" -> 1240.50)
    - Numeric ranges (duration >= 0, charges >= 0, binary event in {0, 1})
    - Stable, unique, sanitized customer IDs
    - Formula injection defense on all text/ID cells
    - Graceful handling of missing values
    """
    warnings_list: List[str] = []

    # 1. Read input safely into a DataFrame
    try:
        if isinstance(file_or_data, pd.DataFrame):
            df = file_or_data.copy()
        elif isinstance(file_or_data, bytes):
            if len(file_or_data) > max_size_bytes:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Uploaded file exceeds maximum allowable size of {max_size_bytes // (1024 * 1024)} MB."
                )
            if len(file_or_data.strip()) == 0:
                return ValidationResult(
                    is_valid=False,
                    error_message="Uploaded CSV file is completely empty."
                )
            if b"\x00" in file_or_data:
                return ValidationResult(
                    is_valid=False,
                    error_message="Unsupported binary content or null bytes detected in CSV."
                )
            try:
                df = pd.read_csv(io.BytesIO(file_or_data), encoding='utf-8-sig')
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(file_or_data), encoding='latin-1')
        elif isinstance(file_or_data, str):
            if os.path.isfile(file_or_data):
                if os.path.getsize(file_or_data) > max_size_bytes:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"File exceeds maximum allowable size of {max_size_bytes // (1024 * 1024)} MB."
                    )
                df = pd.read_csv(file_or_data)
            else:
                if not file_or_data.strip():
                    return ValidationResult(
                        is_valid=False,
                        error_message="Uploaded CSV content is completely empty."
                    )
                if "\x00" in file_or_data:
                    return ValidationResult(
                        is_valid=False,
                        error_message="Unsupported binary content or null bytes detected in CSV."
                    )
                df = pd.read_csv(io.StringIO(file_or_data))
        elif hasattr(file_or_data, 'read'):
            content = file_or_data.read()
            if hasattr(file_or_data, 'seek'):
                file_or_data.seek(0)
            if len(content) > max_size_bytes:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Uploaded file exceeds maximum allowable size of {max_size_bytes // (1024 * 1024)} MB."
                )
            if len(content.strip()) == 0:
                return ValidationResult(
                    is_valid=False,
                    error_message="Uploaded CSV file is completely empty."
                )
            try:
                df = pd.read_csv(
                    io.BytesIO(content) if isinstance(content, bytes) else io.StringIO(content),
                    encoding='utf-8-sig'
                )
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), encoding='latin-1')
        else:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unsupported input type for CSV validation: {type(file_or_data).__name__}"
            )
    except pd.errors.EmptyDataError:
        return ValidationResult(
            is_valid=False,
            error_message="Uploaded CSV contains headers but zero data records, or is completely blank."
        )
    except pd.errors.ParserError as pe:
        return ValidationResult(
            is_valid=False,
            error_message=f"Malformed CSV parsing error: {str(pe)}"
        )
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Error reading CSV file: {str(e)}"
        )

    # 2. Check record count
    if len(df) == 0:
        return ValidationResult(
            is_valid=False,
            error_message="Uploaded CSV contains zero data rows."
        )

    # 3. Normalize column names using aliases
    df, rename_map = _normalize_column_names(df)
    column_mapping: Dict[str, str] = {v: k for k, v in rename_map.items()}
    if rename_map:
        for orig, canonical in rename_map.items():
            warnings_list.append(f"Mapped column '{orig}' to canonical '{canonical}'.")

    # 4. Check for Survival Targets vs Inference-Only mode
    has_duration = 'duration' in df.columns
    has_event = 'event' in df.columns
    is_inference_only = False

    if not has_duration and not has_event:
        is_inference_only = True
        warnings_list.append(
            "Target columns ('duration' and 'event') are absent. Loaded in Inference-Only mode; survival benchmarking metrics disabled."
        )
    elif not has_duration:
        return ValidationResult(
            is_valid=False,
            error_message="Missing required survival duration column ('duration' or 'tenure'). Required for survival analysis."
        )
    elif not has_event:
        return ValidationResult(
            is_valid=False,
            error_message="Missing required event indicator column ('event' or 'churn'). Required for survival analysis."
        )

    # 5. Check customer / business features (supports both Telco and adaptive Retail/SaaS columns)
    RECOGNIZED_CUSTOMER_FEATURES = {
        'monthly_charges', 'total_charges', 'contract_type', 'support_tickets',
        'support_calls', 'tech_support', 'online_security', 'paperless_billing',
        'num_services', 'gender', 'senior_citizen', 'partner', 'dependents',
        'payment_method', 'tenure', 'contract'
    }

    present_recognized = [c for c in df.columns if c in RECOGNIZED_CUSTOMER_FEATURES]

    # Auto-discover monthly_charges from an unmapped positive numeric column if no recognized feature matched
    # (only when the CSV is well-formed without completely empty/truncated columns)
    if len(present_recognized) == 0 and not any(df[c].isna().all() for c in df.columns):
        for cand_col in df.columns:
            if cand_col in ('duration', 'event', 'customer_id'):
                continue
            cand_clean = str(cand_col).strip().lower()
            if any(k in cand_clean for k in ('id', 'code', 'uuid', 'phone', 'zip', 'postal', 'year', 'date')):
                continue
            cand_num = _clean_numeric_series(df[cand_col])
            if cand_num.notna().mean() >= 0.60 and (cand_num.dropna() > 0).any() and not (cand_num.dropna() < 0).any():
                df['monthly_charges'] = cand_num
                column_mapping['monthly_charges'] = str(cand_col)
                warnings_list.append(f"Auto-mapped numeric column '{cand_col}' to 'monthly_charges'.")
                present_recognized.append('monthly_charges')
                break

    feature_candidates = [
        c for c in df.columns
        if c not in ['duration', 'event', 'customer_id']
    ]

    if len(present_recognized) == 0:
        return ValidationResult(
            is_valid=False,
            error_message=(
                "Unsupported CSV schema: Dataset does not contain recognized customer features. "
                "Expected fields such as 'monthly_charges', 'contract_type', 'total_charges', "
                "'tech_support', 'duration', or retail/sales value columns."
            )
        )

    if len(feature_candidates) == 0:
        return ValidationResult(
            is_valid=False,
            error_message="CSV contains only target/ID columns without any customer features for inference."
        )

    # 6. Validate & Sanitize Customer IDs
    if 'customer_id' not in df.columns:
        df['customer_id'] = [f"UPLOAD-{i+1:05d}" for i in range(len(df))]
        warnings_list.append("No customer ID column provided. Auto-generated unique customer IDs ('UPLOAD-00001', ...).")
    else:
        df['customer_id'] = df['customer_id'].astype(str).apply(sanitize_csv_cell)

        if df['customer_id'].duplicated().any():
            dup_count = df['customer_id'].duplicated().sum()
            warnings_list.append(f"Found {dup_count} duplicate customer IDs. Made IDs unique to preserve record tracking.")
            seen = {}
            unique_ids = []
            for cid in df['customer_id']:
                if cid not in seen:
                    seen[cid] = 0
                    unique_ids.append(cid)
                else:
                    seen[cid] += 1
                    unique_ids.append(f"{cid}_dup{seen[cid]}")
            df['customer_id'] = unique_ids

    # 7. Validate Numeric Ranges on Duration & Event (if present)
    if not is_inference_only:
        df['duration'] = _clean_numeric_series(df['duration'])
        if df['duration'].isna().all():
            return ValidationResult(
                is_valid=False,
                error_message="The 'duration' column contains no valid numeric values."
            )
        if (df['duration'] < 0).any():
            neg_count = int((df['duration'] < 0).sum())
            return ValidationResult(
                is_valid=False,
                error_message=f"Found {neg_count} records with negative duration values. Tenure/duration must be non-negative."
            )
        if df['duration'].isna().any():
            med_dur = float(df['duration'].median()) if not pd.isna(df['duration'].median()) else 1.0
            df['duration'] = df['duration'].fillna(max(1.0, med_dur))
            warnings_list.append(f"Imputed missing duration values with cohort median ({med_dur:.1f} months).")

        if df['event'].dtype == 'object':
            event_str = df['event'].astype(str).str.strip().str.lower()
            mapped_ev = event_str.map({
                'yes': 1, 'y': 1, 'true': 1, '1': 1, '1.0': 1, 'churned': 1, 'churn': 1,
                'no': 0, 'n': 0, 'false': 0, '0': 0, '0.0': 0, 'active': 0, 'retained': 0
            })
            if mapped_ev.isna().any():
                invalid_vals = df[mapped_ev.isna()]['event'].unique()[:3]
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Event column contains unrecognized binary status values: {invalid_vals}. Expected binary 0/1 or Yes/No."
                )
            df['event'] = mapped_ev.astype(int)
        else:
            df['event'] = pd.to_numeric(df['event'], errors='coerce')
            if df['event'].isna().all():
                return ValidationResult(
                    is_valid=False,
                    error_message="The 'event' column contains no valid numeric indicators."
                )
            unique_events = set(df['event'].dropna().unique())
            if not unique_events.issubset({0, 1, 0.0, 1.0}):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Event column must contain binary indicators {{0, 1}}. Found values: {sorted(list(unique_events))}."
                )
            if df['event'].isna().any():
                df['event'] = df['event'].fillna(0).astype(int)
                warnings_list.append("Imputed missing event values as censored (event=0).")
            else:
                df['event'] = df['event'].astype(int)

    # 8. Sanitize & Validate Key Financial & Categorical Features
    if 'monthly_charges' in df.columns:
        df['monthly_charges'] = _clean_numeric_series(df['monthly_charges'])
        if df['monthly_charges'].isna().all():
            return ValidationResult(
                is_valid=False,
                error_message="Column 'monthly_charges' contains no valid numeric values."
            )
        if (df['monthly_charges'] < 0).any():
            return ValidationResult(
                is_valid=False,
                error_message="Monthly charges cannot be negative."
            )
        if df['monthly_charges'].isna().any():
            med_mc = float(df['monthly_charges'].median()) if not pd.isna(df['monthly_charges'].median()) else 65.0
            df['monthly_charges'] = df['monthly_charges'].fillna(med_mc)
            warnings_list.append(f"Imputed missing monthly_charges with median (${med_mc:.2f}).")

    if 'total_charges' in df.columns:
        df['total_charges'] = _clean_numeric_series(df['total_charges'])
        if (df['total_charges'] < 0).any():
            return ValidationResult(
                is_valid=False,
                error_message="Total charges cannot be negative."
            )
        if df['total_charges'].isna().any():
            if 'monthly_charges' in df.columns and 'duration' in df.columns:
                df['total_charges'] = df['total_charges'].fillna(df['monthly_charges'] * df['duration'])
            else:
                med_tc = float(df['total_charges'].median()) if not pd.isna(df['total_charges'].median()) else 0.0
                df['total_charges'] = df['total_charges'].fillna(med_tc)

    if 'support_tickets' in df.columns:
        df['support_tickets'] = _clean_numeric_series(df['support_tickets'])
        if (df['support_tickets'] < 0).any():
            return ValidationResult(
                is_valid=False,
                error_message="Support tickets / units cannot be negative."
            )
        df['support_tickets'] = df['support_tickets'].fillna(1.0).round(0).astype(int)

    if 'contract_type' in df.columns:
        encoded_ct, contract_labels = _encode_contract_or_category(df['contract_type'])
        df['contract_type'] = encoded_ct
        if contract_labels is not None:
            df['contract_label'] = contract_labels.astype(str).apply(sanitize_csv_cell)

    # 9. Sanitize remaining text columns against CSV formula injection
    text_cols = [c for c in df.columns if df[c].dtype == 'object' and c != 'customer_id']
    for tc in text_cols:
        df[tc] = df[tc].astype(str).apply(sanitize_csv_cell)

    return ValidationResult(
        is_valid=True,
        df=df,
        is_inference_only=is_inference_only,
        warnings=warnings_list,
        record_count=len(df),
        feature_columns=feature_candidates,
        column_mapping=column_mapping,
    )