"""
Dashboard sub-package for RetainNet UI and service layer.
"""
from .config import (
    MAX_CSV_UPLOAD_BYTES,
    FEATURE_COLUMNS,
    STRICT_CUSTOMER_ID_KEYS,
    STRICT_MONTHLY_CHARGE_KEYS,
    STRICT_CONTRACT_KEYS,
    SIMULATION_POLICY_CONFIG,
    COLUMN_ALIASES,
)
from .recommendations import classify_customer_retention_tier

__all__ = [
    "MAX_CSV_UPLOAD_BYTES",
    "FEATURE_COLUMNS",
    "STRICT_CUSTOMER_ID_KEYS",
    "STRICT_MONTHLY_CHARGE_KEYS",
    "STRICT_CONTRACT_KEYS",
    "SIMULATION_POLICY_CONFIG",
    "COLUMN_ALIASES",
    "classify_customer_retention_tier",
]

