"""
Configuration constants, aliases, and simulation policy assumptions for RetainNet Dashboard.
"""
from typing import Dict, Any

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

# Centralized What-If Simulator Policy Assumptions & Sensitivity Multipliers
SIMULATION_POLICY_CONFIG: Dict[str, Any] = {
    "discount_hazard_slope": 0.015,       # Marginal churn hazard dampening per $1/mo discount
    "discount_min_factor": 0.65,          # Maximum hazard dampening floor from price reduction
    "lockin_contract_gamma": 0.60,        # 12-Month Lock-in contract sensitivity multiplier
    "vip_concierge_gamma": 0.85,          # VIP concierge sensitivity multiplier
    "vip_concierge_cost": 25.0,           # One-time intervention acquisition/retention cost ($)
    "disclaimer": (
        "Simulation projections reflect user-configured sensitivity multipliers (proportional hazard assumptions) "
        "rather than randomized experimental or causal counterfactual estimates."
    ),
}

COLUMN_ALIASES = {
    "customerid": "customer_id",
    "customer_id": "customer_id",
    "subscriber_id": "customer_id",
    "subscriberid": "customer_id",
    "id": "customer_id",
    "user_id": "customer_id",
    "userid": "customer_id",
    "client_id": "customer_id",
    "monthlycharges": "monthly_charges",
    "monthly_charges": "monthly_charges",
    "monthly_fee": "monthly_charges",
    "monthlyfee": "monthly_charges",
    "arpu": "monthly_charges",
    "mrr": "monthly_charges",
    "totalcharges": "total_charges",
    "total_charges": "total_charges",
    "tenure": "duration",
    "duration": "duration",
    "months": "duration",
    "churn": "event",
    "event": "event",
    "churned": "event",
    "status": "event",
    "contract": "contract_type",
    "contract_type": "contract_type",
    "contracttype": "contract_type",
    "supporttickets": "support_tickets",
    "support_tickets": "support_tickets",
    "tickets": "support_tickets",
    "techsupport": "tech_support",
    "tech_support": "tech_support",
    "onlinesecurity": "online_security",
    "online_security": "online_security",
    "paperlessbilling": "paperless_billing",
    "paperless_billing": "paperless_billing",
}

