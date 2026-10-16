import numpy as np
import pandas as pd

class SyntheticSurvivalDataGenerator:
    """Generates synthetic survival data for testing RetainNet models."""
    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate(self, n_samples: int = 2000, max_follow_up: int = 72) -> pd.DataFrame:
        np.random.seed(self.seed)

        # Baseline time-to-event (Weibull)
        k = 1.2
        lmbda = 40
        u = np.clip(np.random.uniform(0, 1, n_samples), 1e-9, 1.0 - 1e-9)
        t_star_base = lmbda * (-np.log(u))**(1/k)

        # Generate features
        monthly_charges = np.random.normal(65, 30, n_samples)
        monthly_charges = np.clip(monthly_charges, 18, 120)
        
        contract_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
        num_services = np.random.randint(1, 9, size=n_samples)
        support_calls = np.random.poisson(2, size=n_samples)
        gender = np.random.randint(0, 2, size=n_samples)
        senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
        partner = np.random.randint(0, 2, size=n_samples)
        dependents = np.random.randint(0, 2, size=n_samples)
        paperless_billing = np.random.randint(0, 2, size=n_samples)
        payment_method = np.random.randint(0, 4, size=n_samples)

        # Influence hazard
        hazard_modifier = np.exp((monthly_charges - 65) / 100 + (contract_type == 0) * 0.5)
        t_star = t_star_base / hazard_modifier
        
        # Censoring
        c = np.random.uniform(1, max_follow_up, n_samples)
        t = np.minimum(t_star, c)
        e = (t_star <= c).astype(int)
        
        # Prior account history at baseline (prediction date)
        prior_tenure = np.random.uniform(1, 36, n_samples)
        total_charges = np.round(monthly_charges * prior_tenure, 2)
        
        df = pd.DataFrame({
            'customer_id': [f"CUST-{i+1:04d}" for i in range(n_samples)],
            'monthly_charges': monthly_charges,
            'contract_type': contract_type,
            'num_services': num_services,
            'support_calls': support_calls,
            'total_charges': total_charges,
            'gender': gender,
            'senior_citizen': senior_citizen,
            'partner': partner,
            'dependents': dependents,
            'paperless_billing': paperless_billing,
            'payment_method': payment_method,
            'duration': t,
            'event': e
        })
        
        return df
