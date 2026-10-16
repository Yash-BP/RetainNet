import pandas as pd
import numpy as np
import os
from typing import Optional

class TelcoDatasetLoader:
    """Loader for the real or benchmark Telco Customer Churn dataset."""
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath

    def load(self) -> pd.DataFrame:
        if self.filepath is not None:
            if not os.path.isfile(self.filepath):
                raise FileNotFoundError(f"Specified Telco dataset file not found: {self.filepath}")
            return self._parse_csv(self.filepath)
        return self._generate_benchmark_telco(n_samples=2500)

    @classmethod
    def load_telco_data(cls, filepath: Optional[str] = None, n_samples: int = 2500) -> pd.DataFrame:
        """
        Class method for direct benchmark dataset loading in dashboard / CLI.
        """
        if filepath is not None:
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"Specified Telco dataset file not found: {filepath}")
            loader = cls(filepath)
            return loader._parse_csv(filepath)
        return cls._generate_benchmark_telco(n_samples=n_samples)

    @staticmethod
    def _parse_csv(filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
            
        if 'Churn' in df.columns:
            df['event'] = (df['Churn'] == 'Yes').astype(int)
            df = df.drop('Churn', axis=1, errors='ignore')
            
        if 'tenure' in df.columns:
            df = df.rename(columns={'tenure': 'duration'})
            
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1, errors='ignore')
            
        return df

    @staticmethod
    def _generate_benchmark_telco(n_samples: int = 2500, seed: int = 42) -> pd.DataFrame:
        """Generates realistic Telco Customer Churn benchmark dataset."""
        np.random.seed(seed)
        
        monthly_charges = np.clip(np.random.normal(65, 30, n_samples), 18, 120)
        contract_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.55, 0.25, 0.20])
        tech_support = np.random.choice([0, 1], size=n_samples, p=[0.70, 0.30])
        online_security = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])
        paperless_billing = np.random.choice([0, 1], size=n_samples, p=[0.40, 0.60])
        support_tickets = np.random.poisson(1.5, size=n_samples)
        
        # Weibull time-to-event with numerical safety clipping
        k = 1.2
        lmbda = 42
        u = np.clip(np.random.uniform(0, 1, n_samples), 1e-9, 1.0 - 1e-9)
        t_star_base = lmbda * (-np.log(u)) ** (1 / k)
        
        hazard_mod = np.exp((monthly_charges - 65) / 90 + (contract_type == 0) * 0.4 + (tech_support == 0) * 0.3)
        t_star = t_star_base / hazard_mod
        
        c = np.random.uniform(1, 72, n_samples)
        t = np.minimum(t_star, c)
        e = (t_star <= c).astype(int)
        prior_tenure = np.random.uniform(1, 36, n_samples)
        total_charges = monthly_charges * prior_tenure
        
        df = pd.DataFrame({
            'monthly_charges': np.round(monthly_charges, 2),
            'total_charges': np.round(total_charges, 2),
            'support_tickets': support_tickets,
            'contract_type': contract_type,
            'tech_support': tech_support,
            'online_security': online_security,
            'paperless_billing': paperless_billing,
            'duration': np.round(t, 1),
            'event': e
        })
        return df
