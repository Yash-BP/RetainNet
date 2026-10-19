import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


class DataPreprocessingPipeline:
    """Pipeline for preprocessing synthetic, Telco, and adaptively mapped customer datasets."""
    def __init__(self, drop_total_charges: bool = False):
        self.drop_total_charges = drop_total_charges
        self.scaler = MinMaxScaler()
        self.feature_names = []
        self.categorical_columns = []
        self.numeric_columns = []
        self.numeric_medians = {}
        self.categorical_modes = {}
        self.fitted = False

    @staticmethod
    def _clean_currency_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ('monthly_charges', 'MonthlyCharges', 'total_charges', 'TotalCharges', 'duration', 'tenure'):
            if col in df.columns and df[col].dtype == 'object':
                cleaned = (
                    df[col].astype(str)
                    .str.strip()
                    .str.replace(r"^[$€£₹\s]+", "", regex=True)
                    .str.replace(",", "", regex=False)
                )
                num_s = pd.to_numeric(cleaned, errors='coerce')
                if num_s.notna().any():
                    df[col] = num_s
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._clean_currency_columns(df)
        col_names = {col.lower(): col for col in df.columns}
        
        total_charges_col = col_names.get('total_charges', col_names.get('totalcharges'))
        tenure_col = col_names.get('tenure', col_names.get('duration'))
        monthly_charges_col = col_names.get('monthly_charges', col_names.get('monthlycharges'))
        num_services_col = col_names.get('num_services')
        contract_col = col_names.get('contract_type', col_names.get('contract'))

        if total_charges_col and tenure_col and not self.drop_total_charges:
            if pd.api.types.is_numeric_dtype(df[total_charges_col]) and pd.api.types.is_numeric_dtype(df[tenure_col]):
                df['AvgMonthlyCharge'] = df[total_charges_col] / np.maximum(df[tenure_col], 1)
        
        if monthly_charges_col and num_services_col:
            if pd.api.types.is_numeric_dtype(df[monthly_charges_col]) and pd.api.types.is_numeric_dtype(df[num_services_col]):
                df['ChargesPerService'] = df[monthly_charges_col] / np.maximum(df[num_services_col], 1)
            
        if contract_col:
            if pd.api.types.is_numeric_dtype(df[contract_col]):
                df['contract_risk'] = (df[contract_col] == 0).astype(int)
            else:
                df['contract_risk'] = (df[contract_col].astype(str).str.lower().str.contains('month-to-month')).astype(int)
                
        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        df = self._engineer_features(df)
        t = df['duration'].values if 'duration' in df.columns else (df['tenure'].values if 'tenure' in df.columns else None)
        e = df['event'].values if 'event' in df.columns else None
        
        drop_cols = ['duration', 'event', 'customer_id', 'customerid', 'tenure', 'contract_label', 'date', 'Date']
        if self.drop_total_charges:
            drop_cols.extend(['total_charges', 'totalcharges', 'TotalCharges', 'AvgMonthlyCharge'])
        X_raw = df.drop([c for c in drop_cols if c in df.columns], axis=1).copy()
        
        self.categorical_columns = [col for col in X_raw.columns if X_raw[col].dtype == 'object' or X_raw[col].dtype.name == 'category']
        self.numeric_columns = [col for col in X_raw.columns if col not in self.categorical_columns]
        
        # Robust imputation for missing values
        self.numeric_medians = {}
        for col in self.numeric_columns:
            median_val = X_raw[col].median()
            val = float(median_val) if not pd.isna(median_val) else 0.0
            self.numeric_medians[col] = val
            X_raw[col] = X_raw[col].fillna(val)

        self.categorical_modes = {}
        for col in self.categorical_columns:
            mode_series = X_raw[col].mode()
            mode_val = mode_series.iloc[0] if len(mode_series) > 0 and not pd.isna(mode_series.iloc[0]) else 'Missing'
            self.categorical_modes[col] = mode_val
            X_raw[col] = X_raw[col].fillna(mode_val)

        if self.categorical_columns:
            X_cat = pd.get_dummies(X_raw[self.categorical_columns], drop_first=True, dtype=float)
        else:
            X_cat = pd.DataFrame(index=X_raw.index)
        
        if self.numeric_columns:
            X_num = X_raw[self.numeric_columns]
            X_num_scaled = self.scaler.fit_transform(X_num)
            X_num_scaled = pd.DataFrame(X_num_scaled, columns=self.numeric_columns, index=X_raw.index)
        else:
            X_num_scaled = pd.DataFrame(index=X_raw.index)
        
        X = pd.concat([X_num_scaled, X_cat], axis=1).astype(np.float32)
        self.feature_names = X.columns.tolist()
        self.fitted = True
        
        return X.values, t, e

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Pipeline has not been fitted yet.")
            
        df = self._engineer_features(df)
        t = df['duration'].values if 'duration' in df.columns else (df['tenure'].values if 'tenure' in df.columns else None)
        e = df['event'].values if 'event' in df.columns else None
        
        drop_cols = ['duration', 'event', 'customer_id', 'customerid', 'tenure', 'contract_label', 'date', 'Date']
        if self.drop_total_charges:
            drop_cols.extend(['total_charges', 'totalcharges', 'TotalCharges', 'AvgMonthlyCharge'])
        X_raw = df.drop([c for c in drop_cols if c in df.columns], axis=1, errors='ignore').copy()
        
        # Apply stored imputation values
        for col in self.numeric_columns:
            if col in X_raw.columns:
                X_raw[col] = X_raw[col].fillna(self.numeric_medians.get(col, 0.0))
            else:
                X_raw[col] = self.numeric_medians.get(col, 0.0)

        for col in self.categorical_columns:
            if col in X_raw.columns:
                X_raw[col] = X_raw[col].fillna(self.categorical_modes.get(col, 'Missing'))
            else:
                X_raw[col] = self.categorical_modes.get(col, 'Missing')

        if self.categorical_columns:
            # Use drop_first=False on transform so single-row (N=1) or homogeneous batches
            # do not accidentally drop their only active category; alignment to self.feature_names
            # below automatically drops the training reference category.
            X_cat = pd.get_dummies(X_raw[self.categorical_columns], drop_first=False, dtype=float)
        else:
            X_cat = pd.DataFrame(index=X_raw.index)
        
        if self.numeric_columns:
            X_num = X_raw[self.numeric_columns]
            X_num_scaled = self.scaler.transform(X_num)
            X_num_scaled = pd.DataFrame(X_num_scaled, columns=self.numeric_columns, index=X_raw.index)
        else:
            X_num_scaled = pd.DataFrame(index=X_raw.index)
        
        X = pd.concat([X_num_scaled, X_cat], axis=1)
        
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names].astype(np.float32)
        
        return X.values, t, e