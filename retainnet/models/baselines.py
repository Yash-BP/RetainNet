import numpy as np
import pandas as pd

try:
    from lifelines import CoxPHFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

try:
    from sksurv.ensemble import RandomSurvivalForest
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False

class CoxPHBaselineModel:
    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer
        self.model = CoxPHFitter(penalizer=penalizer) if HAS_LIFELINES else None
        self.zero_events = False
    
    def fit(self, X, T, E):
        if np.sum(E) == 0:
            self.zero_events = True
            return self
        if not HAS_LIFELINES:
            raise ImportError("lifelines is not installed")
        self.zero_events = False
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df['T'] = T
        df['E'] = E
        self.model.fit(df, duration_col='T', event_col='E')
        return self
        
    def predict_risk(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.zero_events or self.model is None:
            return np.zeros(len(X))
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        return self.model.predict_partial_hazard(df).values
        
    def predict_survival_function(self, X, times):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.zero_events or self.model is None:
            return np.ones((len(X), len(times)))
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        survival_df = self.model.predict_survival_function(df, times=times)
        return survival_df.T.values

class RandomSurvivalForestBaselineModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomSurvivalForest(n_estimators=n_estimators, random_state=random_state) if HAS_SKSURV else None
        self.rate = 0.0
        self.is_fallback = not HAS_SKSURV

    @property
    def model_display_name(self) -> str:
        return "Random Survival Forest" if not self.is_fallback else "RSF (Fallback)"
        
    def fit(self, X, T, E):
        if not HAS_SKSURV or np.sum(E) == 0:
            if not HAS_SKSURV:
                import warnings
                warnings.warn(
                    "scikit-survival is not installed; RandomSurvivalForestBaselineModel is using constant-hazard fallback mode.",
                    RuntimeWarning,
                    stacklevel=2
                )
            self.is_fallback = True
            self._fit_fallback(X, T, E)
            return self
            
        self.is_fallback = False
        y = np.array([(bool(e), t) for e, t in zip(E, T)], dtype=[('event', bool), ('time', float)])
        self.model.fit(X, y)
        return self
        
    def _fit_fallback(self, X, T, E):
        if np.sum(E) == 0:
            self.rate = 0.0
        else:
            self.rate = np.sum(E) / (np.sum(T) + 1e-8)
        
    def predict_risk(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.is_fallback or not HAS_SKSURV:
            return np.ones(len(X)) * getattr(self, 'rate', 0.05)
        return self.model.predict(X)
        
    def predict_survival_function(self, X, times):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.is_fallback or not HAS_SKSURV:
            rate = getattr(self, 'rate', 0.05)
            surv = np.exp(-rate * np.array(times))
            return np.tile(surv, (len(X), 1))
            
        surv_funcs = self.model.predict_survival_function(X)
        res = []
        for fn in surv_funcs:
            res.append(fn(times))
        return np.array(res)
