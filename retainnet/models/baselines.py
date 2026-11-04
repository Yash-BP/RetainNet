import warnings
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

try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN_RF = True
except ImportError:
    HAS_SKLEARN_RF = False


class CoxPHBaselineModel:
    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer
        self.model = CoxPHFitter(penalizer=penalizer) if HAS_LIFELINES else None
        self.zero_events = False

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray):
        if np.sum(E) == 0:
            self.zero_events = True
            return self
        if not HAS_LIFELINES:
            raise ImportError("lifelines is not installed")
        self.zero_events = False
        X_arr = np.asarray(X, dtype=float)
        df = pd.DataFrame(X_arr, columns=[f"feature_{i}" for i in range(X_arr.shape[1])])
        df["T"] = np.asarray(T, dtype=float)
        df["E"] = np.asarray(E, dtype=int)
        try:
            self.model.fit(df, duration_col="T", event_col="E")
        except Exception:
            try:
                # Retry with stronger ridge regularization to handle ill-conditioned covariance
                robust_penalizer = max(float(self.penalizer) * 10.0, 0.5)
                fallback_model = CoxPHFitter(penalizer=robust_penalizer)
                fallback_model.fit(df, duration_col="T", event_col="E")
                self.model = fallback_model
            except Exception:
                self.zero_events = True
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.zero_events or self.model is None:
            return np.zeros(len(X))
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        return self.model.predict_partial_hazard(df).values

    def predict_survival_function(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.zero_events or self.model is None:
            return np.ones((len(X), len(times)))
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        survival_df = self.model.predict_survival_function(df, times=times)
        return survival_df.T.values


class RandomSurvivalForestBaselineModel:
    """
    Random Survival Forest Baseline Model.
    Utilizes scikit-survival when available; provides a feature-dependent ensemble fallback
    when scikit-survival is not installed, maintaining transparent disclosure.
    """
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomSurvivalForest(n_estimators=n_estimators, random_state=random_state) if HAS_SKSURV else None
        self.is_fallback = not HAS_SKSURV
        self.fallback_rf = None
        self.base_rate = 0.05
        self.zero_events = False

    @property
    def model_display_name(self) -> str:
        return "Random Survival Forest" if not self.is_fallback else "RSF (Fallback)"

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        T_arr = np.asarray(T, dtype=float)
        E_arr = np.asarray(E, dtype=float)

        if np.sum(E_arr) == 0:
            self.zero_events = True
            self.base_rate = 0.0
            return self

        self.zero_events = False

        if not HAS_SKSURV:
            warnings.warn(
                "scikit-survival is not installed; RandomSurvivalForestBaselineModel is using fallback mode.",
                RuntimeWarning,
                stacklevel=2
            )
            self.is_fallback = True
            self._fit_fallback(X_arr, T_arr, E_arr)
            return self

        self.is_fallback = False
        y = np.array([(bool(e), t) for e, t in zip(E_arr, T_arr)], dtype=[("event", bool), ("time", float)])
        self.model.fit(X_arr, y)
        return self

    def _fit_fallback(self, X: np.ndarray, T: np.ndarray, E: np.ndarray):
        # Base empirical hazard rate
        total_events = np.sum(E)
        total_time = np.sum(T)
        self.base_rate = float(total_events / (total_time + 1e-8)) if total_time > 0 else 0.05

        if HAS_SKLEARN_RF and len(X) >= 4 and total_events > 0:
            try:
                # Train a tree ensemble on pseudo-hazard / log-event time with censoring weight
                # Target: log-hazard proxy -log(T + 1e-4) * (2*E - 1)
                y_proxy = np.where(E == 1, 1.0 / np.maximum(T, 1.0), 0.1 / np.maximum(T, 1.0))
                rf = RandomForestRegressor(
                    n_estimators=min(self.n_estimators, 50),
                    max_depth=5,
                    random_state=self.random_state
                )
                rf.fit(X, y_proxy)
                self.fallback_rf = rf
            except Exception:
                self.fallback_rf = None
        else:
            self.fallback_rf = None

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.zero_events:
            return np.zeros(len(X))

        if self.is_fallback or not HAS_SKSURV:
            if self.fallback_rf is not None:
                preds = self.fallback_rf.predict(X)
                return np.atleast_1d(preds)
            return np.ones(len(X)) * self.base_rate

        return self.model.predict(X)

    def predict_survival_function(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        times_arr = np.asarray(times, dtype=float)

        if len(X) == 0 or len(times_arr) == 0:
            return np.empty((len(X), len(times_arr)), dtype=float)

        if self.zero_events:
            return np.ones((len(X), len(times_arr)))

        if self.is_fallback or not HAS_SKSURV:
            if self.fallback_rf is not None:
                # Individualized hazard multiplier: scale around base rate
                pred_hazards = np.clip(self.fallback_rf.predict(X), 1e-5, 2.0)
                # Compute S(t|x) = exp(-hazard(x) * t)
                surv_matrix = np.exp(-pred_hazards[:, np.newaxis] * times_arr[np.newaxis, :])
                return np.clip(surv_matrix, 0.0, 1.0)
            else:
                surv = np.exp(-self.base_rate * times_arr)
                return np.tile(surv, (len(X), 1))

        surv_funcs = self.model.predict_survival_function(X)
        res = []
        for fn in surv_funcs:
            res.append(fn(times_arr))
        return np.array(res)
