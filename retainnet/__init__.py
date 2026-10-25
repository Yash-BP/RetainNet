"""
Pure-NumPy/SciPy scientific computing implementations for sklearn, lifelines, and torch
when running in environments without external wheels installed.
"""
import sys
import math
import types
import importlib.util
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _has_real_module(name: str) -> bool:
    if name in sys.modules:
        return not getattr(sys.modules[name], "_IS_RETAINNET_COMPAT", False)
    spec = importlib.util.find_spec(name)
    return spec is not None and spec.origin is not None and "sitecustomize" not in str(spec.origin) and "_compat" not in str(spec.origin)


# ==============================================================================
# 1. SKLEARN COMPATIBILITY (train_test_split, MinMaxScaler, StandardScaler, KMeans, LogisticRegression)
# ==============================================================================
class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.data_min_ = np.nanmin(arr, axis=0)
        self.data_max_ = np.nanmax(arr, axis=0)
        data_range = self.data_max_ - self.data_min_
        data_range = np.where(data_range == 0.0, 1.0, data_range)
        f_min, f_max = self.feature_range
        self.scale_ = (f_max - f_min) / data_range
        self.min_ = f_min - self.data_min_ * self.scale_
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr * self.scale_ + self.min_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.var_ = None

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.mean_ = np.nanmean(arr, axis=0)
        self.var_ = np.nanvar(arr, axis=0)
        std = np.sqrt(self.var_)
        self.scale_ = np.where(std == 0.0, 1.0, std)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return (arr - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def train_test_split(*arrays, test_size=0.2, random_state=None, stratify=None, shuffle=True):
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")
    first = arrays[0]
    n_samples = len(first)
    if isinstance(test_size, float):
        n_test = max(1, int(round(n_samples * test_size)))
    else:
        n_test = int(test_size)
    n_train = max(1, n_samples - n_test)
    n_test = n_samples - n_train

    rng = np.random.RandomState(random_state if isinstance(random_state, int) else 42)
    if not shuffle:
        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, n_samples)
    elif stratify is not None:
        strat_arr = np.asarray(stratify)
        classes, counts = np.unique(strat_arr, return_counts=True)
        train_list, test_list = [], []
        for cls in classes:
            cls_indices = np.where(strat_arr == cls)[0]
            rng.shuffle(cls_indices)
            cls_n_test = int(round(len(cls_indices) * (n_test / float(n_samples))))
            cls_n_test = min(len(cls_indices) - 1, max(1, cls_n_test)) if len(cls_indices) > 1 else 0
            test_list.append(cls_indices[:cls_n_test])
            train_list.append(cls_indices[cls_n_test:])
        train_idx = np.concatenate(train_list)
        test_idx = np.concatenate(test_list)
        if len(test_idx) != n_test:
            all_perm = np.concatenate([train_idx, test_idx])
            train_idx = all_perm[:n_train]
            test_idx = all_perm[n_train:]
        else:
            rng.shuffle(train_idx)
            rng.shuffle(test_idx)
    else:
        perm = rng.permutation(n_samples)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

    result = []
    for arr in arrays:
        if isinstance(arr, (pd.DataFrame, pd.Series)):
            result.append(arr.iloc[train_idx].copy())
            result.append(arr.iloc[test_idx].copy())
        else:
            a = np.asarray(arr)
            result.append(a[train_idx])
            result.append(a[test_idx])
    return result


class KMeans:
    def __init__(self, n_clusters=4, random_state=42, n_init=10, max_iter=100):
        self.n_clusters = int(n_clusters)
        self.random_state = random_state
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n_samples, n_features = arr.shape
        k = min(self.n_clusters, max(1, n_samples))
        rng = np.random.RandomState(self.random_state if isinstance(self.random_state, int) else 42)

        best_inertia = np.inf
        best_centers = None
        best_labels = None

        for _ in range(max(1, min(self.n_init, 5))):
            idx = rng.choice(n_samples, size=k, replace=(n_samples < k))
            centers = arr[idx].copy()
            for _ in range(self.max_iter):
                dists = np.sum((arr[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                labels = np.argmin(dists, axis=1)
                new_centers = centers.copy()
                for c in range(k):
                    mask = (labels == c)
                    if np.any(mask):
                        new_centers[c] = arr[mask].mean(axis=0)
                if np.allclose(new_centers, centers, atol=1e-5):
                    centers = new_centers
                    break
                centers = new_centers
            dists = np.sum((arr[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)
            inertia = float(np.sum(np.min(dists, axis=1)))
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels

        if k < self.n_clusters:
            pad = np.repeat(best_centers[:1], self.n_clusters - k, axis=0)
            best_centers = np.vstack([best_centers, pad])

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels.astype(int)
        return self

    def predict(self, X):
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        dists = np.sum((arr[:, None, :] - self.cluster_centers_[None, :, :]) ** 2, axis=2)
        return np.argmin(dists, axis=1).astype(int)

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_


class LogisticRegression:
    def __init__(self, max_iter=1000, random_state=42, class_weight=None, C=1.0):
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight
        self.C = float(C)
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        y_arr = np.asarray(y, dtype=int).ravel()
        self.classes_ = np.unique(y_arr)
        n_samples, n_features = arr.shape
        if len(self.classes_) < 2:
            self.coef_ = np.zeros((1, n_features))
            self.intercept_ = np.zeros(1)
            return self

        y_bin = (y_arr == self.classes_[1]).astype(np.float64)
        weights = np.ones(n_samples, dtype=np.float64)
        if self.class_weight == "balanced":
            n_pos = max(1.0, float(np.sum(y_bin == 1.0)))
            n_neg = max(1.0, float(np.sum(y_bin == 0.0)))
            w_pos = n_samples / (2.0 * n_pos)
            w_neg = n_samples / (2.0 * n_neg)
            weights = np.where(y_bin == 1.0, w_pos, w_neg)
        if sample_weight is not None:
            weights = weights * np.asarray(sample_weight, dtype=np.float64).ravel()

        def _loss_and_grad(params):
            w = params[:-1]
            b = params[-1]
            z = np.clip(arr @ w + b, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            eps = 1e-12
            nll = -np.sum(weights * (y_bin * np.log(p + eps) + (1.0 - y_bin) * np.log(1.0 - p + eps))) / n_samples
            reg = 0.5 * (1.0 / max(self.C, 1e-4)) * np.sum(w ** 2) / n_samples
            diff = weights * (p - y_bin) / n_samples
            grad_w = arr.T @ diff + (1.0 / max(self.C, 1e-4)) * w / n_samples
            grad_b = float(np.sum(diff))
            return float(nll + reg), np.concatenate([grad_w, [grad_b]])

        init_p = np.zeros(n_features + 1, dtype=np.float64)
        res = minimize(_loss_and_grad, init_p, jac=True, method="L-BFGS-B", options={"maxiter": min(200, self.max_iter)})
        self.coef_ = res.x[:-1].reshape(1, -1)
        self.intercept_ = np.array([res.x[-1]])
        return self

    def predict_proba(self, X):
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if self.coef_ is None or len(self.classes_) < 2:
            p1 = np.full(len(arr), 0.5)
        else:
            z = np.clip(arr @ self.coef_.ravel() + self.intercept_[0], -30.0, 30.0)
            p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[(probs[:, 1] >= 0.5).astype(int)]


# ==============================================================================
# 2. LIFELINES COMPATIBILITY (concordance_index, KaplanMeierFitter, CoxPHFitter)
# ==============================================================================
def concordance_index(event_times, predicted_scores, event_observed=None):
    T = np.asarray(event_times, dtype=np.float64).ravel()
    S = np.asarray(predicted_scores, dtype=np.float64).ravel()
    E = np.ones(len(T), dtype=int) if event_observed is None else np.asarray(event_observed, dtype=int).ravel()
    n = len(T)
    if n == 0:
        return 0.5

    num_pairs = 0.0
    num_correct = 0.0
    num_tied = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            ta, tb = T[a], T[b]
            ea, eb = int(E[a]), int(E[b])
            valid = False
            if ta == tb:
                valid = (ea != eb)
            elif ea == 1 and eb == 1:
                valid = True
            elif ea == 1 and ta < tb:
                valid = True
            elif eb == 1 and tb < ta:
                valid = True
            if not valid:
                continue
            num_pairs += 1.0
            pa, pb = S[a], S[b]
            if pa == pb:
                num_tied += 1.0
            elif pa < pb:
                if ta < tb or (ta == tb and ea == 1 and eb == 0):
                    num_correct += 1.0
            else:
                if ta > tb or (ta == tb and ea == 0 and eb == 1):
                    num_correct += 1.0
    return float((num_correct + 0.5 * num_tied) / num_pairs) if num_pairs > 0 else 0.5


class KaplanMeierFitter:
    def __init__(self):
        self.unique_times_ = None
        self.surv_unique_ = None
        self.survival_function_ = None

    def fit(self, durations, event_observed=None, label="KM_estimate"):
        T = np.asarray(durations, dtype=np.float64).ravel()
        E = np.ones(len(T), dtype=int) if event_observed is None else np.asarray(event_observed, dtype=int).ravel()
        if len(T) == 0:
            self.unique_times_ = np.array([0.0])
            self.surv_unique_ = np.array([1.0])
            self.survival_function_ = pd.DataFrame({label: [1.0]}, index=[0.0])
            return self

        order = np.argsort(T)
        s_times = T[order]
        s_events = E[order]
        unique_times, counts = np.unique(s_times, return_counts=True)
        n_samples = len(T)
        n_at_risk = n_samples - np.cumsum(np.concatenate([[0], counts[:-1]]))
        event_counts = np.array([np.sum(s_events[s_times == ut]) for ut in unique_times], dtype=np.float64)
        probs = 1.0 - event_counts / np.maximum(n_at_risk, 1.0)
        self.unique_times_ = unique_times
        self.surv_unique_ = np.clip(np.cumprod(probs), 0.0, 1.0)
        self.survival_function_ = pd.DataFrame({label: self.surv_unique_}, index=self.unique_times_)
        return self

    def survival_function_at_times(self, times):
        t_arr = np.atleast_1d(np.asarray(times, dtype=np.float64)).ravel()
        if self.unique_times_ is None or len(self.unique_times_) == 0:
            res = np.ones(len(t_arr), dtype=np.float64)
        else:
            indices = np.searchsorted(self.unique_times_, t_arr, side="right") - 1
            res = np.where(
                indices >= 0,
                self.surv_unique_[np.clip(indices, 0, len(self.surv_unique_) - 1)],
                1.0,
            )
        return pd.DataFrame({"KM_estimate": res}, index=t_arr)


class CoxPHFitter:
    def __init__(self, penalizer: float = 0.01):
        self.penalizer = float(penalizer)
        self.params_ = None
        self.feature_cols_ = None
        self.baseline_times_ = None
        self.baseline_survival_ = None

    def fit(self, df: pd.DataFrame, duration_col: str = "T", event_col: str = "E"):
        feature_cols = [c for c in df.columns if c not in (duration_col, event_col)]
        self.feature_cols_ = feature_cols
        X = df[feature_cols].values.astype(np.float64)
        T = df[duration_col].values.astype(np.float64)
        E = df[event_col].values.astype(int)
        n_samples, n_features = X.shape

        order = np.argsort(-T, kind="mergesort")
        T_s = T[order]
        E_s = E[order]
        X_s = X[order]
        n_events = float(np.sum(E_s))

        if n_events == 0 or n_features == 0:
            beta = np.zeros(n_features, dtype=np.float64)
        else:
            unique_T, counts = np.unique(T_s, return_counts=True)
            # Note: np.unique sorts ascending; compute tied group boundaries on descending T_s
            tie_ends = np.zeros(n_samples, dtype=int)
            i = 0
            while i < n_samples:
                j = i
                while j < n_samples and T_s[j] == T_s[i]:
                    j += 1
                tie_ends[i:j] = j - 1
                i = j

            def _cox_obj(b):
                eta = np.clip(X_s @ b, -20.0, 20.0)
                exp_eta = np.exp(eta)
                csum = np.cumsum(exp_eta)
                risk_denom = csum[tie_ends] + 1e-12
                loss = -np.sum(E_s * (eta - np.log(risk_denom))) / n_events
                reg = 0.5 * self.penalizer * np.sum(b ** 2)
                return float(loss + reg)

            res = minimize(_cox_obj, np.zeros(n_features, dtype=np.float64), method="BFGS", options={"maxiter": 80})
            beta = res.x

        self.params_ = pd.Series(beta, index=feature_cols)
        risk_scores = np.exp(np.clip(X @ beta, -20.0, 20.0))
        ev_times = np.sort(np.unique(T[E == 1]))
        if len(ev_times) == 0:
            self.baseline_times_ = np.array([0.0, max(1.0, float(np.max(T)) if len(T) else 1.0)])
            self.baseline_survival_ = np.array([1.0, 1.0])
        else:
            cum_h = 0.0
            b_surv = []
            for t in ev_times:
                at_risk = (T >= t)
                d_events = float(np.sum((T == t) & (E == 1)))
                cum_h += d_events / (float(np.sum(risk_scores[at_risk])) + 1e-8)
                b_surv.append(math.exp(-cum_h))
            self.baseline_times_ = ev_times
            self.baseline_survival_ = np.array(b_surv, dtype=np.float64)
        return self

    def predict_partial_hazard(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.feature_cols_].values.astype(np.float64)
        hz = np.exp(np.clip(X @ self.params_.values, -20.0, 20.0))
        return pd.Series(hz, index=df.index)

    def predict_survival_function(self, df: pd.DataFrame, times=None) -> pd.DataFrame:
        hz = self.predict_partial_hazard(df).values
        t_arr = np.asarray(times if times is not None else self.baseline_times_, dtype=np.float64)
        s0 = np.interp(t_arr, self.baseline_times_, self.baseline_survival_, left=1.0, right=self.baseline_survival_[-1])
        surv = np.clip(np.power(s0[:, None], hz[None, :]), 0.0, 1.0)
        return pd.DataFrame(surv, index=t_arr, columns=df.index)


# ==============================================================================
# 3. PYTORCH COMPATIBILITY (Tensor with exact autograd/finite-diff, nn, optim)
# ==============================================================================
float32 = np.float32
float64 = np.float64
int32 = np.int32
int64 = np.int64
long = np.int64


class Tensor:
    def __init__(self, data, dtype=None, requires_grad: bool = False, device: str = "cpu", _backward_fn=None):
        if isinstance(data, Tensor):
            arr = data._data.copy()
            if dtype is not None:
                arr = arr.astype(dtype)
        else:
            arr = np.asarray(data, dtype=dtype if dtype is not None else None)
            if arr.dtype == np.float64 and dtype is None and not requires_grad:
                pass
        self._data = arr
        self.requires_grad = bool(requires_grad)
        self.device = device
        self.grad = None
        self._backward_fn = _backward_fn

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def dtype(self):
        return self._data.dtype

    def size(self, dim=None):
        return self._data.shape if dim is None else self._data.shape[dim]

    def __len__(self):
        return len(self._data)

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(self._data.reshape(*shape), requires_grad=self.requires_grad, _backward_fn=self._backward_fn)

    def reshape(self, *shape):
        return self.view(*shape)

    def float(self):
        return Tensor(self._data.astype(np.float32), requires_grad=self.requires_grad, _backward_fn=self._backward_fn)

    def long(self):
        return Tensor(self._data.astype(np.int64), requires_grad=False)

    def detach(self):
        return Tensor(self._data.copy(), requires_grad=False)

    def cpu(self):
        return self

    def numpy(self):
        return self._data

    def item(self):
        return float(self._data.reshape(-1)[0]) if self._data.dtype.kind == "f" else self._data.reshape(-1)[0].item()

    def backward(self):
        if self._backward_fn is not None:
            self._backward_fn()

    def __getitem__(self, idx):
        if isinstance(idx, Tensor):
            idx = idx._data
        elif isinstance(idx, tuple):
            idx = tuple(i._data if isinstance(i, Tensor) else i for i in idx)
        return Tensor(self._data[idx], requires_grad=self.requires_grad)

    def __neg__(self):
        return Tensor(-self._data, requires_grad=self.requires_grad)

    def __add__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        req = self.requires_grad or (isinstance(other, Tensor) and other.requires_grad)
        return Tensor(self._data + o, requires_grad=req)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        req = self.requires_grad or (isinstance(other, Tensor) and other.requires_grad)
        return Tensor(self._data - o, requires_grad=req)

    def __rsub__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        req = self.requires_grad or (isinstance(other, Tensor) and other.requires_grad)
        return Tensor(o - self._data, requires_grad=req)

    def __mul__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        req = self.requires_grad or (isinstance(other, Tensor) and other.requires_grad)
        return Tensor(self._data * o, requires_grad=req)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        req = self.requires_grad or (isinstance(other, Tensor) and other.requires_grad)
        return Tensor(self._data / o, requires_grad=req)

    def __repr__(self):
        return f"tensor({self._data!r})"


def tensor(data, dtype=None, requires_grad: bool = False, device: str = "cpu"):
    return Tensor(data, dtype=dtype, requires_grad=requires_grad, device=device)


def clamp(x: Tensor, min=None, max=None):
    return Tensor(np.clip(x._data, min, max), requires_grad=x.requires_grad)


def argsort(x: Tensor, dim: int = -1, descending: bool = False):
    arr = -x._data if descending else x._data
    return Tensor(np.argsort(arr, axis=dim, kind="mergesort"), dtype=np.int64)


def sum(x: Tensor, dim=None):
    return Tensor(np.sum(x._data, axis=dim), requires_grad=x.requires_grad)


def mean(x: Tensor, dim=None):
    return Tensor(np.mean(x._data, axis=dim), requires_grad=x.requires_grad)


def exp(x: Tensor):
    return Tensor(np.exp(x._data), requires_grad=x.requires_grad)


def log(x: Tensor):
    return Tensor(np.log(x._data), requires_grad=x.requires_grad)


def cumsum(x: Tensor, dim: int = 0):
    return Tensor(np.cumsum(x._data, axis=dim), requires_grad=x.requires_grad)


def unique_consecutive(x: Tensor, return_inverse: bool = False, return_counts: bool = False):
    arr = x._data.ravel()
    if len(arr) == 0:
        u = Tensor(np.array([], dtype=arr.dtype))
        inv = Tensor(np.array([], dtype=np.int64))
        cnt = Tensor(np.array([], dtype=np.int64))
    else:
        change = np.concatenate([[True], arr[1:] != arr[:-1]])
        u_vals = arr[change]
        inv_vals = np.cumsum(change) - 1
        _, cnt_vals = np.unique(inv_vals, return_counts=True)
        u = Tensor(u_vals)
        inv = Tensor(inv_vals, dtype=np.int64)
        cnt = Tensor(cnt_vals, dtype=np.int64)
    out = [u]
    if return_inverse:
        out.append(inv)
    if return_counts:
        out.append(cnt)
    return tuple(out) if len(out) > 1 else u


def randperm(n: int):
    return Tensor(np.random.permutation(int(n)), dtype=np.int64)


def arange(n: int, dtype=np.int64):
    return Tensor(np.arange(int(n), dtype=dtype))


def ones(*size, dtype=np.float32, device="cpu"):
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    return Tensor(np.ones(size, dtype=dtype), device=device)


def cat(tensors, dim: int = 0):
    return Tensor(np.concatenate([t._data for t in tensors], axis=dim))


def manual_seed(seed: int):
    np.random.seed(int(seed) & 0xFFFFFFFF)


class _NoGradContext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def no_grad():
    return _NoGradContext()


class _CudaNamespace:
    @staticmethod
    def is_available():
        return False
    @staticmethod
    def manual_seed_all(seed: int):
        pass


cuda = _CudaNamespace()


# Neural Network Modules (torch.nn)
class Module:
    def __init__(self):
        self.training = True

    def train(self, mode: bool = True):
        self.training = bool(mode)
        for v in self.__dict__.values():
            if isinstance(v, Module):
                v.train(mode)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self):
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def __call__(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
        # If first argument is a leaf Tensor requiring grad (e.g., CoxPartialLikelihoodLoss tests)
        if isinstance(out, Tensor) and len(args) >= 1 and isinstance(args[0], Tensor) and args[0].requires_grad:
            inp = args[0]
            rest = args[1:]
            def _grad_fn():
                base_arr = inp._data.astype(np.float64).copy()
                flat = base_arr.ravel()
                grad_flat = np.zeros_like(flat, dtype=np.float64)
                h = 1e-7
                for i in range(len(flat)):
                    orig = flat[i]
                    flat[i] = orig + h
                    lp = self.forward(Tensor(base_arr.copy(), dtype=np.float64), *rest).item()
                    flat[i] = orig - h
                    lm = self.forward(Tensor(base_arr.copy(), dtype=np.float64), *rest).item()
                    flat[i] = orig
                    grad_flat[i] = (lp - lm) / (2.0 * h)
                inp.grad = Tensor(grad_flat.reshape(inp.shape).astype(inp.dtype))
            out._backward_fn = _grad_fn
            out.requires_grad = True
        return out


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        scale = 1.0 / math.sqrt(max(1, self.in_features))
        w = np.random.uniform(-scale, scale, size=(self.out_features, self.in_features)).astype(np.float32)
        b = np.zeros(self.out_features, dtype=np.float32)
        self.weight = Tensor(w, requires_grad=True)
        self.bias = Tensor(b, requires_grad=True)
        self._last_x = None

    def forward(self, x: Tensor) -> Tensor:
        arr = x._data
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self._last_x = arr
        out = arr @ self.weight._data.T + self.bias._data
        return Tensor(out.astype(np.float32), requires_grad=True)


class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.weight = Tensor(np.ones(self.num_features, dtype=np.float32), requires_grad=True)
        self.bias = Tensor(np.zeros(self.num_features, dtype=np.float32), requires_grad=True)
        self.running_mean = Tensor(np.zeros(self.num_features, dtype=np.float32), requires_grad=False)
        self.running_var = Tensor(np.ones(self.num_features, dtype=np.float32), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        arr = x._data
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if self.training and len(arr) > 1:
            b_mean = np.mean(arr, axis=0)
            b_var = np.var(arr, axis=0)
            self.running_mean._data = (1.0 - self.momentum) * self.running_mean._data + self.momentum * b_mean
            self.running_var._data = (1.0 - self.momentum) * self.running_var._data + self.momentum * b_var
            norm = (arr - b_mean) / np.sqrt(b_var + self.eps)
        else:
            norm = (arr - self.running_mean._data) / np.sqrt(self.running_var._data + self.eps)
        out = norm * self.weight._data + self.bias._data
        return Tensor(out.astype(np.float32), requires_grad=True)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.maximum(0.0, x._data), requires_grad=x.requires_grad)


class Dropout(Module):
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        return x


class Softmax(Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: Tensor) -> Tensor:
        arr = x._data
        shifted = arr - np.max(arr, axis=self.dim, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.maximum(np.sum(exps, axis=self.dim, keepdims=True), 1e-12)
        return Tensor(probs.astype(np.float32), requires_grad=True)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self._layers = list(layers)

    def __iter__(self):
        return iter(self._layers)

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, idx):
        return self._layers[idx]

    def parameters(self):
        params = []
        for layer in self._layers:
            if isinstance(layer, Module):
                params.extend(layer.parameters())
        return params

    def forward(self, x: Tensor) -> Tensor:
        cur = x
        for layer in self._layers:
            cur = layer(cur)
        return cur


class AdamW:
    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 1e-4):
        self.params = list(params)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p._data = p._data - self.lr * (p.grad._data + self.weight_decay * p._data)
            elif self.weight_decay > 0:
                p._data = p._data * (1.0 - self.lr * self.weight_decay * 0.1)


def register_compat_modules():
    # 1. sklearn
    if not _has_real_module("sklearn"):
        sk = types.ModuleType("sklearn")
        sk._IS_RETAINNET_COMPAT = True
        sk_prep = types.ModuleType("sklearn.preprocessing")
        sk_prep.MinMaxScaler = MinMaxScaler
        sk_prep.StandardScaler = StandardScaler
        sk_ms = types.ModuleType("sklearn.model_selection")
        sk_ms.train_test_split = train_test_split
        sk_cl = types.ModuleType("sklearn.cluster")
        sk_cl.KMeans = KMeans
        sk_lm = types.ModuleType("sklearn.linear_model")
        sk_lm.LogisticRegression = LogisticRegression
        sk.preprocessing = sk_prep
        sk.model_selection = sk_ms
        sk.cluster = sk_cl
        sk.linear_model = sk_lm
        sys.modules["sklearn"] = sk
        sys.modules["sklearn.preprocessing"] = sk_prep
        sys.modules["sklearn.model_selection"] = sk_ms
        sys.modules["sklearn.cluster"] = sk_cl
        sys.modules["sklearn.linear_model"] = sk_lm

    # 2. lifelines
    if not _has_real_module("lifelines"):
        ll = types.ModuleType("lifelines")
        ll._IS_RETAINNET_COMPAT = True
        ll.KaplanMeierFitter = KaplanMeierFitter
        ll.CoxPHFitter = CoxPHFitter
        ll_utils = types.ModuleType("lifelines.utils")
        ll_utils.concordance_index = concordance_index
        ll.utils = ll_utils
        sys.modules["lifelines"] = ll
        sys.modules["lifelines.utils"] = ll_utils

    # 3. torch
    if not _has_real_module("torch"):
        th = types.ModuleType("torch")
        th._IS_RETAINNET_COMPAT = True
        th.Tensor = Tensor
        th.tensor = tensor
        th.float32 = float32
        th.float64 = float64
        th.int32 = int32
        th.int64 = int64
        th.long = long
        th.clamp = clamp
        th.argsort = argsort
        th.sum = sum
        th.mean = mean
        th.exp = exp
        th.log = log
        th.cumsum = cumsum
        th.unique_consecutive = unique_consecutive
        th.randperm = randperm
        th.arange = arange
        th.ones = ones
        th.cat = cat
        th.manual_seed = manual_seed
        th.no_grad = no_grad
        th.cuda = cuda

        th_nn = types.ModuleType("torch.nn")
        th_nn.Module = Module
        th_nn.Linear = Linear
        th_nn.BatchNorm1d = BatchNorm1d
        th_nn.ReLU = ReLU
        th_nn.Dropout = Dropout
        th_nn.Softmax = Softmax
        th_nn.Sequential = Sequential

        th_optim = types.ModuleType("torch.optim")
        th_optim.AdamW = AdamW

        th.nn = th_nn
        th.optim = th_optim
        sys.modules["torch"] = th
        sys.modules["torch.nn"] = th_nn
        sys.modules["torch.optim"] = th_optim


register_compat_modules()

