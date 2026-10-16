import numpy as np
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class RobustBatchNorm1d(nn.Module):
        """Batch Normalization that safely handles batch size 1 by utilizing running stats in eval mode."""
        def __init__(self, num_features: int):
            super().__init__()
            self.bn = nn.BatchNorm1d(num_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.training and x.size(0) == 1:
                was_training = self.bn.training
                self.bn.eval()
                out = self.bn(x)
                self.bn.train(was_training)
                return out
            return self.bn(x)

    class CoxPartialLikelihoodLoss(nn.Module):
        """Negative Log Cox Partial Likelihood Loss for DeepSurv with numerical stabilization."""
        def __init__(self):
            super(CoxPartialLikelihoodLoss, self).__init__()

        def forward(self, risk_scores: torch.Tensor, T: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
            risk_scores = risk_scores.view(-1)
            # Clamp to prevent float32 exp overflow/underflow
            risk_scores = torch.clamp(risk_scores, -20.0, 20.0)
            
            sorted_indices = torch.argsort(T, descending=True)
            T_sorted = T[sorted_indices]
            risk_scores = risk_scores[sorted_indices]
            E = E[sorted_indices]

            num_events = torch.sum(E.float())
            if num_events == 0:
                return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)

            exp_risk = torch.exp(risk_scores)
            cumsum_exp_risk = torch.cumsum(exp_risk, dim=0)

            # Breslow tied failure time adjustment:
            # All observations tied at time t share the risk set sum sum_{j: T_j >= t} exp(risk_j)
            unique_T, inverse_indices, counts = torch.unique_consecutive(T_sorted, return_inverse=True, return_counts=True)
            last_occurrence_idx = torch.cumsum(counts, dim=0) - 1
            risk_set_sums = cumsum_exp_risk[last_occurrence_idx][inverse_indices]

            log_cumsum_exp_risk = torch.log(risk_set_sums + 1e-8)

            uncensored_likelihood = risk_scores - log_cumsum_exp_risk
            event_loss = -uncensored_likelihood * E.float()
            
            return torch.sum(event_loss) / (num_events + 1e-8)

    class DeepSurvNetwork(nn.Module):
        def __init__(self, in_features: int, hidden_dims: List[int] = [64, 32, 16], dropout: float = 0.2):
            super(DeepSurvNetwork, self).__init__()
            layers = []
            curr_dim = in_features
            for h_dim in hidden_dims:
                layers.append(nn.Linear(curr_dim, h_dim))
                layers.append(RobustBatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                curr_dim = h_dim
            layers.append(nn.Linear(curr_dim, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)


class DeepSurvModel:
    """DeepSurv Wrapper: Training, Evaluation, and Breslow Baseline Hazard Estimation."""
    def __init__(self, hidden_dims: List[int] = [64, 32, 16], lr: float = 1e-3, weight_decay: float = 1e-4, random_state: Optional[int] = None):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.network = None
        self.is_fitted = False
        
        self.baseline_times = None
        self.baseline_survival = None

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, epochs: int = 150, batch_size: int = 256, seed: Optional[int] = None):
        if not HAS_TORCH: raise ImportError('PyTorch is required for DeepSurv.')
        eff_seed = seed if seed is not None else self.random_state
        if eff_seed is not None:
            torch.manual_seed(int(eff_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(eff_seed))

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        T_arr = np.asarray(T, dtype=float)
        E_arr = np.asarray(E, dtype=float)

        in_features = X_arr.shape[1]
        self.network = DeepSurvNetwork(in_features=in_features, hidden_dims=self.hidden_dims)
        optimizer = optim.AdamW(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = CoxPartialLikelihoodLoss()

        X_tensor = torch.tensor(X_arr, dtype=torch.float32)
        T_tensor = torch.tensor(T_arr, dtype=torch.float32)
        E_tensor = torch.tensor(E_arr, dtype=torch.int32)

        self.network.train()
        n_samples = len(X_arr)

        for epoch in range(epochs):
            perm = torch.randperm(n_samples)
            X_shuffled, T_shuffled, E_shuffled = X_tensor[perm], T_tensor[perm], E_tensor[perm]

            for i in range(0, n_samples, batch_size):
                x_b = X_shuffled[i:i+batch_size]
                t_b = T_shuffled[i:i+batch_size]
                e_b = E_shuffled[i:i+batch_size]

                optimizer.zero_grad()
                risk_pred = self.network(x_b)
                loss = criterion(risk_pred, t_b, e_b)
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()

        self.network.eval()
        with torch.no_grad():
            risk_all = torch.exp(torch.clamp(self.network(X_tensor), -20.0, 20.0)).view(-1).detach().cpu().numpy()
            
        self._compute_breslow_baseline(T_arr, E_arr, risk_all)
        self.is_fitted = True
        return self

    def _compute_breslow_baseline(self, T: np.ndarray, E: np.ndarray, risk_scores: np.ndarray):
        unique_times, event_counts = np.unique(T[E == 1], return_counts=True)
        unique_times = np.sort(unique_times)

        cum_hazard = 0.0
        baseline_cum_hazards = []

        for t in unique_times:
            at_risk_mask = (T >= t)
            sum_risk_at_risk = np.sum(risk_scores[at_risk_mask])
            events_at_t = np.sum((T == t) & (E == 1))

            dh0 = events_at_t / (sum_risk_at_risk + 1e-8)
            cum_hazard += dh0
            baseline_cum_hazards.append(cum_hazard)

        if len(unique_times) == 0:
            min_t = float(np.min(T)) if len(T) > 0 else 0.0
            max_t = float(np.max(T)) if len(T) > 0 else 1.0
            if min_t == max_t:
                max_t = min_t + 1.0
            self.baseline_times = np.array([min_t, max_t])
            self.baseline_survival = np.array([1.0, 1.0])
        else:
            self.baseline_times = unique_times
            self.baseline_survival = np.exp(-np.array(baseline_cum_hazards))

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("DeepSurvModel must be fitted first.")
            
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        self.network.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_arr, dtype=torch.float32)
            risk = torch.exp(torch.clamp(self.network(x_t), -20.0, 20.0)).view(-1).detach().cpu().numpy()
        return np.atleast_1d(risk)

    def predict_survival_function(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("DeepSurvModel must be fitted first.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        times_arr = np.asarray(times, dtype=float)
        if len(times_arr) == 0:
            return np.empty((len(X_arr), 0))

        if self.baseline_survival is None or np.all(self.baseline_survival == 1.0):
            return np.ones((len(X_arr), len(times_arr)))

        risk_scores = self.predict_risk(X_arr)
        b_times = self.baseline_times if (self.baseline_times is not None and len(self.baseline_times) > 0) else np.array([times_arr.min(), times_arr.max()])
        b_surv = self.baseline_survival if (self.baseline_survival is not None and len(self.baseline_survival) > 0) else np.array([1.0, 1.0])
        s0_interp = np.interp(times_arr, b_times, b_surv, left=1.0, right=b_surv[-1])
        
        isd = np.power(s0_interp[np.newaxis, :], risk_scores[:, np.newaxis])
        return np.clip(isd, 0.0, 1.0)
