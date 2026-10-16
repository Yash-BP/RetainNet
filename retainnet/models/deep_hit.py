from typing import Optional
import numpy as np

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


class DeepHitModel:
    def __init__(self, num_bins=20, num_durations=None, hidden_dims=[64, 32, 16], lr=1e-3, random_state: Optional[int] = None):
        self.num_bins = num_durations if num_durations is not None else num_bins
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.random_state = random_state
        self.is_fitted = False
        self.network = None
        self.max_time = None
        self.bin_edges = None
        self.bin_centers = None
        
    def _build_network(self, in_features):
        layers = []
        curr_dim = in_features
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(RobustBatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, self.num_bins))
        layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)

    def fit(self, X, T, E, epochs=150, batch_size=256, seed: Optional[int] = None):
        if not HAS_TORCH: raise ImportError('PyTorch is required for DeepHit.')
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

        self.max_time = max(float(T_arr.max()), 1.0) if len(T_arr) > 0 else 1.0
        # Discretize time into equal intervals
        self.bin_edges = np.linspace(0, self.max_time, self.num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
        # Digitize T
        T_binned = np.digitize(T_arr, self.bin_edges) - 1
        T_binned = np.clip(T_binned, 0, self.num_bins - 1)
        
        in_features = X_arr.shape[1]
        self.network = self._build_network(in_features)
        optimizer = optim.AdamW(self.network.parameters(), lr=self.lr)
        
        X_tensor = torch.tensor(X_arr, dtype=torch.float32)
        T_binned_tensor = torch.tensor(T_binned, dtype=torch.long)
        E_tensor = torch.tensor(E_arr, dtype=torch.float32)
        
        self.network.train()
        n_samples = len(X_arr)
        
        for epoch in range(epochs):
            perm = torch.randperm(n_samples)
            X_shuffled = X_tensor[perm]
            T_shuffled = T_binned_tensor[perm]
            E_shuffled = E_tensor[perm]
            
            for i in range(0, n_samples, batch_size):
                x_b = X_shuffled[i:i+batch_size]
                t_b = T_shuffled[i:i+batch_size]
                e_b = E_shuffled[i:i+batch_size]
                
                optimizer.zero_grad()
                pmf = self.network(x_b)
                cdf = torch.cumsum(pmf, dim=1)
                surv_at_start = torch.cat([torch.ones(len(x_b), 1, device=x_b.device), 1.0 - cdf[:, :-1]], dim=1)
                surv_at_start = torch.clamp(surv_at_start, min=1e-8, max=1.0)

                eps = 1e-8
                pmf_at_t = pmf[torch.arange(len(t_b)), t_b]
                surv_at_t = surv_at_start[torch.arange(len(t_b)), t_b]

                # Full discrete-time survival likelihood (Lee et al. 2018):
                # Observed events in bin t: log P(T in bin t)
                # Censored instances at bin t: log P(T >= bin t) = log S(bin t start)
                log_pmf = torch.log(pmf_at_t + eps)
                log_surv = torch.log(surv_at_t + eps)

                sample_loss = -(e_b * log_pmf + (1.0 - e_b) * log_surv)
                loss = torch.mean(sample_loss)

                loss.backward()
                optimizer.step()
                
        self.is_fitted = True
        return self

    def predict_risk(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        self.network.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_arr, dtype=torch.float32)
            pmf = self.network(x_t).detach().cpu().numpy()
            
        expected_time = np.sum(pmf * self.bin_centers, axis=1)
        return -expected_time # lower expected time = higher risk

    def predict_survival_function(self, X, times):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        times_arr = np.asarray(times, dtype=float)
        if len(times_arr) == 0:
            return np.empty((len(X_arr), 0))

        self.network.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_arr, dtype=torch.float32)
            pmf = self.network(x_t).detach().cpu().numpy()
            
        cdf = np.cumsum(pmf, axis=1)
        surv = 1.0 - cdf
        
        isd = np.zeros((len(X_arr), len(times_arr)))
        for i in range(len(X_arr)):
            isd[i] = np.interp(times_arr, self.bin_centers, surv[i], left=1.0, right=surv[i, -1])
            
        return np.clip(isd, 0.0, 1.0)

