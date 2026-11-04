from typing import Optional, List
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

    class DeepHitRankingLoss(nn.Module):
        """
        Cause-specific pairwise ranking loss for DeepHit (Lee et al., 2018).
        Enforces concordance for pairs (i, j) where subject i failed before subject j.
        """
        def __init__(self, sigma: float = 0.1):
            super().__init__()
            self.sigma = sigma

        def forward(self, cdf: torch.Tensor, T_binned: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
            batch_size = cdf.size(0)
            if batch_size < 2:
                return torch.tensor(0.0, device=cdf.device, requires_grad=True)

            t_i = T_binned.view(-1, 1)
            t_j = T_binned.view(1, -1)
            e_i = E.view(-1, 1)

            mask = (t_i < t_j) & (e_i == 1)
            num_pairs = torch.sum(mask.float())
            if num_pairs == 0:
                return torch.tensor(0.0, device=cdf.device, requires_grad=True)

            # F(t_i | x_i)
            bin_idx = T_binned.view(-1, 1)
            f_i_at_ti = cdf.gather(1, bin_idx)  # (N, 1)

            # F(t_i | x_j)
            f_all_at_ti = cdf[:, T_binned].T   # (N, N): row i is evaluated at time T_binned[i] for subject j

            diff = torch.clamp((f_i_at_ti - f_all_at_ti) / self.sigma, -15.0, 15.0)
            pair_losses = torch.exp(-diff)

            loss = torch.sum(pair_losses * mask.float()) / (num_pairs + 1e-8)
            return loss


class DeepHitModel:
    """
    DeepHit: Deep Learning Approach to Survival Analysis (Lee et al., 2018).
    Optimizes joint discrete log-likelihood (L1) and cause-specific pairwise ranking loss (L2).
    """
    def __init__(
        self,
        num_bins: int = 20,
        num_durations: Optional[int] = None,
        hidden_dims: List[int] = [64, 32, 16],
        lr: float = 1e-3,
        alpha: float = 0.5,
        sigma: float = 0.1,
        random_state: Optional[int] = None
    ):
        self.num_bins = num_durations if num_durations is not None else num_bins
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.alpha = alpha
        self.sigma = sigma
        self.random_state = random_state
        self.is_fitted = False
        self.network = None
        self.max_time = None
        self.bin_edges = None
        self.bin_centers = None

    def _build_network(self, in_features: int):
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

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, epochs: int = 150, batch_size: int = 256, seed: Optional[int] = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for DeepHit.")

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
        self.bin_edges = np.linspace(0, self.max_time, self.num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        T_binned = np.digitize(T_arr, self.bin_edges) - 1
        T_binned = np.clip(T_binned, 0, self.num_bins - 1)

        in_features = X_arr.shape[1]
        self.network = self._build_network(in_features)
        optimizer = optim.AdamW(self.network.parameters(), lr=self.lr)
        ranking_criterion = DeepHitRankingLoss(sigma=self.sigma)

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
                x_b = X_shuffled[i:i + batch_size]
                t_b = T_shuffled[i:i + batch_size]
                e_b = E_shuffled[i:i + batch_size]

                optimizer.zero_grad()
                pmf = self.network(x_b)
                cdf = torch.cumsum(pmf, dim=1)

                surv_at_start = torch.cat([torch.ones(len(x_b), 1, device=x_b.device), 1.0 - cdf[:, :-1]], dim=1)
                surv_at_start = torch.clamp(surv_at_start, min=1e-8, max=1.0)

                eps = 1e-8
                pmf_at_t = pmf[torch.arange(len(t_b)), t_b]
                surv_at_t = surv_at_start[torch.arange(len(t_b)), t_b]

                # L1: Discrete survival log-likelihood (Lee et al. 2018)
                log_pmf = torch.log(pmf_at_t + eps)
                log_surv = torch.log(surv_at_t + eps)
                nll_loss = -torch.mean(e_b * log_pmf + (1.0 - e_b) * log_surv)

                # L2: Pairwise concordance ranking loss
                rank_loss = ranking_criterion(cdf, t_b, e_b)

                total_loss = (1.0 - self.alpha) * nll_loss + self.alpha * rank_loss
                if total_loss.requires_grad:
                    total_loss.backward()
                    optimizer.step()

        self.is_fitted = True
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
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
        return -expected_time  # lower expected time = higher risk

    def predict_survival_function(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
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
        # Midpoint survival at bin_centers anchored at S(0) = 1.0
        surv_mid = np.clip(1.0 - (cdf - 0.5 * pmf), 0.0, 1.0)
        knot_times = np.concatenate([[0.0], self.bin_centers])

        isd = np.zeros((len(X_arr), len(times_arr)))
        for i in range(len(X_arr)):
            knot_surv = np.concatenate([[1.0], surv_mid[i]])
            isd[i] = np.interp(times_arr, knot_times, knot_surv, left=1.0, right=surv_mid[i, -1])

        return np.clip(isd, 0.0, 1.0)
