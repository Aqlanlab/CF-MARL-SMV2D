import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional


class LearnableDomainMask(nn.Module):
    def __init__(
        self,
        d: int,
        C: np.ndarray,
        E: np.ndarray,
        init_scale: float = 0.01,
        device: Optional[str] = None,
        seed: int = 42
    ):
        super().__init__()
        self._d = d
        self._init_scale = init_scale

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        C_t = torch.from_numpy(C).float().to(self._device)
        E_t = torch.from_numpy(E).float().to(self._device)
        I_t = torch.eye(d, device=self._device)

        g = torch.Generator(device=self._device).manual_seed(seed)
        theta_init = self._init_scale * torch.randn((d, d), generator=g, device=self._device)
        theta_init = theta_init * C_t

        self.ThetaM = nn.Parameter(theta_init)

        self.register_buffer("C", C_t)
        self.register_buffer("E", E_t)
        self.register_buffer("I", I_t)

        self._optimization_steps = 0

    def forward(self) -> torch.Tensor:
        return self.mask_rollout()

    def mask_rollout(self) -> torch.Tensor:
        _eps = 1e-8
        _mask_raw = torch.sigmoid(self.ThetaM + _eps)
        _mask = _mask_raw * self.C * (1.0 - self.I)
        return _mask

    def mask_update_objective(self, lambda_prior: float) -> torch.Tensor:
        _eps = 1e-12
        _mask_raw = torch.sigmoid(self.ThetaM)
        M = _mask_raw * self.C
        allowed = self.C * (1.0 - self.I)

        _sparsity_numerator = (M * allowed).sum()
        _sparsity_denominator = allowed.sum() + _eps
        Lsparsity = _sparsity_numerator / _sparsity_denominator

        _prior_deviation = (1.0 - M) ** 2
        Lprior = lambda_prior * torch.mean(self.E * _prior_deviation)

        _total_loss = Lsparsity + Lprior
        return _total_loss

    @torch.no_grad()
    def zero_forbidden_grads(self) -> None:
        if self.ThetaM.grad is not None:
            self.ThetaM.grad *= self.C

    def get_sparsity_level(self) -> float:
        with torch.no_grad():
            mask = self.mask_rollout()
            return float(mask.mean().item())

    def get_edge_count(self, threshold: float = 0.5) -> int:
        with torch.no_grad():
            mask = self.mask_rollout()
            return int((mask > threshold).sum().item())

    @property
    def optimization_steps(self) -> int:
        return self._optimization_steps

    def increment_optimization_steps(self) -> None:
        self._optimization_steps += 1


class MaskOptimizer:
    def __init__(
        self,
        mask_model: LearnableDomainMask,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0
    ):
        self._mask_model = mask_model
        self._optimizer = optim.Adam(
            mask_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def update(self, lambda_prior: float, num_steps: int = 1) -> float:
        total_loss = 0.0
        for _ in range(num_steps):
            self._optimizer.zero_grad(set_to_none=True)
            loss = self._mask_model.mask_update_objective(lambda_prior)
            loss.backward()
            self._mask_model.zero_forbidden_grads()
            self._optimizer.step()
            self._mask_model.increment_optimization_steps()
            total_loss += loss.item()
        return total_loss / num_steps

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer