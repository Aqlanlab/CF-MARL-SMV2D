from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import yaml
import json


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    target_kl: Optional[float] = None


@dataclass
class MaskConfig:
    learning_rate: float = 5e-4
    lambda_prior: float = 0.15
    update_steps: int = 10
    update_frequency: int = 5000
    weight_decay: float = 0.0
    gradient_clip: Optional[float] = None


@dataclass
class ConsensusConfig:
    zeta: float = 0.3
    interval: int = 5000
    enabled: bool = True
    majority_threshold: float = 0.5


@dataclass
class EnvironmentConfig:
    alpha: float = 0.02
    beta: float = 10.0
    tau: float = 0.5
    z_clip: float = 10.0
    action_clip: float = 0.5
    cache_size: int = 256


@dataclass
class CounterfactualConfig:
    max_samples: Optional[int] = None
    max_values_per_feature: Optional[int] = None
    use_cache: bool = True
    cache_size: int = 1024


@dataclass
class TrainingConfig:
    total_timesteps: int = 500000
    num_agents: int = 3
    eval_interval: int = 25000
    checkpoint_interval: int = 50000
    seed: int = 42
    device: str = "auto"
    verbose: int = 1
    tensorboard_log: Optional[str] = "./experiments/logs"
    checkpoint_dir: str = "./experiments/checkpoints"
    results_dir: str = "./experiments/results"

    ppo: PPOConfig = field(default_factory=PPOConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    counterfactual: CounterfactualConfig = field(default_factory=CounterfactualConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        with open(path, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        ppo_data = data.pop('ppo', {})
        mask_data = data.pop('mask', {})
        consensus_data = data.pop('consensus', {})
        env_data = data.pop('environment', {})
        cf_data = data.pop('counterfactual', {})

        return cls(
            **data,
            ppo=PPOConfig(**ppo_data),
            mask=MaskConfig(**mask_data),
            consensus=ConsensusConfig(**consensus_data),
            environment=EnvironmentConfig(**env_data),
            counterfactual=CounterfactualConfig(**cf_data)
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> None:
        assert self.total_timesteps > 0, "total_timesteps must be positive"
        assert self.num_agents > 0, "num_agents must be positive"
        assert 0.0 <= self.consensus.zeta <= 1.0, "zeta must be in [0, 1]"
        assert self.environment.tau > 0, "tau must be positive"
        assert self.environment.alpha >= 0, "alpha must be non-negative"
        assert self.environment.beta >= 0, "beta must be non-negative"