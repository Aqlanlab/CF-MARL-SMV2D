import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from typing import Dict, Optional, Tuple, Any

from ..algorithms.vec2dag import Vec2DAGTransformer
from ..algorithms.sem_estimator import SEMEstimator
from ..algorithms.counterfactual import CounterfactualAnalyzer, CFAccConfig


class CausalEnvCF(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        d: int,
        data_cat: np.ndarray,
        mask_model: Any,
        alpha: float = 0.02,
        beta: float = 10.0,
        tau: float = 0.5,
        z_clip: float = 10.0,
        action_clip: float = 0.5,
        cf_cfg: Optional[CFAccConfig] = None,
        cache_size: int = 256,
        seed: int = 42,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        self._d = d
        self._data_cat = data_cat
        self._mask_model = mask_model
        self._alpha = alpha
        self._beta = beta
        self._tau = tau
        self._z_clip = z_clip
        self._action_clip = action_clip
        self._cf_cfg = cf_cfg or CFAccConfig()
        self._rng = np.random.default_rng(seed)
        self._render_mode = render_mode

        self._obs_dim = d + (d * (d - 1)) // 2
        self.action_space = spaces.Box(
            low=-action_clip, high=action_clip,
            shape=(self._obs_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-z_clip, high=z_clip,
            shape=(self._obs_dim,), dtype=np.float32
        )

        self._z = np.zeros((self._obs_dim,), dtype=np.float32)

        self._vec2dag = Vec2DAGTransformer(d, tau)
        self._sem_estimator = SEMEstimator()
        self._cf_analyzer = CounterfactualAnalyzer(self._cf_cfg)

        self._cache: Dict[Tuple[int, ...], Dict] = {}
        self._cache_order: list = []
        self._cache_size = cache_size

        self._step_count = 0
        self._episode_rewards = []
        self._current_adjacency = None
        self._current_metrics = {}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        init_scale = options.get("init_scale", 0.01) if options else 0.01
        self._z = (init_scale * self._rng.standard_normal(self._obs_dim)).astype(np.float32)
        self._step_count = 0
        self._episode_rewards = []
        self._current_adjacency = None
        self._current_metrics = {}

        return self._z.copy(), {}

    def _cache_get(self, key: Tuple) -> Optional[Dict]:
        if key in self._cache:
            item = self._cache[key]
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return item
        return None

    def _cache_put(self, key: Tuple, value: Dict) -> None:
        if key in self._cache:
            self._cache_order.remove(key)
        elif len(self._cache_order) >= self._cache_size:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)

        self._cache[key] = value
        self._cache_order.append(key)

    def _compute_reward_components(
        self,
        A: np.ndarray
    ) -> Tuple[float, Dict]:

        if self._vec2dag.is_cyclic(A):
            reward = -1e6
            info = {
                "cyclic": True,
                "edges": int(A.sum()),
                "bic": float("inf"),
                "cf_acc": 0.0,
                "rsparse": 0.0
            }
            return reward, info

        key = tuple(A.flatten().tolist())
        cached = self._cache_get(key)

        if cached is None:
            bic, sems, topo = self._sem_estimator.compute_bic_and_sems(A, self._data_cat)
            rsparse = -self._alpha * float(A.sum())

            sems_dict = {}
            for node_id, sem_obj in sems.items():
                sems_dict[node_id] = sem_obj

            cf_acc = self._cf_analyzer.compute_cf_acc(
                A, self._data_cat, sems_dict, topo
            )

            reward = -bic + rsparse + self._beta * cf_acc

            cached = {
                "bic": bic,
                "rsparse": rsparse,
                "cf_acc": cf_acc,
                "reward": reward,
                "sems": sems_dict,
                "topo": topo,
            }
            self._cache_put(key, cached)
        else:
            reward = float(cached["reward"])

        info = {
            "cyclic": False,
            "edges": int(A.sum()),
            "bic": float(cached["bic"]),
            "cf_acc": float(cached["cf_acc"]),
            "rsparse": float(cached["rsparse"]),
        }

        return reward, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, -self._action_clip, self._action_clip).astype(np.float32)
        self._z = np.clip(self._z + action, -self._z_clip, self._z_clip)

        with torch.no_grad():
            M = self._mask_model.mask_rollout().detach().cpu().numpy()

        A_soft = self._vec2dag.vec2dag_soft(self._z, M)
        A = self._vec2dag.binarize_adjacency(A_soft, tau=self._tau)

        reward, info = self._compute_reward_components(A)

        self._step_count += 1
        self._episode_rewards.append(reward)
        self._current_adjacency = A
        self._current_metrics = info

        terminated = False
        truncated = False

        info["step_count"] = self._step_count
        info["cumulative_reward"] = sum(self._episode_rewards)

        return self._z.copy(), float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self._render_mode == "human":
            print(f"Step: {self._step_count}")
            print(f"Current metrics: {self._current_metrics}")
        elif self._render_mode == "rgb_array":
            return self._current_adjacency
        return None

    def get_current_adjacency(self) -> Optional[np.ndarray]:
        return self._current_adjacency

    def get_cache_stats(self) -> Dict:
        return {
            "cache_size": len(self._cache),
            "cache_capacity": self._cache_size,
            "cache_utilization": len(self._cache) / self._cache_size
        }

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def z_state(self) -> np.ndarray:
        return self._z.copy()