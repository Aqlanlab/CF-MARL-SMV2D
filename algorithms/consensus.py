import torch
import numpy as np
from typing import List, Any, Dict, Optional


class ConsensusModule:
    def __init__(self, zeta: float = 0.3):
        self._zeta = zeta
        self._consensus_history = []
        self._consensus_count = 0

    @torch.no_grad()
    def update(
        self,
        envs: List[Any],
        agents: List[Any],
        zeta: Optional[float] = None
    ) -> Dict[str, float]:

        if zeta is None:
            zeta = self._zeta

        K = len(envs)
        if K <= 1:
            return {"z_variance": 0.0, "param_variance": 0.0}

        z_states = self._aggregate_z_states(envs, zeta)
        param_variance = self._aggregate_policy_parameters(agents, zeta)

        self._consensus_count += 1

        metrics = {
            "z_variance": z_states,
            "param_variance": param_variance,
            "consensus_count": self._consensus_count
        }
        self._consensus_history.append(metrics)

        return metrics

    def _aggregate_z_states(self, envs: List[Any], zeta: float) -> float:
        z_list = []
        for env in envs:
            if hasattr(env, 'z_state'):
                z_list.append(env.z_state)
            elif hasattr(env, '_z'):
                z_list.append(env._z)
            else:
                z_list.append(env.z)

        z_stack = np.stack(z_list, axis=0)
        z_bar = z_stack.mean(axis=0)
        variance_before = np.var(z_stack)

        for i, env in enumerate(envs):
            new_z = (1.0 - zeta) * z_list[i] + zeta * z_bar
            if hasattr(env, 'z_state'):
                env._z = new_z
            elif hasattr(env, '_z'):
                env._z = new_z
            else:
                env.z = new_z

        z_stack_after = np.stack([
            env.z_state if hasattr(env, 'z_state') else
            (env._z if hasattr(env, '_z') else env.z)
            for env in envs
        ], axis=0)
        variance_after = np.var(z_stack_after)

        return float(variance_after)

    def _aggregate_policy_parameters(self, agents: List[Any], zeta: float) -> float:
        state_dicts = [a.policy.state_dict() for a in agents]
        avg_state = {}
        variances = []

        for key in state_dicts[0].keys():
            tensors = [sd[key].detach().clone().float() for sd in state_dicts]
            stacked = torch.stack(tensors, dim=0)
            avg_state[key] = stacked.mean(dim=0)
            variances.append(float(stacked.var().item()))

        for agent in agents:
            sd = agent.policy.state_dict()
            for key, value in sd.items():
                blended = (1.0 - zeta) * value.float() + zeta * avg_state[key]
                sd[key] = blended.to(value.dtype)
            agent.policy.load_state_dict(sd)

        return float(np.mean(variances))

    def majority_vote_adjacency(
        self,
        adjacency_matrices: List[np.ndarray],
        threshold: float = 0.5
    ) -> np.ndarray:

        if not adjacency_matrices:
            raise ValueError("No adjacency matrices provided")

        if len(adjacency_matrices) == 1:
            return adjacency_matrices[0]

        stacked = np.stack(adjacency_matrices, axis=0)
        mean_adjacency = np.mean(stacked, axis=0)
        consensus = (mean_adjacency > threshold).astype(np.int8)
        np.fill_diagonal(consensus, 0)

        return consensus

    def weighted_vote_adjacency(
        self,
        adjacency_matrices: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:

        if not adjacency_matrices:
            raise ValueError("No adjacency matrices provided")

        if weights is None:
            weights = [1.0 / len(adjacency_matrices)] * len(adjacency_matrices)
        else:
            weights = np.array(weights) / np.sum(weights)

        weighted_sum = sum(
            w * A for w, A in zip(weights, adjacency_matrices)
        )
        consensus = (weighted_sum > 0.5).astype(np.int8)
        np.fill_diagonal(consensus, 0)

        return consensus

    def get_consensus_history(self) -> List[Dict]:
        return self._consensus_history

    @property
    def consensus_count(self) -> int:
        return self._consensus_count