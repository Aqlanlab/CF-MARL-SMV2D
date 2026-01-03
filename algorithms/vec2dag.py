import numpy as np
import networkx as nx
from typing import Tuple, Optional


class Vec2DAGTransformer:
    def __init__(self, d: int, tau: float = 0.5):
        self._d = d
        self._tau = tau
        self._cache = {}

    def _extract_components(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = z[:self._d]
        E_vec = z[self._d:]
        return p, E_vec

    def _build_edge_matrix(self, E_vec: np.ndarray) -> np.ndarray:
        E_mat = np.zeros((self._d, self._d), dtype=np.float32)
        iu = np.triu_indices(self._d, 1)
        E_mat[iu] = E_vec.astype(np.float32)
        E_full = E_mat + E_mat.T
        return E_full

    def _compute_position_difference(self, p: np.ndarray) -> np.ndarray:
        p_diff = p[:, None] - p[None, :]
        return p_diff

    def _apply_sigmoid(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped) + eps)

    def vec2dag_soft(self, z: np.ndarray, M: np.ndarray) -> np.ndarray:
        cache_key = tuple(z.round(4)) + tuple(M.flatten().round(4))
        if cache_key in self._cache and len(self._cache) < 1000:
            return self._cache[cache_key]

        p, E_vec = self._extract_components(z)
        E_full = self._build_edge_matrix(E_vec)
        p_diff = self._compute_position_difference(p)

        sigma_E = self._apply_sigmoid(E_full)
        sigma_P = self._apply_sigmoid(p_diff)

        A_soft = sigma_E * sigma_P * M
        np.fill_diagonal(A_soft, 0.0)

        if len(self._cache) < 1000:
            self._cache[cache_key] = A_soft

        return A_soft

    def binarize_adjacency(self, A_soft: np.ndarray, tau: Optional[float] = None) -> np.ndarray:
        if tau is None:
            tau = self._tau

        A = (A_soft > tau).astype(np.int8)
        np.fill_diagonal(A, 0)
        return A

    def is_cyclic(self, A: np.ndarray) -> bool:
        return not nx.is_directed_acyclic_graph(nx.DiGraph(A))

    def get_topological_order(self, A: np.ndarray) -> Optional[list]:
        G = nx.DiGraph(A)
        if nx.is_directed_acyclic_graph(G):
            return list(nx.topological_sort(G))
        return None

    def remove_cycles(self, A: np.ndarray, A_weights: Optional[np.ndarray] = None) -> np.ndarray:
        A_clean = A.copy()
        G = nx.DiGraph(A_clean)

        if A_weights is None:
            A_weights = A.astype(float)

        max_iterations = 100
        iteration = 0

        while not nx.is_directed_acyclic_graph(G) and iteration < max_iterations:
            try:
                cycle = nx.find_cycle(G, orientation="original")
            except nx.NetworkXNoCycle:
                break

            min_weight = float('inf')
            min_edge = None
            for u, v, _ in cycle:
                weight = A_weights[u, v]
                if weight < min_weight:
                    min_weight = weight
                    min_edge = (u, v)

            if min_edge is None:
                break

            u, v = min_edge
            A_clean[u, v] = 0
            G.remove_edge(u, v)
            iteration += 1

        return A_clean

    def clear_cache(self):
        self._cache.clear()