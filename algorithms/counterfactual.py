import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder


@dataclass
class CFAccConfig:
    max_samples: Optional[int] = None
    max_values_per_feature: Optional[int] = None
    seed: int = 42
    use_cache: bool = True
    cache_size: int = 1024


class CounterfactualAnalyzer:
    def __init__(self, config: CFAccConfig = None):
        self._config = config or CFAccConfig()
        self._rng = np.random.default_rng(self._config.seed)
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_sample_indices(self, N: int) -> np.ndarray:
        idxs = np.arange(N)
        if self._config.max_samples is not None and self._config.max_samples < N:
            idxs = self._rng.choice(idxs, size=self._config.max_samples, replace=False)
        return idxs

    def _get_unique_values(self, data: np.ndarray) -> List[np.ndarray]:
        d = data.shape[1]
        unique_vals = [np.unique(data[:, j]) for j in range(d)]

        if self._config.max_values_per_feature is not None:
            unique_vals = [
                (vals if len(vals) <= self._config.max_values_per_feature
                 else self._rng.choice(vals, size=self._config.max_values_per_feature, replace=False))
                for vals in unique_vals
            ]
        return unique_vals

    def _propagate_intervention(
        self,
        x_cf: np.ndarray,
        intervened_node: int,
        sems: Dict,
        topo_order: List[int]
    ) -> Tuple[bool, np.ndarray]:

        for node in topo_order:
            if node == intervened_node:
                continue

            if node in sems:
                model = sems[node]
                try:
                    if hasattr(model, 'predict'):
                        parents = model.parents if hasattr(model, 'parents') else model[1]
                        model_obj = model.model if hasattr(model, 'model') else model[0]
                        pred = int(model_obj.predict(x_cf[parents].reshape(1, -1))[0])
                    else:
                        model_obj, parents = model
                        pred = int(model_obj.predict(x_cf[parents].reshape(1, -1))[0])
                    x_cf[node] = pred
                except Exception:
                    return False, x_cf

        return True, x_cf

    def compute_cf_acc(
        self,
        A: np.ndarray,
        data: np.ndarray,
        sems: Dict,
        topo_order: List[int]
    ) -> float:

        if self._config.use_cache:
            cache_key = (tuple(A.flatten()), data.shape[0], data.shape[1])
            if cache_key in self._cache and len(self._cache) < self._config.cache_size:
                self._cache_hits += 1
                return self._cache[cache_key]
            self._cache_misses += 1

        N, d = data.shape
        target_idx = d - 1

        if target_idx not in sems:
            return 0.0

        unique_vals = self._get_unique_values(data)
        idxs = self._get_sample_indices(N)

        total = 0
        matches = 0

        for n in idxs:
            x = data[n].copy()
            observed_y = int(x[target_idx])

            for i in range(d - 1):
                original_val = x[i]
                for alt in unique_vals[i]:
                    if alt == original_val:
                        continue

                    x_cf = x.copy()
                    x_cf[i] = int(alt)

                    valid, x_cf = self._propagate_intervention(x_cf, i, sems, topo_order)

                    if not valid:
                        continue

                    total += 1
                    if int(x_cf[target_idx]) == observed_y:
                        matches += 1

        result = float(matches / total) if total > 0 else 0.0

        if self._config.use_cache and len(self._cache) < self._config.cache_size:
            cache_key = (tuple(A.flatten()), data.shape[0], data.shape[1])
            self._cache[cache_key] = result

        return result

    def compute_cf_attribution(
        self,
        A: np.ndarray,
        data: np.ndarray,
        encoders: List[LabelEncoder],
        feature_names: List[str],
        sems: Dict,
        topo_order: List[int]
    ) -> np.ndarray:

        N, d = data.shape
        target_idx = d - 1
        n_features = d - 1
        classes = encoders[-1].classes_
        n_classes = len(classes)

        if target_idx not in sems:
            return np.zeros((n_features, n_classes), dtype=np.float32)

        unique_vals = self._get_unique_values(data)
        idxs = self._get_sample_indices(N)

        attribution = np.zeros((n_features, n_classes), dtype=np.float32)

        for c in range(n_classes):
            class_samples = [n for n in idxs if int(data[n, target_idx]) == c]
            if not class_samples:
                continue

            for i in range(n_features):
                total = 0
                stable = 0
                for n in class_samples:
                    x = data[n].copy()
                    original_val = x[i]
                    for alt in unique_vals[i]:
                        if alt == original_val:
                            continue

                        x_cf = x.copy()
                        x_cf[i] = int(alt)

                        valid, x_cf = self._propagate_intervention(x_cf, i, sems, topo_order)

                        if not valid:
                            continue
                        total += 1
                        if int(x_cf[target_idx]) == c:
                            stable += 1

                s = (stable / total) if total > 0 else 0.0
                attribution[i, c] = float(-15.0 + 23.0 * s)

        return np.clip(attribution, -15.0, 8.0)

    def get_cache_stats(self) -> Dict[str, int]:
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses + 1e-12)
        }

    def clear_cache(self):
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0