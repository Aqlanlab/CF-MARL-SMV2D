import numpy as np
import networkx as nx
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Tuple, Optional


class StructuralEquationModel:
    def __init__(self, model: Pipeline, parents: np.ndarray):
        self._model = model
        self._parents = parents

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def parents(self) -> np.ndarray:
        return self._parents

    @property
    def model(self) -> Pipeline:
        return self._model


class SEMEstimator:
    def __init__(
        self,
        regularization_strength: float = 1.0,
        max_iter: int = 500,
        random_state: int = 42
    ):
        self._C = regularization_strength
        self._max_iter = max_iter
        self._random_state = random_state
        self._sems: Dict[int, StructuralEquationModel] = {}
        self._cache = {}

    def _create_onehot_encoder(self):
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)

    def _fit_sem_logistic(self, X: np.ndarray, y: np.ndarray) -> Pipeline:
        n_classes = len(np.unique(y))
        if n_classes <= 1:
            raise ValueError("Target has <=1 class")

        if n_classes == 2:
            lr = LogisticRegression(
                max_iter=self._max_iter,
                solver="liblinear",
                C=self._C,
                random_state=self._random_state,
            )
        else:
            lr = LogisticRegression(
                max_iter=self._max_iter,
                solver="lbfgs",
                multi_class="multinomial",
                C=self._C,
                random_state=self._random_state,
            )

        pipe = Pipeline([
            ("oh", self._create_onehot_encoder()),
            ("lr", lr),
        ])
        pipe.fit(X, y)
        return pipe

    def _log_likelihood_from_model(self, model: Pipeline, X: np.ndarray, y: np.ndarray) -> float:
        proba = model.predict_proba(X)
        proba = np.maximum(proba, 1e-12)
        return float(np.sum(np.log(proba[np.arange(len(y)), y])))

    def _param_count_from_model(self, model: Pipeline) -> int:
        lr = model.named_steps["lr"]
        return int(lr.coef_.size + lr.intercept_.size)

    def compute_bic_and_sems(
        self,
        A: np.ndarray,
        data: np.ndarray
    ) -> Tuple[float, Dict[int, StructuralEquationModel], List[int]]:

        cache_key = tuple(A.flatten())
        if cache_key in self._cache:
            return self._cache[cache_key]

        N, d = data.shape
        LL = 0.0
        k_params = 0

        self._sems = {}

        for i in range(d):
            parents = np.where(A[:, i] == 1)[0]
            y = data[:, i]

            n_classes = len(np.unique(y))
            if n_classes <= 1:
                continue

            if len(parents) == 0:
                for c in range(n_classes):
                    p_c = np.mean(y == c)
                    p_c = max(p_c, 1e-12)
                    LL += float(np.sum(y == c) * np.log(p_c))
                k_params += (n_classes - 1)
            else:
                X = data[:, parents]
                model = self._fit_sem_logistic(X, y)
                LL += self._log_likelihood_from_model(model, X, y)
                k_params += self._param_count_from_model(model)
                self._sems[i] = StructuralEquationModel(model, parents)

        bic = -2.0 * LL + k_params * np.log(N)

        G = nx.DiGraph(A)
        topo_order = list(nx.topological_sort(G)) if nx.is_directed_acyclic_graph(G) else list(range(d))

        result = (float(bic), self._sems, topo_order)
        self._cache[cache_key] = result

        return result

    def get_sem_for_node(self, node: int) -> Optional[StructuralEquationModel]:
        return self._sems.get(node)

    def clear_cache(self):
        self._cache.clear()

    @property
    def fitted_sems(self) -> Dict[int, StructuralEquationModel]:
        return self._sems