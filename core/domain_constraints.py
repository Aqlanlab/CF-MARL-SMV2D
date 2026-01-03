import numpy as np
from typing import List, Optional, Tuple, Dict


class DomainConstraintBuilder:
    def __init__(self, feature_names: List[str], d: int):
        self._feature_names = feature_names
        self._d = d
        self._name_to_idx = self._build_name_index()
        self._C = None
        self._E = None

    def _build_name_index(self) -> Dict[str, int]:
        return {n.lower().replace(" ", "_"): i for i, n in enumerate(self._feature_names)}

    def _get_idx(self, term: str) -> Optional[int]:
        term = term.lower().replace(" ", "_")
        for k, v in self._name_to_idx.items():
            if term in k:
                return v
        return None

    def build_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        C = np.ones((self._d, self._d), dtype=np.int8)
        E = np.zeros((self._d, self._d), dtype=np.int8)

        np.fill_diagonal(C, 0)

        root_idx = self._d - 1
        C[root_idx, :] = 0

        self._apply_manufacturing_hierarchy(C, E)
        self._apply_product_hierarchy(C, E)
        self._apply_memory_hierarchy(C, E)

        self._C = C
        self._E = E
        return C, E

    def _apply_manufacturing_hierarchy(self, C: np.ndarray, E: np.ndarray) -> None:
        prod_stage = self._get_idx("production_stage")
        dest_stage = self._get_idx("destination_stage")
        orig_stage = self._get_idx("original_production_stage")
        process = self._get_idx("process")

        if prod_stage is not None and dest_stage is not None:
            C[dest_stage, prod_stage] = 0
            E[prod_stage, dest_stage] = 1

        if orig_stage is not None and prod_stage is not None:
            C[prod_stage, orig_stage] = 0
            E[orig_stage, prod_stage] = 1

        if process is not None and prod_stage is not None:
            C[prod_stage, process] = 0
            E[process, prod_stage] = 1

    def _apply_product_hierarchy(self, C: np.ndarray, E: np.ndarray) -> None:
        product_series = self._get_idx("product_series")
        product_type = self._get_idx("product_type")
        subproduct_type = self._get_idx("subproduct_type")

        if product_series is not None and product_type is not None:
            C[product_type, product_series] = 0
            E[product_series, product_type] = 1

        if product_type is not None and subproduct_type is not None:
            C[subproduct_type, product_type] = 0
            E[product_type, subproduct_type] = 1

    def _apply_memory_hierarchy(self, C: np.ndarray, E: np.ndarray) -> None:
        mem_type = self._get_idx("memory_type")
        mem_subtype = self._get_idx("memory_subtype")

        if mem_type is not None and mem_subtype is not None:
            C[mem_subtype, mem_type] = 0
            E[mem_type, mem_subtype] = 1

    @property
    def forbidden_edges(self) -> np.ndarray:
        if self._C is None:
            self.build_constraints()
        return 1 - self._C

    @property
    def mandatory_edges(self) -> np.ndarray:
        if self._E is None:
            self.build_constraints()
        return self._E