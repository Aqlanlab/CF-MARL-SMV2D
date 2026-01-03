import os
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    def __init__(self, seed: int = 42):
        self._seed = seed
        np.random.seed(seed)
        self._encoders: List[LabelEncoder] = []
        self._feature_names: List[str] = []
        self._metadata: Dict = {}

    def _normalize_defect_code_to_int(self, x) -> Optional[int]:
        if pd.isna(x):
            return None
        s = str(x)
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else None

    def _map_root_cause_from_defect_code(self, defect_code_int: Optional[int]) -> str:
        _mapping_ranges = [
            (1, 26, "Internal Failure"),
            (27, 39, "Damage"),
            (40, 77, "Poor Connections"),
            (78, 102, "Other")
        ]

        if defect_code_int is None:
            return "Unknown"

        for start, end, cause in _mapping_ranges:
            if start <= defect_code_int <= end:
                return cause
        return "Unknown"

    def _remove_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        hashes = {}
        keep_cols = []
        for col in df.columns:
            col_hash = pd.util.hash_pandas_object(df[col], index=False).sum()
            if col_hash not in hashes:
                hashes[col_hash] = col
                keep_cols.append(col)
            else:
                if not df[col].equals(df[hashes[col_hash]]):
                    keep_cols.append(col)
        return df[keep_cols]

    def _mode_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            if out[col].isna().any():
                mode_val = out[col].mode(dropna=True)
                fill_val = mode_val.iloc[0] if len(mode_val) else "Unknown"
                out[col] = out[col].fillna(fill_val)
        return out

    def load_and_preprocess(
        self,
        file_path: str,
        target_col: str = "Root Cause",
        defect_code_col: str = "Defect Code",
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[LabelEncoder], List[str], Dict]:

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_excel(file_path)

        if target_col not in df.columns:
            df[target_col] = np.nan

        if defect_code_col in df.columns:
            dc_int = df[defect_code_col].apply(self._normalize_defect_code_to_int)
            mapped = dc_int.apply(self._map_root_cause_from_defect_code)

            if df[target_col].isna().any():
                df[target_col] = df[target_col].fillna(mapped)
            else:
                df[target_col] = df[target_col].replace("", np.nan).fillna(mapped)

        if feature_cols is None:
            drop_like = {"id", "index", "timestamp", "time", "date"}
            feature_cols = [c for c in df.columns if c != target_col and c.lower() not in drop_like]

        missing = [c for c in feature_cols + [target_col] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df[feature_cols + [target_col]].copy()

        df = self._remove_duplicate_columns(df)
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1 and c != target_col]
        if constant_cols:
            df = df.drop(columns=constant_cols)
            feature_cols = [c for c in feature_cols if c in df.columns and c != target_col]

        df = self._mode_impute(df)

        self._encoders: List[LabelEncoder] = []
        data_enc = np.zeros(df.shape, dtype=np.int64)
        for j, col in enumerate(df.columns):
            le = LabelEncoder()
            data_enc[:, j] = le.fit_transform(df[col].astype(str))
            self._encoders.append(le)

        self._feature_names = [c for c in df.columns if c != target_col]
        d = data_enc.shape[1]
        N = data_enc.shape[0]
        target_classes = self._encoders[-1].classes_.tolist()

        self._metadata = {
            "N": N,
            "d": d,
            "n_features": d - 1,
            "feature_names": self._feature_names,
            "target_col": target_col,
            "target_classes": target_classes,
            "columns": df.columns.tolist(),
        }

        return data_enc, self._encoders, self._feature_names, self._metadata

    @property
    def encoders(self) -> List[LabelEncoder]:
        return self._encoders

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def metadata(self) -> Dict:
        return self._metadata