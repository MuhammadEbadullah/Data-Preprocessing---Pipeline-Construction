"""Custom scikit-learn transformers that work on pandas DataFrames.

Design rules followed by every transformer in this module
---------------------------------------------------------
1. ``fit`` learns ONLY from the data it receives (so inside cross-validation
   it only ever sees the training fold -> no data leakage).
2. ``transform`` never mutates the input DataFrame (works on a copy).
3. Learned state is stored in attributes ending with ``_`` (sklearn convention).
4. ``get_feature_names_out`` is implemented so the pipeline can return
   a DataFrame with proper column names (``set_output(transform="pandas")``).
5. ``check_is_fitted`` is called in ``transform`` to fail loudly if unfitted.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _as_frame(X) -> pd.DataFrame:
    """Return a *copy* of X as a DataFrame (raises a clear error otherwise)."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(X).__name__}. "
            "These transformers rely on column names."
        )
    return X.copy()


# --------------------------------------------------------------------------- #
# 1. ColumnDropper  (stateless)
# --------------------------------------------------------------------------- #
class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop a fixed list of columns (ignores columns that are not present)."""

    def __init__(self, columns: Sequence[str] = ()):
        self.columns = columns

    def fit(self, X, y=None):
        X = _as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.columns_to_keep_ = [c for c in X.columns if c not in set(self.columns)]
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_to_keep_")
        X = _as_frame(X)
        return X[self.columns_to_keep_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "columns_to_keep_")
        return np.asarray(self.columns_to_keep_, dtype=object)


# --------------------------------------------------------------------------- #
# 2. TitanicFeatureEngineer  (stateless, domain specific)
# --------------------------------------------------------------------------- #
class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create domain features from raw Titanic columns.

    New columns
    -----------
    FamilySize : SibSp + Parch + 1
    IsAlone    : 1 if FamilySize == 1 else 0
    Title      : extracted from ``Name`` (Mr, Mrs, Miss, Master, Rare)
    Deck       : first letter of ``Cabin`` (NaN if cabin is unknown)
    HasCabin   : 1 if Cabin is known else 0

    This transformer is a *template* for domain-specific feature engineering:
    keep it stateless (or learn everything inside ``fit``) and it is safe to
    use inside a Pipeline.
    """

    _TITLE_MAP = {
        "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    }

    def fit(self, X, y=None):
        X = _as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = list(self._engineer(X).columns)
        return self

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

        title = X["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
        X["Title"] = title.map(self._TITLE_MAP).fillna("Rare")

        cabin = X["Cabin"].astype("object")
        X["Deck"] = cabin.str[0]
        X["HasCabin"] = cabin.notna().astype(int)
        return X

    def transform(self, X):
        check_is_fitted(self, "feature_names_out_")
        return self._engineer(_as_frame(X))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)


# --------------------------------------------------------------------------- #
# 3. OutlierClipper  (stateful: learns IQR bounds)
# --------------------------------------------------------------------------- #
class OutlierClipper(BaseEstimator, TransformerMixin):
    """Winsorise numeric columns using the IQR rule.

    Bounds are ``[Q1 - factor*IQR, Q3 + factor*IQR]`` and are learned ONLY
    from the data passed to ``fit`` (training data). NaNs are left untouched,
    so this can safely run *before* imputation.

    Parameters
    ----------
    columns : list[str] | None
        Columns to clip. ``None`` -> all numeric columns seen in ``fit``.
    factor : float
        IQR multiplier (1.5 = classic Tukey fences, 3.0 = only extreme outliers).
    """

    def __init__(self, columns: Iterable[str] | None = None, factor: float = 1.5):
        self.columns = columns
        self.factor = factor

    def fit(self, X, y=None):
        X = _as_frame(X)
        if self.factor < 0:
            raise ValueError("factor must be >= 0")
        cols = list(self.columns) if self.columns is not None else list(
            X.select_dtypes(include="number").columns
        )
        q1 = X[cols].quantile(0.25)
        q3 = X[cols].quantile(0.75)
        iqr = q3 - q1
        self.columns_ = cols
        self.lower_bounds_ = (q1 - self.factor * iqr).to_dict()
        self.upper_bounds_ = (q3 + self.factor * iqr).to_dict()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        X = _as_frame(X)
        for c in self.columns_:
            X[c] = X[c].clip(self.lower_bounds_[c], self.upper_bounds_[c])
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_


# --------------------------------------------------------------------------- #
# 4. RareCategoryGrouper  (stateful: learns frequent categories)
# --------------------------------------------------------------------------- #
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Group infrequent categories into a single ``other_label`` bucket.

    Categories whose relative frequency in the *training* data is below
    ``min_freq`` are replaced by ``other_label``. Categories never seen
    during ``fit`` are ALSO mapped to ``other_label`` at transform time,
    which makes the downstream encoder robust to unseen values in production.
    Missing values (NaN) are preserved so a later imputer can handle them.
    """

    def __init__(
        self,
        columns: Iterable[str] | None = None,
        min_freq: float = 0.05,
        other_label: str = "Other",
    ):
        self.columns = columns
        self.min_freq = min_freq
        self.other_label = other_label

    def fit(self, X, y=None):
        X = _as_frame(X)
        if not 0 <= self.min_freq < 1:
            raise ValueError("min_freq must be in [0, 1)")
        cols = list(self.columns) if self.columns is not None else list(
            X.select_dtypes(include=["object", "category", "string"]).columns
        )
        self.columns_ = cols
        self.frequent_categories_ = {}
        for c in cols:
            freq = X[c].astype("object").value_counts(normalize=True, dropna=True)
            self.frequent_categories_[c] = set(freq[freq >= self.min_freq].index)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "frequent_categories_")
        X = _as_frame(X)
        for c in self.columns_:
            col = X[c].astype("object")
            keep = col.isin(self.frequent_categories_[c]) | col.isna()
            X[c] = col.where(keep, self.other_label)
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_
