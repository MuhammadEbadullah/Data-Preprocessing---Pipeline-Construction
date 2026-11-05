"""Pipeline factory: raw DataFrame -> model-ready numeric DataFrame.

Architecture
------------
raw df
  |-- TitanicFeatureEngineer      (adds FamilySize, IsAlone, Title, Deck, HasCabin)
  |-- ColumnDropper               (drops IDs / raw text columns)
  `-- ColumnTransformer
        |-- numeric     : OutlierClipper -> Scaler -> SimpleImputer(median, +indicator)
        |-- binary      : passthrough
        `-- categorical : RareCategoryGrouper -> SimpleImputer("missing") -> OneHotEncoder
"""
from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .transformers import (
    ColumnDropper,
    OutlierClipper,
    RareCategoryGrouper,
    TitanicFeatureEngineer,
)

TARGET = "Survived"
RAW_COLUMNS_TO_DROP = ["PassengerId", "Name", "Ticket", "Cabin"]

NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "Pclass"]
BINARY_FEATURES = ["IsAlone", "HasCabin"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "Deck"]


def build_preprocessor(
    scaler: BaseEstimator | None = None,
    clip_factor: float = 1.5,
    rare_min_freq: float = 0.02,
) -> Pipeline:
    """Return the (unfitted) preprocessing pipeline.

    Parameters
    ----------
    scaler : sklearn scaler instance, default ``StandardScaler()``
        Swap for ``RobustScaler()`` / ``MinMaxScaler()`` as needed.
    clip_factor : float
        IQR multiplier used by :class:`OutlierClipper`.
    rare_min_freq : float
        Categories rarer than this (in training data) -> ``"Other"``.
    """
    scaler = scaler if scaler is not None else StandardScaler()

    numeric_pipe = Pipeline(
        steps=[
            ("clipper", OutlierClipper(factor=clip_factor)),
            # Scalers ignore NaN, so scaling BEFORE imputing keeps the 0/1
            # missing-indicator column unscaled.
            ("scaler", scaler),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("rare_grouper", RareCategoryGrouper(min_freq=rare_min_freq)),
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    preprocessor = Pipeline(
        steps=[
            ("feature_engineering", TitanicFeatureEngineer()),
            ("drop_raw", ColumnDropper(columns=RAW_COLUMNS_TO_DROP)),
            ("column_transformer", column_transformer),
        ]
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def build_full_pipeline(model: BaseEstimator, **preprocessor_kwargs) -> Pipeline:
    """Preprocessing + estimator in ONE object (this is what prevents leakage)."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(**preprocessor_kwargs)),
            ("model", model),
        ]
    )
