import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from src.preprocessing import (
    ColumnDropper,
    OutlierClipper,
    RareCategoryGrouper,
    TitanicFeatureEngineer,
    build_full_pipeline,
    build_preprocessor,
)


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv("data/titanic.csv")


# ---------- OutlierClipper ----------
def test_clipper_learns_bounds_from_train_only():
    train = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    test = pd.DataFrame({"a": [-1000, 5, 1000]})
    c = OutlierClipper(factor=1.5).fit(train)
    out = c.transform(test)
    assert out["a"].iloc[0] == c.lower_bounds_["a"]
    assert out["a"].iloc[2] == c.upper_bounds_["a"]
    assert out["a"].iloc[1] == 5


def test_clipper_keeps_nan_and_does_not_mutate_input():
    df = pd.DataFrame({"a": [1.0, np.nan, 100.0, 2.0, 3.0]})
    before = df.copy()
    out = OutlierClipper().fit(df).transform(df)
    assert out["a"].isna().sum() == 1
    pd.testing.assert_frame_equal(df, before)


def test_clipper_requires_fit():
    with pytest.raises(Exception):
        OutlierClipper().transform(pd.DataFrame({"a": [1]}))


# ---------- RareCategoryGrouper ----------
def test_rare_grouper_groups_rare_and_unseen():
    train = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 45 + ["z"] * 5})
    g = RareCategoryGrouper(min_freq=0.1).fit(train)
    out = g.transform(pd.DataFrame({"c": ["a", "z", "never_seen", np.nan]}))
    assert out["c"].tolist()[:3] == ["a", "Other", "Other"]
    assert pd.isna(out["c"].iloc[3])  # NaN preserved for the imputer


def test_rare_grouper_invalid_min_freq():
    with pytest.raises(ValueError):
        RareCategoryGrouper(min_freq=1.5).fit(pd.DataFrame({"c": ["a"]}))


# ---------- ColumnDropper / FeatureEngineer ----------
def test_column_dropper_ignores_missing_columns():
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = ColumnDropper(columns=["b", "not_there"]).fit_transform(df)
    assert list(out.columns) == ["a"]


def test_feature_engineer(raw):
    out = TitanicFeatureEngineer().fit_transform(raw)
    assert {"FamilySize", "IsAlone", "Title", "Deck", "HasCabin"} <= set(out.columns)
    assert (out["FamilySize"] == raw["SibSp"] + raw["Parch"] + 1).all()
    assert set(out["Title"].unique()) <= {"Mr", "Mrs", "Miss", "Master", "Rare"}


def test_non_dataframe_input_raises_clear_error():
    with pytest.raises(TypeError):
        OutlierClipper().fit(np.array([[1, 2], [3, 4]]))


# ---------- Whole pipeline ----------
def test_preprocessor_output_clean(raw):
    X = raw.drop(columns="Survived")
    out = build_preprocessor().fit_transform(X)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == len(X)
    assert out.isna().sum().sum() == 0
    assert all(np.issubdtype(t, np.number) for t in out.dtypes)


def test_preprocessor_handles_unseen_category_and_all_missing_row(raw):
    X = raw.drop(columns="Survived")
    p = build_preprocessor().fit(X)
    new = X.head(2).copy()
    new.loc[new.index[0], "Embarked"] = "ZZZ"          # unseen category
    new.loc[new.index[1], ["Age", "Cabin", "Fare"]] = np.nan  # missing values
    out = p.transform(new)
    assert out.isna().sum().sum() == 0
    assert out.shape[1] == p.transform(X).shape[1]  # same schema as training


def test_output_is_deterministic(raw):
    X = raw.drop(columns="Survived")
    a = build_preprocessor().fit_transform(X)
    b = build_preprocessor().fit_transform(X)
    pd.testing.assert_frame_equal(a, b)


def test_pipeline_is_clonable_and_cv_compatible(raw):
    X, y = raw.drop(columns="Survived"), raw["Survived"]
    pipe = build_full_pipeline(LogisticRegression(max_iter=1000))
    clone(pipe)
    scores = cross_val_score(pipe, X, y, cv=3)
    assert scores.mean() > 0.75
