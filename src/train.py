"""Train, tune and persist the full (preprocessing + model) pipeline.

Run from the project root:
    python -m src.train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split

from src.preprocessing import build_full_pipeline
from src.preprocessing.pipeline import TARGET

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "titanic.csv"
MODEL_PATH = ROOT / "models" / "titanic_pipeline.joblib"
REPORTS = ROOT / "reports"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop(columns=TARGET), df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1) Baseline model comparison -- preprocessing is re-fit inside every fold
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }
    cv_results = {}
    for name, model in candidates.items():
        scores = cross_val_score(build_full_pipeline(model), X_train, y_train, cv=cv, scoring="roc_auc")
        cv_results[name] = {"cv_auc_mean": round(scores.mean(), 4), "cv_auc_std": round(scores.std(), 4)}
        print(f"{name:20s} CV ROC-AUC = {scores.mean():.4f} +/- {scores.std():.4f}")

    # 2) Tune preprocessing AND model hyper-parameters together
    pipe = build_full_pipeline(RandomForestClassifier(random_state=RANDOM_STATE))
    param_grid = {
        "preprocessor__column_transformer__num__clipper__factor": [1.5, 3.0],
        "preprocessor__column_transformer__cat__rare_grouper__min_freq": [0.02, 0.05],
        "model__n_estimators": [200, 400],
        "model__max_depth": [4, 6, None],
    }
    search = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    print("\nBest params:", search.best_params_)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

    # 3) Final, untouched hold-out evaluation
    best = search.best_estimator_
    proba = best.predict_proba(X_test)[:, 1]
    pred = best.predict(X_test)
    metrics = {
        "cv_models": cv_results,
        "best_params": search.best_params_,
        "best_cv_auc": round(search.best_score_, 4),
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "test_f1": round(f1_score(y_test, pred), 4),
        "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
    }
    print("\n", classification_report(y_test, pred))
    print(json.dumps(metrics, indent=2, default=str))
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # 4) Plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax[0], colorbar=False)
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax[1])
    ax[0].set_title("Confusion matrix (hold-out)")
    ax[1].set_title("ROC curve (hold-out)")
    fig.tight_layout()
    fig.savefig(REPORTS / "evaluation.png", dpi=150)

    # 5) Persist ONE artefact: preprocessing + model together
    joblib.dump(best, MODEL_PATH)
    print(f"\nSaved pipeline -> {MODEL_PATH.relative_to(ROOT)}")

    # 6) Reload and predict on RAW rows (what production would do)
    reloaded = joblib.load(MODEL_PATH)
    print("Reload check, first 5 raw rows ->", reloaded.predict(X_test.head()))


if __name__ == "__main__":
    main()
