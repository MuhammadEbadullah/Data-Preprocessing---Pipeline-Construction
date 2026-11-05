"""Why Pipelines matter: a demonstration of data leakage.

Run:  python -m src.leakage_demo
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import build_full_pipeline, build_preprocessor
from src.preprocessing.pipeline import TARGET
from src.train import DATA_PATH, RANDOM_STATE


def titanic_demo() -> None:
    """Fit preprocessing on ALL rows (leaky) vs inside each CV fold (correct)."""
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop(columns=TARGET), df[TARGET]
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    model = LogisticRegression(max_iter=1000)

    # WRONG: statistics (median, scaler mean/std, rare categories) see the test folds
    X_leaky = build_preprocessor().fit_transform(X, y)
    leaky = cross_val_score(model, X_leaky, y, cv=cv, scoring="roc_auc")

    # RIGHT: preprocessing is re-fit on the training fold only
    correct = cross_val_score(build_full_pipeline(model), X, y, cv=cv, scoring="roc_auc")

    print("== Titanic: preprocessing fitted on all data vs per-fold ==")
    print(f"leaky   CV ROC-AUC: {leaky.mean():.4f}")
    print(f"correct CV ROC-AUC: {correct.mean():.4f}")
    print("(the gap is small here because imputing/scaling leaks little,\n"
          " but it is NOT zero and it grows with supervised steps - see below)\n")


def noise_demo(n_repeats: int = 10) -> None:
    """Pure-noise data: the true accuracy is 50%. Leakage fakes a great model.

    Averaged over several random datasets because 100 samples is very noisy.
    """
    leaky_scores, correct_scores = [], []
    for seed in range(n_repeats):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(100, 2000))
        y = rng.integers(0, 2, size=100)
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)

        # WRONG: feature selection looks at ALL labels before CV
        X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)
        leaky_scores.append(
            cross_val_score(LogisticRegression(max_iter=1000), X_sel, y, cv=cv).mean()
        )

        # RIGHT: selection happens inside the pipeline, per fold
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=20)),
            ("model", LogisticRegression(max_iter=1000)),
        ])
        correct_scores.append(cross_val_score(pipe, X, y, cv=cv).mean())

    print(f"== Pure noise, mean over {n_repeats} datasets (true accuracy = 0.50) ==")
    print(f"leaky   CV accuracy: {np.mean(leaky_scores):.3f}   <- looks like a great model, but it's a lie")
    print(f"correct CV accuracy: {np.mean(correct_scores):.3f}   <- honest estimate (~chance)")


if __name__ == "__main__":
    titanic_demo()
    noise_demo()
