import pandas as pd
from src.preprocessing import build_preprocessor

df = pd.read_csv("data/titanic.csv")
X = df.drop(columns="Survived")
X_clean = build_preprocessor().fit_transform(X)
print(X_clean.shape)
print(X_clean.head())