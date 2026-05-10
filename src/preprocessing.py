import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def binarize_target(df, target_col="num"):
    df = df.copy()
    df[target_col] = (df[target_col] > 0).astype(int)
    return df


def identify_columns(df, target_col, id_cols=None):
    exclude = {target_col}
    if id_cols:
        exclude.update(id_cols)
    cols = [c for c in df.columns if c not in exclude]
    numeric_cols = df[cols].select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df[cols].select_dtypes(exclude=["number"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])
    return preprocessor


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def prepare_data(df):
    id_cols = ["id"] if "id" in df.columns else []
    df = binarize_target(df, target_col="num")

    X = df.drop(columns=["num"] + id_cols)
    y = df["num"]

    numeric_cols, categorical_cols = identify_columns(
        df, target_col="num", id_cols=id_cols
    )

    X_train_raw, X_test_raw, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    return X_train_raw, X_test_raw, y_train, y_test, preprocessor
