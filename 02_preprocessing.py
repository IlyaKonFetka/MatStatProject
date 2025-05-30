from pathlib import Path
from typing import Tuple, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Функции для очистки, масштабирования и кодирования данных

def _get_column_types(df: pd.DataFrame, categorical_threshold: int = 20) -> Tuple[List[str], List[str]]:
    """Определяет числовые и категориальные колонки.
    Числовые < threshold уникальных значений считаются категориальными."""
    cat_cols, num_cols = [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() < categorical_threshold:
                cat_cols.append(col)
            else:
                num_cols.append(col)
        else:
            cat_cols.append(col)
    return cat_cols, num_cols


def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Очищает, кодирует и масштабирует данные; возвращает разбиение train/test.

    Возвращает кортеж: (X_train, X_test, y_train, y_test, preprocessor)
    """
    # Простая очистка пропусков: удаляем строки с NaN (можно усложнить при необходимости)
    df = df.dropna().reset_index(drop=True)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    cat_cols, num_cols = _get_column_types(X)

    # Пайплайн преобразований
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )

    X_transformed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=test_size, random_state=random_state, stratify=None
    )

    return X_train, X_test, y_train, y_test, preprocessor

# Пример использования:
# X_train, X_test, y_train, y_test = preprocess_data(df) 