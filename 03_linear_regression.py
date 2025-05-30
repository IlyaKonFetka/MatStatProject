from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Функция для обучения и оценки линейной регрессии

def train_evaluate_linear_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train,
    y_test,
) -> Tuple[Dict[str, float], LinearRegression]:
    """Обучает LinearRegression и возвращает метрики и модель."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    metrics = {"RMSE": rmse, "R2": r2}
    return metrics, model 