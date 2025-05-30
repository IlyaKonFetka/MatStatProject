from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import SVC, SVR

# Функции для обучения и оценки SVM

def train_evaluate_svc(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train,
    y_test,
    kernel: str = "rbf",
) -> Tuple[Dict[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray], SVC]:
    """Обучает SVC (классификация). Возвращает метрики, roc-данные и модель."""
    model = SVC(kernel=kernel, probability=True)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = float("nan")
    fpr, tpr, thresholds = roc_curve(y_test, probs)

    metrics = {"Accuracy": acc, "F1": f1, "AUC": auc}
    roc_data = (fpr, tpr, thresholds)
    return metrics, roc_data, model


def train_evaluate_svr(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train,
    y_test,
    kernel: str = "rbf",
) -> Tuple[Dict[str, float], SVR]:
    """Обучает SVR (регрессия). Возвращает метрики и модель."""
    model = SVR(kernel=kernel)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    metrics = {"RMSE": rmse, "R2": r2}
    return metrics, model 