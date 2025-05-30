from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

# Функция для обучения и оценки логистической регрессии

def train_evaluate_logistic_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train,
    y_test,
) -> Tuple[Dict[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray], LogisticRegression]:
    """Обучает LogisticRegression и возвращает метрики, roc-данные и модель."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = float('nan')
    fpr, tpr, thresholds = roc_curve(y_test, probs)

    metrics = {"Accuracy": acc, "F1": f1, "AUC": auc}
    roc_data = (fpr, tpr, thresholds)
    return metrics, roc_data, model 