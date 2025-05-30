import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Загружаем локальные модули через exec
exec(open('02_preprocessing.py').read())
exec(open('03_linear_regression.py').read())
exec(open('04_logistic_regression.py').read()) 
exec(open('05_svm.py').read())
exec(open('06_evaluation.py').read())

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

# Конфигурация датасетов с указанием типа задачи
DATASETS_CONFIG = {
    "insurance": {"file": "data/insurance.csv", "target": "charges", "task": "regression"},
    "winequality": {"file": "data/winequality.csv", "target": "quality", "task": "both"},  
    "titanic": {"file": "data/titanic.csv", "target": "Survived", "task": "classification"},
    "heart": {"file": "data/heart.csv", "target": "target", "task": "classification"},
}

def run_classification_models(X_train, X_test, y_train, y_test, dataset_name):
    """Запускает модели классификации и сохраняет результаты."""
    results = {}
    
    # Logistic Regression
    try:
        log_metrics, (fpr, tpr, _), _ = train_evaluate_logistic_regression(X_train, X_test, y_train, y_test)
        results["LogisticRegression"] = log_metrics
        
        # ROC curve
        roc_path = IMAGES_DIR / f"roc_curve_{dataset_name}_logistic.png"
        plot_roc_curve(fpr, tpr, f"ROC Curve - {dataset_name} (Logistic)", roc_path)
        print(f"[INFO] Logistic Regression для {dataset_name}: {log_metrics}")
    except Exception as e:
        print(f"[ERROR] Logistic Regression для {dataset_name}: {e}")
    
    # SVC
    try:
        svc_metrics, (fpr, tpr, _), _ = train_evaluate_svc(X_train, X_test, y_train, y_test)
        results["SVC"] = svc_metrics
        
        # ROC curve
        roc_path = IMAGES_DIR / f"roc_curve_{dataset_name}_svc.png"
        plot_roc_curve(fpr, tpr, f"ROC Curve - {dataset_name} (SVC)", roc_path)
        print(f"[INFO] SVC для {dataset_name}: {svc_metrics}")
    except Exception as e:
        print(f"[ERROR] SVC для {dataset_name}: {e}")
    
    return results

def run_regression_models(X_train, X_test, y_train, y_test, dataset_name):
    """Запускает модели регрессии и сохраняет результаты."""
    results = {}
    
    # Linear Regression
    try:
        lin_metrics, _ = train_evaluate_linear_regression(X_train, X_test, y_train, y_test)
        results["LinearRegression"] = lin_metrics
        print(f"[INFO] Linear Regression для {dataset_name}: {lin_metrics}")
    except Exception as e:
        print(f"[ERROR] Linear Regression для {dataset_name}: {e}")
    
    # SVR
    try:
        svr_metrics, _ = train_evaluate_svr(X_train, X_test, y_train, y_test)
        results["SVR"] = svr_metrics
        print(f"[INFO] SVR для {dataset_name}: {svr_metrics}")
    except Exception as e:
        print(f"[ERROR] SVR для {dataset_name}: {e}")
    
    return results

def main():
    all_metrics = {}
    
    for dataset_name, config in DATASETS_CONFIG.items():
        print(f"\n{'='*50}")
        print(f"Обработка датасета: {dataset_name}")
        print(f"{'='*50}")
        
        # Загрузка данных
        try:
            df = pd.read_csv(config["file"])
            print(f"Загружен {config['file']}: {df.shape}")
        except Exception as e:
            print(f"[ERROR] Не удалось загрузить {config['file']}: {e}")
            continue
        
        # Preprocessing
        try:
            X_train, X_test, y_train, y_test, _ = preprocess_data(df, config["target"])
            print(f"Preprocessing завершён: train={X_train.shape}, test={X_test.shape}")
        except Exception as e:
            print(f"[ERROR] Preprocessing для {dataset_name}: {e}")
            continue
        
        # Выбор моделей по типу задачи
        dataset_metrics = {}
        
        if config["task"] == "classification":
            dataset_metrics = run_classification_models(X_train, X_test, y_train, y_test, dataset_name)
        elif config["task"] == "regression":
            dataset_metrics = run_regression_models(X_train, X_test, y_train, y_test, dataset_name)
        elif config["task"] == "both":
            # Для wine quality: классификация (бинаризация по порогу) и регрессия
            print(f"[INFO] Wine quality: выполняю регрессию и классификацию")
            
            # Регрессия
            reg_metrics = run_regression_models(X_train, X_test, y_train, y_test, dataset_name)
            dataset_metrics.update({f"{k}_regression": v for k, v in reg_metrics.items()})
            
            # Классификация (бинаризация: quality >= 6 = хорошее вино)
            y_train_binary = (y_train >= 6).astype(int)
            y_test_binary = (y_test >= 6).astype(int)
            
            class_metrics = run_classification_models(X_train, X_test, y_train_binary, y_test_binary, f"{dataset_name}_binary")
            dataset_metrics.update({f"{k}_classification": v for k, v in class_metrics.items()})
        
        all_metrics[dataset_name] = dataset_metrics
    
    # Сравнение моделей
    print(f"\n{'='*50}")
    print("Сравнение всех моделей")
    print(f"{'='*50}")
    
    comparison_path = IMAGES_DIR / "model_comparison.png"
    plot_model_comparison(all_metrics, comparison_path)
    print(f"[INFO] Сравнительный график сохранён: {comparison_path}")
    
    # Итоговый отчёт
    print(f"\n{'='*50}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print(f"{'='*50}")
    
    for dataset_name, models in all_metrics.items():
        print(f"\n{dataset_name.upper()}:")
        for model_name, metrics in models.items():
            metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            print(f"  {model_name}: {metrics_str}")

if __name__ == "__main__":
    main() 