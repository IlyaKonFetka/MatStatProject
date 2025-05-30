from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

# Функции для построения графиков сравнения моделей и ROC-кривых

def plot_roc_curve(fpr, tpr, title: str, save_path: Path):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_model_comparison(metrics_dict: Dict[str, Dict[str, Dict[str, float]]], save_path: Path):
    """Сравнение моделей по ключевой метрике (R2 или Accuracy) в виде bar plot."""
    records = []
    for dataset, models_metrics in metrics_dict.items():
        for model_name, metric_vals in models_metrics.items():
            if "R2" in metric_vals:
                score = metric_vals["R2"]
                metric_name = "R2"
            elif "Accuracy" in metric_vals:
                score = metric_vals["Accuracy"]
                metric_name = "Accuracy"
            else:
                continue
            records.append({"Dataset": dataset, "Model": model_name, "Score": score, "Metric": metric_name})

    if not records:
        print("[WARN] Нет данных для построения графика сравнения моделей.")
        return

    import pandas as pd

    df_plot = pd.DataFrame(records)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_plot, x="Dataset", y="Score", hue="Model")
    plt.title("Сравнение моделей по основной метрике")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close() 