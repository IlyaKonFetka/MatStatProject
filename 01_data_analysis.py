import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

# Конфигурация датасетов: путь и колонка-цель
DATASETS_INFO = {
    "insurance": {"file": "data/insurance.csv", "target": "charges"},
    "titanic": {"file": "data/titanic.csv", "target": "Survived"},
    "winequality": {"file": "data/winequality.csv", "target": "quality"},
    "heart": {"file": "data/heart.csv", "target": "target"},
}


def plot_histograms(df: pd.DataFrame, numeric_cols: List[str], title: str, save_path: Path):
    """Строит гистограммы для всех числовых признаков."""
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    plt.figure(figsize=(n_cols * 4, n_rows * 3))
    for idx, col in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, n_cols, idx)
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(col)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, title: str, save_path: Path):
    """Рисует тепловую карту корреляций между числовыми признаками."""
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    for name, info in DATASETS_INFO.items():
        path = Path(info["file"])
        if not path.exists():
            print(f"[WARN] Файл {path} не найден, пропускаю {name}.")
            continue
        df = pd.read_csv(path)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # Гистограммы
        hist_path = IMAGES_DIR / f"hist_{name}.png"
        plot_histograms(df, numeric_cols, f"Гистограммы: {name}", hist_path)

        # Корреляция
        corr_path = IMAGES_DIR / f"correlation_heatmap_{name}.png"
        plot_correlation_heatmap(df, f"Корреляция: {name}", corr_path)

        print(f"[INFO] Сохранены графики для {name} в {IMAGES_DIR}.")


if __name__ == "__main__":
    main() 