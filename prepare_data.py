import pandas as pd
from pathlib import Path

# Объединение данных wine quality
red_wine = pd.read_csv("data/wine+quality/winequality-red.csv", sep=";")
white_wine = pd.read_csv("data/wine+quality/winequality-white.csv", sep=";")

red_wine["type"] = "red"
white_wine["type"] = "white"

wine_combined = pd.concat([red_wine, white_wine], ignore_index=True)
wine_combined.to_csv("data/winequality.csv", index=False)
print(f"[INFO] Создан winequality.csv: {len(wine_combined)} строк")

# Объединение данных titanic
train_titanic = pd.read_csv("data/titanic/train.csv")
test_titanic = pd.read_csv("data/titanic/test.csv")

# В test.csv нет колонки Survived, добавим NaN
test_titanic["Survived"] = None

titanic_combined = pd.concat([train_titanic, test_titanic], ignore_index=True)
titanic_combined.to_csv("data/titanic.csv", index=False)
print(f"[INFO] Создан titanic.csv: {len(titanic_combined)} строк")

print("[INFO] Подготовка данных завершена!") 