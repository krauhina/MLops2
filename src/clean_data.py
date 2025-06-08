import pandas as pd
import yaml
import numpy as np

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["clean_params"]

df = pd.read_csv("data/raw/your_data.csv")

# Безопасное удаление столбцов
cols_to_drop = [col for col in params.get("drop_columns", []) if col in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

# Обработка пропущенных значений (если появятся)
if "threshold" in params:
    df = df.dropna(thresh=params["threshold"] * len(df.columns))

df.to_csv("data/processed/cleaned_data.csv", index=False)
print("Данные успешно очищены. Сохранено строк:", len(df))