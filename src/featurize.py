import pandas as pd
import yaml
from sklearn.preprocessing import PolynomialFeatures

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["feature_params"]

df = pd.read_csv("data/processed/cleaned_data.csv")

# Сохраняем целевой столбец перед преобразованиями
target = df['target']

# Генерация полиномиальных признаков (только для числовых столбцов)
numeric_cols = df.select_dtypes(include=['number']).columns.drop('target')
poly = PolynomialFeatures(degree=params["polynomial_degree"])
features = poly.fit_transform(df[numeric_cols])

# Создаем новый DataFrame с сохранением target
feature_names = poly.get_feature_names_out(numeric_cols)
df_features = pd.DataFrame(features, columns=feature_names)
df_features['target'] = target.values  # Добавляем целевой столбец обратно

df_features.to_csv("data/processed/features.csv", index=False)