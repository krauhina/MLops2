import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train_params"]

df = pd.read_csv("data/processed/features.csv")

# Проверяем наличие целевого столбца
if 'target' not in df.columns:
    raise ValueError("Столбец 'target' отсутствует в данных")

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=params["random_state"])

model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    random_state=params["random_state"]
)
model.fit(X_train, y_train)

joblib.dump(model, "model/model.pkl")
print("Модель успешно обучена и сохранена")