import json
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score
)
import yaml

def main():
    # Загрузка параметров
    with open("params.yaml", "r") as f:
        eval_params = yaml.safe_load(f)["eval_params"]
    
    # Загрузка данных и модели
    data = pd.read_csv("data/processed/features.csv")
    model = joblib.load("model/model.pkl")
    
    # Разделение данных (если нужно)
    X = data.drop(columns=["target"])
    y = data["target"]
    
    # Предсказание
    y_pred = model.predict(X)
    
    # Расчет метрик
    metrics = {}
    if "accuracy" in eval_params["metrics"]:
        metrics["accuracy"] = accuracy_score(y, y_pred)
    if "f1" in eval_params["metrics"]:
        metrics["f1"] = f1_score(y, y_pred)
    if "precision" in eval_params["metrics"]:
        metrics["precision"] = precision_score(y, y_pred)
    if "recall" in eval_params["metrics"]:
        metrics["recall"] = recall_score(y, y_pred)
    
    # Сохранение результатов
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("Метрики сохранены в metrics.json")

if __name__ == "__main__":
    main()