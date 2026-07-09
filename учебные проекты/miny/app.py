from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
import pandas as pd
from catboost import CatBoostRegressor
from datetime import datetime  # ← Добавили

app = FastAPI(title="Wine Quality Model")

# Загружаем модель
model = CatBoostRegressor()
model.load_model('wine_quality_model.cbm')

# Метрики Prometheus
PREDICTIONS = Counter('wine_predictions_total', 'Total predictions')
LATENCY = Histogram('wine_prediction_latency_seconds', 'Prediction time')
AVG_QUALITY = Gauge('wine_avg_predicted_quality', 'Average predicted quality')
MODEL_VERSION = Gauge('wine_model_version', 'Model version (timestamp)')

# Устанавливаем версию модели (добавили это)
MODEL_VERSION.set(int(datetime.now().timestamp()))

@app.post("/predict")
async def predict(request: Request):
    start = time.time()
    data = await request.json()
    
    df = pd.DataFrame([data])
    quality = float(model.predict(df)[0])
    
    PREDICTIONS.inc()
    LATENCY.observe(time.time() - start)
    AVG_QUALITY.set(quality)
    
    return {"quality": quality}

@app.get("/metrics")
def metrics():
    """Отдаём метрики в правильном формате для Prometheus"""
    return generate_latest(), 200, {
        "Content-Type": "text/plain; version=0.0.4; charset=utf-8"
    }

@app.get("/")
def home():
    return {"message": "Модель работает! POST на /predict"}