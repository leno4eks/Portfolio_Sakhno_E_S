from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import numpy as np

app = FastAPI(
    title="Рекомендательный сервис",
    description="Топ-3 рекомендации по visitor_id",
    version="1.0"
)

# Загружаем модель
model = joblib.load("models/xgbranker_final.pkl")

# ТОЧНО те фичи, на которых обучена модель
feature_columns = [
    'views_x', 'adds_x', 'purchases_x', 'total_interactions', 'days_between',
    'interaction_intensity', 'views_y', 'adds_y', 'purchases_y', 'unique_items',
    'add_to_view_ratio_x', 'purchase_to_view_ratio_x', 'conversion_rate_x',
    'total_sessions', 'days_active_x', 'events_per_session', 'views', 'adds',
    'add_to_view_ratio_y', 'days_active_y', 'is_recent', 'category_level',
    'category_id'
]

# Загружаем item_features
try:
    item_features = pd.read_csv("features/item_popularity.csv")
except FileNotFoundError:
    item_features = pd.read_csv("features/item_mapping.csv")  # альтернатива
print(f"✅ Loaded {len(item_features)} items for candidates")

@app.get("/")
def root():
    return {"status": "ok", "message": "Сервис работает"}

@app.get("/recommend/{visitor_id}")
def recommend(visitor_id: int, top_k: int = 3):
    try:
        candidates = item_features.copy().sample(300, random_state=42)
        candidates = candidates.assign(visitorid=visitor_id)

        # Создаём DataFrame с правильными колонками
        X = pd.DataFrame(0.0, index=range(len(candidates)), columns=feature_columns)

        for col in feature_columns:
            if col in candidates.columns:
                X[col] = candidates[col].astype(float)

        scores = model.predict(X)

        top_indices = np.argsort(scores)[::-1][:top_k]
        top_items = candidates.iloc[top_indices]

        recommendations = []
        for i, row in enumerate(top_items.itertuples()):
            recommendations.append({
                "item_id": int(row.itemid),
                "score": float(scores[top_indices[i]])
            })

        return {
            "visitor_id": visitor_id,
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))