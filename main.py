from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import joblib
from typing import Dict
import pandas as pd



# Загрузка модели
loaded_model = joblib.load('model.pkl')

# Инициализация FastAPI приложения
app = FastAPI()

# Переменные для отслеживания запросов
total_queries = 0
successful_queries = 0


def json2df(data: Dict) -> pd.DataFrame:
    """Преобразует JSON в DataFrame."""
    df_json = pd.DataFrame([data])

    # Переименование столбцов согласно ожиданиям модели
    df_json = df_json.rename(columns={
        "promo_type": "Promo_type",
        'feat_2': 'Feat_2',
        'feat_3': 'Feat_3',
        "agent": "Agent",
        'feat_7': 'Feat_7',
        'promo_class': 'Promo_class',
        'item_id': 'Item_id',
        'feat_9': 'Feat_9',
        'feat_10': 'Feat_10',
        'feat_11': 'Feat_11',
        'feat_12': 'Feat_12'
    })

    # Вычисление продолжительности промоакции и доставки
    df_json['promo_duration'] = (pd.to_datetime(df_json['promo_end']) - pd.to_datetime(df_json['promo_start'])).dt.days
    df_json['shipping_duration'] = (
                pd.to_datetime(df_json['shipping_end']) - pd.to_datetime(df_json['shipping_start'])).dt.days

    # Удаление ненужных столбцов
    df_json = df_json.drop(columns=['promo_start', 'promo_end', 'shipping_start', 'shipping_end', 'promo_id'])
    return df_json


@app.get('/ping')
async def ping():
    """Обработка GET запроса для проверки статуса сервера."""
    global total_queries, successful_queries
    total_queries += 1
    successful_queries += 1  # Успешный запрос
    return JSONResponse(status_code=200, content={
        "status": "ok",
        "total_queries": total_queries,
        "successful_queries": successful_queries
    })


@app.post('/inference')
async def predict(data: Dict = Body(...)):
    """Обработка POST запроса для получения прогноза."""
    global successful_queries
    try:
        if len(data) != 16:
            return JSONResponse(status_code=400, content={"error": "Должно быть ровно 16 входных значений."})

        x = json2df(data)
        pred = loaded_model.predict(x)[0]

        # Увеличение счётчика успешных запросов
        successful_queries += 1

        return {"prediction": int(pred), "model": "catboost"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
