# Предсказание объема реализованного товара 

Рассмотренны такие модели, как:
*  xgboost
*  Ridge
*  catboost
*  Нейронные сети

Подбор гиперпараметров с помощью `optuna`.

Предобработка данных с помощью `IterativeImputer`.
  

# Инструкция по запуску сервера

1. Убедитесь, что у вас установлен Python 3.6 или выше.

2. Установите необходимые библиотеки:
    pip install fastapi joblib pandas catboost
    pip install uvicorn  python-multipart# Установим ASGI-сервер

3. Поместите вашу модель `model.pkl` в ту же директорию, что и `main.py`.

4. Запустите сервер:
 nohup uvicorn main:app --reload &

5. Проверка статуса:
- Откройте в браузере или используйте cURL:
```
curl http://127.0.0.1:8000/ping
```
Пример ответа:
```
{"status":"ok","total_queries":4,"successful_queries":4}
```
6.Осуществите запрос на предсказание:
```
curl -X POST "http://127.0.0.1:8000/inference" -H "Content-Type: application/json" -d '{
    "promo_start": "2024-07-24",
    "promo_end": "2024-07-30",
    "shipping_start": "2024-06-25",
    "shipping_end": "2024-07-30",
    "promo_type": "J",
    "feat_2": 10328.21,
    "feat_3": 58.22,
    "agent": "C",
    "promo_id": "Promo №5483.0",
    "item_id": "Item ID: 125.0",
    "feat_7": 32270.51,
    "promo_class": "D",
    "feat_9": 84812245.76,
    "feat_10": 5718813.46,
    "feat_11": 25.96,
    "feat_12": 84212
}'
```
Пример ответа:
```
{"prediction":933,"model":"catboost"}
```
