# Потоковый скоринг фродовых транзакций

Сервис получает транзакции формата `test.csv` из Kafka, выполняет препроцессинг и CPU-инференс модели, публикует transaction_id, score, fraud_flag в топик scores. Отдельный сервис сохраняет результаты в PostgreSQL. Интерфейс позволяет отправить CSV и увидеть последние случаи фрода и распределение скоров.

## Что понадобится

- Docker и Docker Compose (`docker compose`)
- Свободные порты 8501 и 9092
- Для проверки через интерфейс: файл `test.csv` из [соревнования Kaggle](https://www.kaggle.com/competitions/teta-ml-1-2025)

Данные CSV не включены в репозиторий: они нужны только пользователю для отправки транзакций и не копируются в Docker-образ. Обученная модель уже включена в `fraud_detector/models/fraud_logreg.json`, поэтому обучать её для запуска не требуется.

## Запуск

```bash
git clone https://github.com/NovikovArtur/MLOps_HW1.git
cd MLOps_HW1
docker compose up --build -d
docker compose ps
```

Дождитесь, когда kafka и postgres будут healthy, kafka-setup завершится с кодом 0, а fraud-detector, score-writer и interface будут running. Первый запуск может занять несколько минут из-за загрузки образов.

Откройте [http://localhost:8501](http://localhost:8501), выберите `test.csv`, оставьте 100 строк для первого теста и нажмите **«Отправить в Kafka»**. Через несколько секунд перейдите во вкладку **«Результаты»** и нажмите **«Посмотреть результаты»**. В ней появится гистограмма последних 100 скоров (или меньшего числа, если записей пока меньше) и до 10 последних транзакций с fraud_flag = 1. Если фродовых записей ещё нет, интерфейс сообщит об этом.

Проверить количество результатов напрямую:

```bash
docker compose exec -T postgres psql -U fraud -d fraud -c "SELECT COUNT(*) FROM transaction_scores;"
```

Посмотреть последние результаты:

```bash
docker compose exec -T postgres psql -U fraud -d fraud -c "SELECT transaction_id, ROUND(score::numeric, 5) AS score, fraud_flag FROM transaction_scores ORDER BY scored_at DESC LIMIT 10;"
```

Логи при проблемах:

```bash
docker compose logs --tail=100 kafka kafka-setup fraud-detector score-writer interface
```

Остановка (данные PostgreSQL сохраняются в Docker volume):

```bash
docker compose down
```

## Формат сообщений

Топик transactions: один JSON-объект на сообщение, со всеми полями `test.csv` и строковым transaction_id. Интерфейс добавляет уникальный ID автоматически. Входные поля: transaction_time, merch, cat_id, amount, name_1, name_2, gender, street, one_city, us_state, post_code, lat, lon, population_city, jobs, merchant_lat, merchant_lon.

Топик scores: ровно три поля:

```json
{"transaction_id": "example-123", "score": 0.23, "fraud_flag": 1}
```

score — вероятность фрода от 0 до 1; fraud_flag равен 1, если score не ниже порога, выбранного на отложенной части обучающего набора. Повторная обработка сообщения не создаёт дублей в таблице благодаря ключу transaction_id.

## Структура проекта

- `fraud_detector/` — препроцессинг, модель, Kafka consumer/producer, отдельный сервис записи скоров, SQL-схема, Dockerfile и зависимости сервиса.
- `interface/` — Streamlit, Dockerfile и зависимости интерфейса.
- `docker-compose.yml` — запуск и соединение всех контейнеров.

Для повторного обучения локально положите `train.csv` в `fraud_detector/teta-ml-1-2025/`, установите зависимости из `fraud_detector/requirements-train.txt` и выполните 
```
python -m fraud_detector.train_model
``` 

из корня проекта. Препроцессинг при обучении и инференсе общий. Обучение запускается только вручную; на CPU контейнеры выполняют только инференс.
