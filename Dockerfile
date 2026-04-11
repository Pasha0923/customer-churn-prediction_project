# Базовий образ контейнерва
FROM python:3.11-slim

# Встановимо робочу директорію всередині контейнера
WORKDIR /app

# Встановлюємо  poetry всередині контейнера
RUN pip install poetry

# Копіюємо файли конфігурації і встановлюємо залежності poetry  
COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

 # Копіюємо файли проєкту
COPY app.py .
COPY models ./models
COPY data ./data

# Відкриваємо порт Streamlit
EXPOSE 8501

# Запуск Streamlit при старті контейнера
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]