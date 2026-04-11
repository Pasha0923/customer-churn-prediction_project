# Базовий образ контейнерва
FROM python:3.11-slim

# Встановимо робочу директорію всередині контейнера
WORKDIR /app

# Встановлюємо системні залежності
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

 # Копіюємо файли проєкту
COPY requirements.txt .
COPY app.py .
COPY models ./models
COPY data ./data

# Встановлюємо залежності
RUN pip install --no-cache-dir -r requirements.txt

# Відкриваємо порт Streamlit
EXPOSE 8501

# Запуск Streamlit при старті контейнера
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]