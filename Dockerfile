# Указываем базовый образ
FROM python:3.9-slim

# Копируем текущую директорию в директорию /app контейнера
COPY . /app

# Устанавливаем необходимые библиотеки
RUN pip install --no-cache-dir -r /app/requirements.txt

# Указываем рабочую директорию
WORKDIR /app

# Запускаем скрипт
CMD ["python", "work_with_tables.py"]

