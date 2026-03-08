FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY pyproject.toml .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "flask-api/main.py"]