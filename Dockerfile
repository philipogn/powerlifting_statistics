FROM python:3.11.5-slim

WORKDIR /app

COPY api/requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY models/ ./models/
COPY src/ ./src/
COPY api/app.py api/scraper.py ./api/

WORKDIR /app/api

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]