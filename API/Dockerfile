FROM python:3.11.14

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY API_req.txt .

RUN pip install --no-cache-dir -r API_req.txt

COPY skripte/ ./skripte

EXPOSE 8000

CMD ["uvicorn", "skripte.main:app", "--host", "0.0.0.0", "--port", "8000"]