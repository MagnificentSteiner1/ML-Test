FROM python:3.11.14

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


COPY . .
RUN pip install -r ML_req.txt


EXPOSE 5000

CMD ["uvicorn", "skripte.model_predict:app", "--host", "0.0.0.0", "--port", "5000"]