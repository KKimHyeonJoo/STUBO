FROM python:3.10-slim
WORKDIR /app
COPY gateway/requirements.txt .
RUN pip install --no-deps-r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]