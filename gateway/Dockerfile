# gateway/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 1. requirements.txt 복사 (먼저 복사해야 캐시 활용)
COPY gateway/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# 2. gateway 소스 코드 복사
COPY gateway/ /app

# 3. 포트 노출
EXPOSE 8010

# 4. FastAPI 실행 (main.py 기준)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port=8010"]