FROM stubo-base:latest
WORKDIR /app

# ✅ requirements 복사 → 캐시 활용
COPY subject_langmedia/requirements_langmedia.txt ./requirements_langmedia.txt
RUN pip install -r requirements_langmedia.txt
RUN pip install numpy==2.0.2 --force-reinstall
# ✅ 중앙 모델 복사 (context 기준으로 작성!)
COPY models /app/models

# ✅ 코드 복사
COPY config.py /app/config.py
COPY subject_langmedia /app/subject_langmedia
CMD ["uvicorn", "subject_langmedia.main:app", "--host", "0.0.0.0", "--port", "8000"]