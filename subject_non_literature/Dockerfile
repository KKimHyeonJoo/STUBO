FROM stubo-base:latest
WORKDIR /app

# ✅ requirements 복사 → 캐시 활용
COPY subject_non_literature/requirements_non_literature.txt ./requirements_non_literature.txt
RUN pip install -r requirements_non_literature.txt
RUN pip install numpy==2.0.2 --force-reinstall
# ✅ 중앙 모델 복사 (context 기준으로 작성!)
COPY models /app/models

# ✅ 코드 복사
COPY config.py /app/config.py
COPY subject_non_literature/pipeline_non_literature.py /app/pipeline_non_literature.py

CMD ["uvicorn", "subject_non_literature.main:app", "--host", "0.0.0.0", "--port", "8000"]