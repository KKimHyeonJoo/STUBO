FROM stubo-base:latest
WORKDIR /app

# ✅ requirements 복사 → 캐시 활용
COPY subject_speechcomp/requirements_speechcomp.txt ./requirements_speechcomp.txt
RUN pip install -r requirements_speechcomp.txt
RUN pip install numpy==2.0.2 --force-reinstall
# ✅ 중앙 모델 복사 (context 기준으로 작성!)
COPY models /app/models
# ✅ 코드 복사
COPY config.py /app/config.py
COPY subject_speechcomp/pipeline_speechcomp.py /app/pipeline_speechcomp.py

CMD ["uvicorn", "subject_speechcomp.main:app", "--host", "0.0.0.0", "--port", "8000"]