# frontend/Dockerfile

FROM stubo-base:latest
WORKDIR /app

# ✅ 1. requirements.txt 먼저 복사
COPY frontend/streamlit_app/requirements_frontend.txt ./requirements_frontend.txt
RUN pip install -r requirements_frontend.txt
RUN pip install numpy==2.0.2 --force-reinstall
# ✅ 2. 나머지 코드 복사 (절대 context는 프로젝트 루트 기준)
COPY frontend/streamlit_app /app
COPY subject_literature /app/subject_literature
COPY subject_non_literature /app/subject_non_literature
COPY subject_speechcomp /app/subject_speechcomp
COPY subject_langmedia /app/subject_langmedia
COPY config.py /app/config.py
COPY models /app/models

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]