import os

SUBJECT_SERVICE_MAP = {
    "문학": os.getenv("SERVICE_LITERATURE_URL", "http://subject-literature:8000/process"),
    "비문학": os.getenv("SERVICE_NON_LITERATURE_URL", "http://subject-non_literature:8000/process"),
    "화법과 작문": os.getenv("SERVICE_SPEECHCOMP_URL", "http://subject-speechcomp:8000/process"),
    "언어와 매체": os.getenv("SERVICE_LANGMEDIA_URL", "http://subject-langmedia:8000/process"),
}