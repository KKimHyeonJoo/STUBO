from dotenv import load_dotenv
from pathlib import Path 
import os
from openai import OpenAI

load_dotenv()

# OpenAI API 설정
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)

IS_DOCKER = os.environ.get("DOCKER_ENV", "false").lower() == "true"

if IS_DOCKER:
    BASE_DIR = Path("/app")
else:
    BASE_DIR = Path.home() / "MSA" / "STUBO"

LITERATURE_IMAGE_DIR = BASE_DIR / "subject_literature" / "data" / "output_images_문학"
LITERATURE_DATA = BASE_DIR / "subject_literature" / "data"
LITERATURE_FAISS_EXAMINE = BASE_DIR / "subject_literature" / "faiss_index_examine"
LITERATURE_FAISS_SIMILAR = BASE_DIR / "subject_literature" / "faiss_index_similar"

NON_LITERATURE_IMAGE_DIR = BASE_DIR / "subject_non_literature" / "data" / "output_images_비문학"
NON_LITERATURE_DATA = BASE_DIR / "subject_non_literature" / "data"

SPEECH_COMP_IMAGE_DIR = BASE_DIR / "subject_speechcomp" / "data" / "output_images_화법과작문"
SPEECH_COMP_FAISS_DIR = BASE_DIR / "subject_speechcomp" / "outputs"

LANGMEDIA_IMAGE_DIR = BASE_DIR / "subject_langmedia" / "data" / "output_images_언어와매체"
LANGMEDIA_JSON_DIR = BASE_DIR / "subject_langmedia" / "data" / "save_json_tagged"


