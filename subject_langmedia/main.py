from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import tempfile
from subject_langmedia.pipeline_langmedia import pipeline_langmedia  # 언어와 매체용 파이프라인 함수

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ok"}

class ProblemRequest(BaseModel):
    passage_img_base64: str = ""  # 선택적 입력
    question_img_base64: str
    top_k: int = 10

@app.post("/process")
def process_problem(req: ProblemRequest):
    try:
        passage_path = None

        # ✅ passage_img_base64가 비어있지 않으면 저장
        if req.passage_img_base64.strip():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_passage:
                tmp_passage.write(base64.b64decode(req.passage_img_base64))
                passage_path = tmp_passage.name

        # ✅ 질문 이미지 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_question:
            tmp_question.write(base64.b64decode(req.question_img_base64))
            question_path = tmp_question.name

        # ✅ 파이프라인 실행 (지문이 없어도 실행 가능)
        result = pipeline_langmedia(
            context_image_path=passage_path,
            question_image_path=question_path,
            top_k=req.top_k
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))