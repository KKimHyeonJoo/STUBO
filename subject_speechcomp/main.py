from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import tempfile
from subject_speechcomp.pipeline_speechcomp import analyze_problem

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ok"}

class ProblemRequest(BaseModel):
    passage_img_base64: str  # 필수
    question_img_base64: str  # 필수
    top_k: int = 10

@app.post("/process")
def process_problem(req: ProblemRequest):
    try:
        # ✅ 지문 이미지 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_passage:
            tmp_passage.write(base64.b64decode(req.passage_img_base64))
            passage_path = tmp_passage.name

        # ✅ 문제 이미지 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_question:
            tmp_question.write(base64.b64decode(req.question_img_base64))
            question_path = tmp_question.name

        # ✅ 분석 실행
        result = analyze_problem(
            context_image_path=passage_path,
            question_image_path=question_path,
            top_k=req.top_k
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))