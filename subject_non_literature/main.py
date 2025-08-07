from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import tempfile
from subject_non_literature.pipeline_non_literature import pipeline_non_literature

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ok"}

class ProblemRequest(BaseModel):
    passage_img_base64: str
    question_img_base64 : str
    top_k : int = 3

@app.post("/process")
def process_problem(req: ProblemRequest):
    try:
        # 지문 이미지 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_passage:
            tmp_passage.write(base64.b64decode(req.passage_img_base64))
            passage_path = tmp_passage.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_question:
            tmp_question.write(base64.b64decode(req.question_img_base64))
            question_path = tmp_question.name

        # 파이프라인 실행
        gpt_response, similar_problems = pipeline_non_literature(
            passage_img_path=passage_path,
            question_img_path=question_path,
            top_k=req.top_k
        )

        return {
            "answer_explanation" : gpt_response,
            "similar_problems" : similar_problems
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))