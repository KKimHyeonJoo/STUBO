from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from service_config import SUBJECT_SERVICE_MAP

app = FastAPI()

class ProblemRequest(BaseModel):
    subject: str # ex. 문학
    passage_img_base64 : str
    question_img_base64 : str
    top_k : int = 10

@app.post("/solve")
async def solve_problem(req : ProblemRequest):
    if req.subject not in SUBJECT_SERVICE_MAP:
        raise HTTPException(status_code=400, detail="지원하지 않는 과목입니다.")
    service_url = SUBJECT_SERVICE_MAP[req.subject]

    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(service_url, json=req.dict())
            res.raise_for_status()
            return res.json()
        
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"마이크로서비스 요청 실패 : {e}")