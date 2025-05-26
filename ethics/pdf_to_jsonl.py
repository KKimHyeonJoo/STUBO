import pdfplumber
import json

pdf_path = "EBS 2026학년도 수능특강 사회탐구영역 윤리와 사상(교사용).pdf"
output_path = "윤리와사상_final.jsonl"

data = []
current_question = ""
current_answer = ""
current_explanation = ""

with pdfplumber.open(pdf_path) as pdf:
    full_text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

chunks = full_text.split("#문항코드")
for chunk in chunks[1:]:
    lines = chunk.strip().split("\n")

    current_question = ""
    current_answer = ""
    current_explanation = ""
    step = None

    for idx, line in enumerate(lines):
        line = line.strip()

        if line.startswith("[문제]"):
            step = "question"
            current_question = line.replace("[문제]", "").strip()
        elif step == "question":
            if line.startswith("[정답") or line.startswith("[해설]"):
                step = None
            else:
                current_question += " " + line

        if "[정답" in line:
            if idx + 1 < len(lines):
                current_answer = lines[idx + 1].strip()

        if "[해설]" in line:
            step = "explanation"
        elif step == "explanation":
            current_explanation += " " + line

    if current_question and current_answer and current_explanation:
        entry = {
            "instruction": "다음 수능 문제를 풀고 정답과 해설을 제시해줘.",
            "input": current_question.strip(),
            "output": f"정답은 {current_answer}번이야. {current_explanation.strip()}"
        }
        data.append(entry)

with open(output_path, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 총 {len(data)}개의 문항을 JSONL로 저장했습니다.")
print(f"📁 저장 위치: {output_path}")
