import pdfplumber
import json
import os
import re

# 경로 설정
pdf_path = "/Users/chaewon/Desktop/STUBO/ethics/data/pdf/ethics.pdf"
image_dir = "/Users/chaewon/Desktop/STUBO/ethics/data/images"
output_path = "/Users/chaewon/Desktop/STUBO/ethics/data/jsonl/parsing.jsonl"

def find_image_path(question_id):
    match = re.search(r"(\d{4})$", question_id)
    if match:
        qnum = match.group(1)[-3:]
        filename = f"q_{qnum}_0.jpeg"
        full_path = os.path.join(image_dir, filename)
        if os.path.exists(full_path):
            return f"[IMAGE:{full_path}]"
    return None

def extract_question_id(lines):
    for l in lines:
        m = re.match(r"^\[?0*([1-9]\d*)", l.strip())
        if m:
            return l.strip()
    return lines[0].strip() if lines else "unknown"

data = []
shared_context_text = ""
shared_context_count = 0

with pdfplumber.open(pdf_path) as pdf:
    full_text = "\n".join([p.extract_text() or "" for p in pdf.pages])
    chunks = full_text.split("#문항코드")

    for chunk in chunks[1:]:
        lines = chunk.strip().split("\n")
        question_id = extract_question_id(lines)

        # [지문] 감지 및 저장
        found_shared = False
        for idx, l in enumerate(lines):
            if "[지문]" in l:
                shared_context_text = ' '.join([x.strip() for x in lines[idx+1:]]).strip()
                shared_context_count = 2
                found_shared = True
                break
        if found_shared:
            continue  # [지문] chunk는 문제로 저장하지 않음

        explanation_title = ""
        question_lines = []
        context_lines = []
        choices = []
        answer = ""
        context_explanation = ""
        answer_explanation = ""
        step = None
        before_answer = True
        question_done = False

        for line in lines[1:]:
            line = line.strip()
            if line.startswith("[해설]"):
                step = "explanation_title"
                continue
            elif line.startswith("[문제]"):
                step = "question"
                question_done = False
                continue
            elif line.startswith("[정답/모범답안]"):
                step = "answer"
                before_answer = False
                continue
            elif line.startswith("{문제 분석}"):
                step = "context_explanation"
                continue
            elif line.startswith("{정답 찾기}") or line.startswith("{오답 피하기}"):
                step = "answer_explanation"
                continue
            elif re.match(r"^[①-⑤]", line) and before_answer:
                step = "choices"
            else:
                if step == "choices" and not re.match(r"^[①-⑤]", line):
                    step = None

            if step == "explanation_title" and not explanation_title:
                explanation_title = line
            elif step == "question" and not question_done:
                question_lines.append(line)
                if line.endswith("?") or line.endswith("것은.") or line.endswith("것은?") or line.endswith("무엇인가?"):
                    question_done = True
                    step = None  # 질문 끝, 다음 줄부터 본문
            elif (step == "question" and question_done) or (step is None and question_done and before_answer):
                # 질문이 끝난 뒤, choices/정답/해설/파트 키워드가 아니면 context로 누적
                if not (re.match(r"^[①-⑤]", line) or line.startswith("[정답/모범답안]") or line.startswith("[해설]") or line.startswith("{문제 분석}") or line.startswith("{정답 찾기}") or line.startswith("{오답 피하기}")):
                    context_lines.append(line)
            elif step == "choices" and before_answer:
                if re.match(r"^[①-⑤]", line):
                    choices.append(line)
                elif choices:
                    choices[-1] += " " + line
            elif step == "answer" and not answer:
                answer = re.sub(r"[^0-9]", "", line).strip()
            elif step == "context_explanation":
                context_explanation += line + " "
            elif step == "answer_explanation":
                answer_explanation += line + " "

        question = ' '.join(question_lines).strip()
        context = ' '.join(context_lines).strip()

        # 이미지 컨텍스트 우선 처리
        image_context = find_image_path(question_id)
        if image_context:
            final_context = image_context
        else:
            final_context = context

        # shared_context 적용
        if shared_context_count > 0:
            item_shared_context = shared_context_text
            shared_context_count -= 1
        else:
            item_shared_context = ""

        item = {
            "explanation_title": explanation_title.strip(),
            "question": question,
            "shared_context": item_shared_context,
            "context": final_context,
            "choices": choices,
            "answer": answer,
            "context_explanation": context_explanation.strip(),
            "answer_explanation": answer_explanation.strip()
        }
        data.append(item)

# 저장
with open(output_path, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 총 {len(data)}개 문항 저장 완료: {output_path}")
