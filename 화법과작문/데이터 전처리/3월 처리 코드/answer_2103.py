# 3월 모의고사(2021) 해설에서 문제 번호, 정답, 해설 추출하는 코드
# 평가원 답지랑 형식이 달라서 기존의 6, 9, 수능 모의고사와는 다른 형태로 추출해야 함
# 결과를 텍스트 파일로 저장

import fitz  # PyMuPDF
import re
import json

def extract_hwajak_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

def extract_explanations(text):
    explanations = {}

    start = text.find("화법과 작문")
    end = text.find("언어와 매체")
    if start == -1 or end == -1:
        print("'화법과 작문' 또는 '언어와 매체' 구간을 찾을 수 없습니다.")
        return {}

    text_hwajak = text[start:end]

    # 개선된 정규식 패턴
    pattern = re.compile(
        r"(\d+)[\.:]\s*출제의도\s*(.*?)\[(.*?)\](.*?)(?=\d+[\.:]\s*출제의도|\Z)", 
        re.DOTALL
    )

    for match in pattern.finditer(text_hwajak):
        q_num = int(match.group(1))
        if q_num in explanations:
            continue

        purpose = match.group(2).strip() + " " + match.group(3).strip()
        explanation = match.group(4).strip()

        explanations[q_num] = {
            "출제의도": purpose.strip(),
            "해설": explanation.strip()
        }

    return explanations

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    pdf_path = "/Users/chaewon/Desktop/STUBO/화법과 작문/해설/2021-03-화작 해설.pdf"
    output_path = "/Users/chaewon/Desktop/STUBO/화법과 작문/answer_text/2021-03-화작 해설.txt"

    text = extract_hwajak_text(pdf_path)
    print(text)

    # 수동 정답 입력
    answers = {
        35: "3", 36: "5", 37: "5", 38: "2", 39: "4",
        40: "2", 41: "1", 42: "4", 43: "3", 44: "1", 45: "4"
    }

    # 해설 자동 추출
    explanations = extract_explanations(text)
    print(explanations)

    final_output = []
    for q_num in sorted(answers):
        if q_num not in explanations:
            print(f"문항 {q_num} 해설 없음")
            continue
        final_output.append(
            {
                "문제번호": q_num,
                "정답": answers[q_num],
                "해설": explanations[q_num]["해설"]
            }
        )

    save_json(final_output, output_path)
    print(f"\n저장 완료: {output_path}")

if __name__ == "__main__":
    main()
