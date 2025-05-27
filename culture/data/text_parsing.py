import re
import json

with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()

questions_raw = re.split(r"#문항코드 \d{5}-\d{4}", text)
parsed_questions = []

for q in questions_raw[1:]:
    item = {"문제": "", "선지": [], "정답": "", "해설": ""}

    # 문제 텍스트 추출
    문제_match = re.search(r"\[문제\](.*?)①", q, re.DOTALL)
    if 문제_match:
        item["문제"] = 문제_match.group(1).strip()

    # 선택지 정확히 ①~⑤까지 고정으로 파싱
    선지_패턴 = r"①(.*?)②(.*?)③(.*?)④(.*?)⑤(.*?)(?:\[정답/모범답안|\[해설|\Z)"
    선지_match = re.search(선지_패턴, q, re.DOTALL)
    if 선지_match:
        item["선지"] = [("① " + 선지_match.group(1).strip()),
                      "② " + 선지_match.group(2).strip(),
                      "③ " + 선지_match.group(3).strip(),
                      "④ " + 선지_match.group(4).strip(),
                      "⑤ " + 선지_match.group(5).strip()]
    
    # 정답
    정답_match = re.search(r"\[정답/모범답안\]\s*(\d)", q)
    if 정답_match:
        item["정답"] = 정답_match.group(1)

    # 해설
    해설_match = re.search(r"\[해설\](.*)", q, re.DOTALL)
    if 해설_match:
        item["해설"] = 해설_match.group(1).strip()

    parsed_questions.append(item)

# 저장
with open("questions_clean.json", "w", encoding="utf-8") as f:
    json.dump(parsed_questions, f, ensure_ascii=False, indent=2)

print(f"✅ 총 {len(parsed_questions)}개 문항을 구조화하여 'questions_clean.json'에 저장했습니다.")
