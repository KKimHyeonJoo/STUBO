# 화작 문제 유형 태그 생성 코드
# 문제 저장한 json 파일에 유형 태그까지 추가함

import json
import openai
import os
import time
# OpenAI API 키 설정 (환경변수 사용 권장, 여기선 하드코딩)
client = openai.OpenAI(api_key="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")

## 태그 생성 함수
def generate_tags(question, ty):
    prompt = f"""아래는 국어 영역 언어와 매체 문제입니다.
    유형이 '언어' 인 경우, 언어 관련 태그를, '매체'인 경우 매체 관련 태그를 생성해 주세요.

    유형이 '언어'인 경우에는, 언어 관련 태그를 5개 생성하고, 
    유형이 '매체'인 경우에는, 매체 관련 태그를 3개, 문제의 출제 의도 태그를 2개씩 생성하여 총 5개의 태그를 생성해주세요.
    태그는 쉼표로 구분해서 출력해 주세요.
    
    언어 관련 태그의 후보에는 '훈민정음 이해', '중세 국어 변화', '단어 변화 양상', '모음 조화 이해', '시간 표현 이해', '부사어', '조사어', '서술어 자릿수', '사잇소리 표기 분석', '동음이의어 분석', '다의어 분석', 
    '자음 분석', '음운의 변동', '간접 인용', '종결 어미', '발화 형식 비교', '관형사절', '시간적 관계 이해', '사동 및 피동 표현', '주동 및 능동 표현', '문장 성분 분석' 등이 있고,
    
    위에서 언급한 언어 관련 태그도 예시일 뿐, 이 중에서만 찾으려고 하지말고, 다른 태그의 후보도 적극적으로 다양하게 찾으세요.

    매체 관련 태그의 후보에는 '정보 제시 이해', '매체 구성 요소 분석', '매체 표현 방식 이해', '매체의 목적 분석', '매체의 효과 분석', '매체의 시각적 요소 이해', '매체의 청각적 요소 이해', '매체의 상징적 의미 이해', '매체의 사회적 맥락 이해' 등이 있습니다.
    그리고 매체 관련 태그의 출제 의도는 '매체 구성 이해', '대화 내용 반영', '정보 제공 방식 분석', '비판적 사고', '자료 분석', '문장 요소 분석', '적절하지 않은 것 찾기' 등이 있습니다.

    매체 관련 태그의 후보, 출제 의도도 위에서 적어준 것은 예시일 뿐, 절대 이 중에서만 찾으려고 하지말고, 적극적으로 다양하게 찾으세요.

    언어 관련한 문제 출력 예시) 안긴문장 분석, 서술어 자릿수, 의미 관계 파악
    매체 관련한 문제 출력 예시) 게시판 구성 이해, 대화 내용 반영, 정보 제공 방식 분석, 비판적 사고, 자료 분석
  


문제: {question}

유형 : {ty}

태그:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 수능국어 언어와 매체 교육 전문가입니다."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
            temperature=0.5,
        )
        tags_text = response.choices[0].message.content.strip()
        tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
        return tags
    except Exception as e:
        print("⚠️ 에러 발생:", e)
        return []

# 메인 처리 함수
def main():
    input_dir = "C:/Users/user/STUBO/언어와매체/save_json"          # 문제 원본 디렉토리
    output_dir = "C:/Users/user/STUBO/언어와매체/save_json_tagged"  # 태그 추가된 문제 저장 디렉토리
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue

        input_path = os.path.join(input_dir, filename)
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"\n📄 [{filename}] 태깅 시작")

        for idx, item in enumerate(data):
            question = item.get("문제", "")
            passage = item.get("지문", "")
            ty = item.get("type", "언어")
            if not passage:
                tags = generate_tags(f"문제:{question}", ty)
            else:
                tags = generate_tags(f"지문:{passage} , 문제:{question}", ty)
            item["tags"] = tags   # 기존 `"tag": []` 항목에 덮어쓰기
            print(f"  #{idx+1} → {tags}")
            time.sleep(1)  # API 사용량 제한 대비

        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ 저장 완료: {output_path}")

if __name__ == "__main__":
    main()
