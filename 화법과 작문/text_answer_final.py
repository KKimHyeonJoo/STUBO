import json
import re
import os

# 폴더 경로 지정
json_dir = '/Users/chaewon/Desktop/STUBO/화법과 작문/save_json'
answer_dir = '/Users/chaewon/Desktop/STUBO/화법과 작문/answer_text'
output_dir = '/Users/chaewon/Desktop/STUBO/화법과 작문/save_json_with_answers'  # 결과 저장할 새 폴더

os.makedirs(output_dir, exist_ok=True)

# JSON 폴더 내 파일 목록 가져오기
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

for json_file in json_files:
    # json 파일 이름 예: '2021-06-화작.json'
    base_name = os.path.splitext(json_file)[0]  # '2021-06-화작'

    # 대응하는 텍스트 파일 이름 (띄어쓰기 + ' 해설.txt')
    answer_file = base_name + ' 해설.txt'

    json_path = os.path.join(json_dir, json_file)
    answer_path = os.path.join(answer_dir, answer_file)

    # 정답/해설 텍스트 파일이 없으면 건너뜀
    if not os.path.exists(answer_path):
        print(f"정답 텍스트 파일 없음: {answer_path}, 건너뜀")
        continue

    # 1. JSON 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # 2. 정답 텍스트 읽기
    with open(answer_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 3. 텍스트에서 정답/해설 파싱 (패턴 필요에 따라 조정)
    pattern = r'문제번호:\s*(\d+)\s*정답:\s*(\d+)\s*해설:\s*([\s\S]*?)(?=문제번호:|\Z)'
    matches = re.findall(pattern, content)

    answer_dict = {}
    for num, ans, expl in matches:
        answer_dict[int(num)] = {
            "정답": ans.strip(),
            "해설": expl.strip()
        }

    # 4. 문제번호 추출 후 병합
    for q in questions:
        m = re.match(r'(\d+)\.', q['문제'].strip())
        if m:
            num = int(m.group(1))
            if num in answer_dict:
                q['답'] = answer_dict[num]['정답']
                q['해설'] = answer_dict[num]['해설']

    # 5. 결과 저장 (원본 파일명에 _with_answers 추가)
    output_path = os.path.join(output_dir, base_name + '_with_answers.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"{json_file} 처리 완료 → {output_path}")
