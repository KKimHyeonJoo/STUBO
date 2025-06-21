# 추출한 텍스트 파일에서 문제 추출 후 json 파일로 저장하는 코드
# type은 화작으로 고정

import re
import os
import json

# 연도와 시험 종류 리스트 정의
years = [2021, 2022, 2023, 2024]
exams = ['03']
subject = '화작'

def extract_problems(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    problems = []
    current_problem = []
    in_problem = False

    # 35~38, 40~45: 다양한 형태(점, 괄호, 공백 등)
    problem_num_pattern = re.compile(r'^[\.\s\(]*([3-4][0-9])[\.\s\)]*')
    # 39번만: 152039로 시작
    problem_39_pattern = re.compile(r'^152039')

    for line in lines:
        if problem_num_pattern.match(line.strip()) or problem_39_pattern.match(line.strip()):
            if current_problem:
                problems.append(''.join(current_problem).strip())
                current_problem = []
            in_problem = True
            current_problem.append(line)
        elif in_problem:
            current_problem.append(line)
    if current_problem:
        problems.append(''.join(current_problem).strip())

    return problems

def make_json(file_path, save_path):
    problems = extract_problems(file_path)
    json_list = []
    for prob in problems:
        item = {
            "문제": prob,
            "답": "",      # 수동 입력 필요
            "해설": "",    # 수동 입력 필요
            "type": "화작",
            "문제유형": "" # 수동 입력 필요
        }
        json_list.append(item)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_list, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {save_path}")

# 모든 연도/시험에 대해 반복 처리
base_dir = '/Users/chaewon/Desktop/STUBO/화법과 작문/save_text'
save_dir = '/Users/chaewon/Desktop/STUBO/화법과 작문/save_json'
for year in years:
    for exam in exams:
        src = os.path.join(base_dir, f'{year}-{exam}-{subject}.txt')
        dst = os.path.join(save_dir, f'{year}-{exam}-{subject}.json')
        if os.path.exists(src):
            make_json(src, dst)
        else:
            print(f"파일이 존재하지 않습니다: {src}")