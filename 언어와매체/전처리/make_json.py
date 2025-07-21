# 추출한 텍스트 파일에서 문제 추출 후 json 파일로 저장하는 코드
# type은 언어와매체으로 고정

import re
import os
import json

years = [2022, 2023, 2024, 2025]
exams = ['03', '06', '09', '수능']
subject = '언어와매체'
subject_initial = '언어와매체'



# 전처리 설정
UNWANTED_NUMBERS = [
    "120", "220", "320", "420", "520", "620", "720", "820", "920", "1020",
    "1120", "1220", "1320", "1420", "1520", "1620", "1720", "1820", "1920", "2020"
]

UNWANTED_PATTERNS = [
    r"국어영역\(언어와매체\)[^\n]*",
    r"이 문제지에 관한 저작권은 한국교육과정평가원에 있습니다."
    r"국어영역[^\n]*",
    r"\d{4}학년도\s*[^\n]*?모의평가\s*문제지",
    r"\d{4}학년도\s*3월\s*고3\s*전국연합학력평가\s*문제지",
    r"제1\s*교\s*시",
    r"\*+\s*확인\s*사항[^\n]*",
    r"Çt\s+[^0-9\n]+",  # 깨진 문자열
    r"◦[^\n]*"
]

# 불필요한 숫자 제거 정규식
unwanted_numbers_regex = re.compile(r'\b(?:' + '|'.join(UNWANTED_NUMBERS) + r')\b')

# 온점 뒤 숫자 붙는 경우 처리 정규식 (줄바꿈 없이)
dot_number_split_regex = re.compile(r'(?<=\.)(?=(3[5-9]|4[0-5])[\.\)])')

# 공유 지문 예고 분리 정규식
shared_passage_notice_regex = re.compile(r'(?<!^)(?=\[\d{1,2}~\d{1,2}\])')

# 2020[44~45] 같이 연도와 붙은 공유 지문 예고 분리
year_shared_regex = re.compile(r'(?<=\d{4})\[(\d{1,2}~\d{1,2})\]')

# 불필요한 패턴 제거 함수
def remove_unwanted_patterns(text):
    text = unwanted_numbers_regex.sub('', text)
    for pattern in UNWANTED_PATTERNS:
        text = re.sub(pattern, '', text)
    return text

# 온점 뒤 숫자 붙는 줄 분리 및 공유 지문 예고 분리
def split_problem_lines(text):
    text = dot_number_split_regex.sub('\n', text)
    text = shared_passage_notice_regex.sub('\n', text)
    text = year_shared_regex.sub(r'\n[\1]', text)
    return text

# 문제 추출
def extract_problems_from_text(text):
    lines = text.split('\n')
    problems = []
    current = []

    pattern = re.compile(r'^\(?(3[5-9]|4[0-5])[\.\)\s]')


    for line in lines:
        if pattern.match(line.strip()):
            if current:
                problems.append('\n'.join(current).strip())
                current = []
            current.append(line)
        else:
            current.append(line)
    if current:
        problems.append('\n'.join(current).strip())
    return problems



def make_json(file_path, save_path):
    # 전체 파이프라인 실행
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    cleaned = remove_unwanted_patterns(raw_text)
    split_text = split_problem_lines(cleaned)
    problems = extract_problems_from_text(split_text)

    json_list = []
    for prob in problems:
        item = {
            "문제": prob,
            "type": "언어와매체",
            "med": False,
            "tags": []  # 수동 입력 가능
        }
        json_list.append(item)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_list, f, ensure_ascii=False, indent=2)
    print(f"✅ 저장 완료: {save_path}")

# 전체 연도와 시험에 대해 반복 실행
base_dir = f'C:/Users/user/STUBO/{subject}/save_text'
save_dir = f'C:/Users/user/STUBO/{subject}/save_json_temp'
os.makedirs(save_dir, exist_ok=True)

for year in years:
    for exam in exams:
        src = os.path.join(base_dir, f'{year}-{exam}-{subject_initial}.txt')
        dst = os.path.join(save_dir, f'{year}-{exam}-{subject_initial}.json')
        if os.path.exists(src):
            make_json(src, dst)
        else:
            print(f"❌ 파일이 존재하지 않습니다: {src}")