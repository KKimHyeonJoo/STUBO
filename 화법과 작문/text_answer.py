import pdfplumber
import re
import os

years = [2021, 2022, 2023, 2024]
exams = ['06', '09', '수능']
subject = '화작'

base_pdf_dir = '/Users/chaewon/Desktop/STUBO/화법과 작문/해설'
base_txt_dir = '/Users/chaewon/Desktop/STUBO/화법과 작문/answer_text'

pattern = re.compile(
    r'(3[5-9]|4[0-5])\..*?정답해설\s*:\s*(.*?)(정답\s*[①-⑤1-5])',
    re.DOTALL
)

def extract_subject_section(text, subject_name, next_subject_name=None):
    """
    text 내에서 subject_name으로 시작하는 구간을 찾아 자르고,
    next_subject_name이 있으면 그 구간 바로 전까지 자름.
    """
    start_idx = text.find(subject_name)
    if start_idx == -1:
        return ""

    if next_subject_name:
        end_idx = text.find(next_subject_name, start_idx)
        if end_idx == -1:
            return text[start_idx:]
        else:
            return text[start_idx:end_idx]
    else:
        return text[start_idx:]

for year in years:
    for exam in exams:
        pdf_path = os.path.join(base_pdf_dir, f'{year}-{exam}-{subject} 해설.pdf')
        save_path = os.path.join(base_txt_dir, f'{year}-{exam}-{subject} 해설.txt')

        if not os.path.exists(pdf_path):
            print(f"파일이 없습니다: {pdf_path}")
            continue

        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

        # 1) 화작 섹션만 추출 (화작 다음 과목이 '언어와 매체')
        subject_text = extract_subject_section(full_text, '[선택: 화법과 작문]', '[선택: 언어와 매체]')

        results = []
        for match in pattern.finditer(subject_text):
            num = match.group(1)
            explanation = match.group(2).strip().replace('\n', ' ')
            answer_raw = match.group(3)
            answer_match = re.search(r'[①-⑤1-5]', answer_raw)
            if answer_match:
                answer = answer_match.group(0)
                if '①' <= answer <= '⑤':
                    answer_num = str(ord(answer) - ord('①') + 1)
                else:
                    answer_num = answer
            else:
                answer_num = ""
            results.append(f"문제번호: {num} \n정답: {answer_num}\n해설: {explanation}\n")

        with open(save_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(r + "\n")

        print(f"저장 완료: {save_path}")
