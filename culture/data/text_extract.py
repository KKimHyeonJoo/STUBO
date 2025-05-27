from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

pdf_path = "text.pdf"
raw_text = extract_text_from_pdf(pdf_path)

with open("text.txt", "w", encoding="utf-8") as f:
    f.write(raw_text)
