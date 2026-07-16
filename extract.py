from pypdf import PdfReader
from docx import Document

def extract_text(file_path, filename):
    if filename.lower().endswith(".pdf"):
        return extract_pdf(file_path)
    elif filename.lower().endswith(".docx"):
        return extract_docx(file_path)
    elif filename.lower().endswith(".txt"):
        return extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {filename}")

def extract_pdf(file_path):
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)

def extract_docx(file_path):
    doc = Document(file_path)
    text_parts = [para.text for para in doc.paragraphs]
    return "\n".join(text_parts)

def extract_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

if __name__ == "__main__":
    # quick manual test - place a test file in your project folder and update the path/name below
    text = extract_text("test.pdf", "test.pdf")
    print(f"Extracted {len(text)} characters")
    print(text[:500])