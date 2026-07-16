import uuid
from sentence_transformers import SentenceTransformer
from extract import extract_text
from db import create_file_record, update_file_status, add_upload_chunks

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def process_uploaded_file(file_path, filename, chat_id):
    file_id = str(uuid.uuid4())
    create_file_record(file_id, chat_id, filename)

    try:
        text = extract_text(file_path, filename)

        if not text or len(text.strip()) < 20:
            update_file_status(file_id, "failed", 0)
            return {"file_id": file_id, "status": "failed", "reason": "No readable text found"}

        chunks = chunk_text(text)
        embeddings = model.encode(chunks).tolist()

        add_upload_chunks(chat_id, file_id, filename, chunks, embeddings)

        update_file_status(file_id, "ready", len(chunks))
        return {"file_id": file_id, "status": "ready", "chunk_count": len(chunks)}

    except Exception as e:
        update_file_status(file_id, "failed", 0)
        return {"file_id": file_id, "status": "failed", "reason": str(e)}

if __name__ == "__main__":
    result = process_uploaded_file("test.pdf", "test.pdf", "test_chat_id")
    print(result)