import json

with open("data/qa_pairs.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

# check answer lengths (in words) across the dataset
lengths = [len(qa["answer"].split()) for qa in qa_pairs]

print(f"Total QA pairs: {len(qa_pairs)}")
print(f"Shortest answer: {min(lengths)} words")
print(f"Longest answer: {max(lengths)} words")
print(f"Average answer length: {sum(lengths) / len(lengths):.0f} words")

import json

with open("data/qa_pairs.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

def chunk_text(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward, but re-include the overlap

    return chunks

chunked_data = []

for qa in qa_pairs:
    answer = qa["answer"]

    # skip junk / near-empty answers
    if len(answer.split()) < 5:
        continue
    
    text_chunks = chunk_text(answer)

    for i, chunk in enumerate(text_chunks):
        chunked_data.append({
            "focus": qa["focus"],
            "question": qa["question"],
            "chunk_text": chunk,
            "chunk_index": i,
            "source": qa["source"]
        })

print(f"Total chunks created: {len(chunked_data)}")
print(chunked_data[0])

with open("data/chunked_data.json", "w", encoding="utf-8") as f:
    json.dump(chunked_data, f, indent=2)

print("Saved to data/chunked_data.json")