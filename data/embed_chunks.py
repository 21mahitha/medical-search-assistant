import json
from sentence_transformers import SentenceTransformer

# load the chunked data from step 2
with open("data/chunked_data.json", "r", encoding="utf-8") as f:
    chunked_data = json.load(f)

print(f"Loaded {len(chunked_data)} chunks")

# load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# extract just the text from every chunk
texts = [chunk["chunk_text"] for chunk in chunked_data]

# embed all of them (this does it in efficient batches, not one by one)
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

print(f"Created {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} numbers")  # should be 384

# attach each embedding back to its corresponding chunk
for i, chunk in enumerate(chunked_data):
    chunk["embedding"] = embeddings[i].tolist()  # convert to plain list so it can be saved as JSON

# save the result
with open("data/embedded_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunked_data, f)

print("Saved to data/embedded_chunks.json")