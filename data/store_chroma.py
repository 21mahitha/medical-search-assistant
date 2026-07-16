import json
import chromadb

# load your embedded chunks
with open("data/embedded_chunks.json", "r", encoding="utf-8") as f:
    embedded_chunks = json.load(f)

print(f"Loaded {len(embedded_chunks)} embedded chunks")

# create a persistent ChromaDB client (saves to disk, not just memory)
client = chromadb.PersistentClient(path="data/chroma_db")

# create (or get) a collection - think of this like a "table" for our vectors
collection = client.get_or_create_collection(name="medquad_chunks")

# prepare data in the format ChromaDB expects
ids = [str(i) for i in range(len(embedded_chunks))]
embeddings = [chunk["embedding"] for chunk in embedded_chunks]
documents = [chunk["chunk_text"] for chunk in embedded_chunks]
metadatas = [
    {
        "focus": chunk["focus"] or "",
        "question": chunk["question"] or "",
        "source": chunk["source"] or "",
        "chunk_index": chunk["chunk_index"] if chunk["chunk_index"] is not None else 0
    }
    for chunk in embedded_chunks
]

print(f"Items already in collection: {collection.count()}")

# add everything to ChromaDB in batches (adding 36k at once can be too much for one call)
batch_size = 5000
for start in range(0, len(ids), batch_size):
    end = start + batch_size
    collection.add(
        ids=ids[start:end],
        embeddings=embeddings[start:end],
        documents=documents[start:end],
        metadatas=metadatas[start:end]
    )
    print(f"Added batch {start} to {end}")

print(f"Total items in collection: {collection.count()}")