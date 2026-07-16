import chromadb
from sentence_transformers import SentenceTransformer

# load the embedding model (same one used to build the database)
model = SentenceTransformer("all-MiniLM-L6-v2")

# connect to the existing ChromaDB database
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection(name="medquad_chunks")

def retrieve(query, top_k=5):
    # embed the user's question
    query_embedding = model.encode([query]).tolist()

    # search ChromaDB for the closest matches
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return results

# quick test
if __name__ == "__main__":
    query = "what causes migraines"
    results = retrieve(query)

    for i in range(len(results["documents"][0])):
        print(f"\n--- Result {i+1} ---")
        print("Focus:", results["metadatas"][0][i]["focus"])
        print("Source:", results["metadatas"][0][i]["source"])
        print("Text:", results["documents"][0][i][:200], "...")  # first 200 chars only