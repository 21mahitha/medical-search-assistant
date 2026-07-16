import os
from dotenv import load_dotenv
from groq import Groq
from retrieve import retrieve
from db import save_message, get_chat_history, update_chat_title, get_chat_files
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.3-70b-versatile"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="data/chroma_db")

RELEVANCE_THRESHOLD = 1.0  # lower distance = more relevant; tune this based on testing

def reformulate_query(query, history):
    if not history:
        return query

    history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history[-6:]])

    prompt = f"""Given this conversation history and a follow-up question, rewrite the follow-up 
into a standalone question that includes any necessary context. If the follow-up is already 
standalone, return it unchanged. Return ONLY the rewritten question, nothing else.

Conversation history:
{history_text}

Follow-up question: {query}

Standalone question:"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_chat_title(query):
    prompt = f"""Generate a short, concise title (3-6 words) summarizing this medical question. 
Return ONLY the title, no quotes, no punctuation at the end.

Question: {query}

Title:"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def has_ready_files(chat_id):
    files = get_chat_files(chat_id)
    return any(f["status"] == "ready" for f in files)

from db import query_upload_chunks

def retrieve_from_uploads(query, chat_id, top_k=5):
    query_embedding = embed_model.encode([query]).tolist()[0]
    return query_upload_chunks(chat_id, query_embedding, top_k=top_k)

def retrieve_from_medquad(query, top_k=5):
    results = retrieve(query, top_k=top_k)

    if not results["documents"][0]:
        return []

    if "distances" in results and results["distances"][0]:
        relevant_chunks = [
            doc for doc, dist in zip(results["documents"][0], results["distances"][0])
            if dist < RELEVANCE_THRESHOLD
        ]
        return relevant_chunks

    return results["documents"][0]

def generate_answer_stream(query, chat_id, top_k=5):
    history = get_chat_history(chat_id)

    if not history:
        title = generate_chat_title(query)
        update_chat_title(chat_id, title)

    standalone_query = reformulate_query(query, history)

    # --- ROUTING: document -> medquad -> general knowledge ---
    source = None
    context_chunks = []

    if has_ready_files(chat_id):
        context_chunks = retrieve_from_uploads(standalone_query, chat_id, top_k=top_k)
        if context_chunks:
            source = "document"

    if source is None and not has_ready_files(chat_id):
        context_chunks = retrieve_from_medquad(standalone_query, top_k=top_k)
        if context_chunks:
            source = "medquad"

    if source is None:
        source = "general"

    history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history[-6:]])

    if source == "document":
        context = "\n\n".join(context_chunks)
        instructions = """Answer the user's question using ONLY the information in the sources below, 
which come from a document the user uploaded. Do not mention that you were given "sources" - just answer naturally."""
    elif source == "medquad":
        context = "\n\n".join(context_chunks)
        instructions = """Answer the user's question using ONLY the information in the sources below, 
which come from trusted NIH medical sources. Do not mention that you were given "sources" - just answer naturally."""
    else:
        context = "(no relevant sources found)"
        instructions = """No relevant information was found in the uploaded document or the medical knowledge base. 
Answer using your own general knowledge instead. Start your answer by clearly stating: 
"I couldn't find this in your uploaded document or my medical sources, so here's a general answer:" 
then proceed with your best answer."""

    prompt = f"""You are a medical information assistant for medical students.
{instructions}
Keep the conversation natural and consistent with prior messages.

Conversation so far:
{history_text}

Sources:
{context}

Question: {query}

Answer:"""

    full_answer = ""
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_answer += delta
            yield delta

    save_message(chat_id, "user", query)
    save_message(chat_id, "assistant", full_answer)

if __name__ == "__main__":
    import uuid
    from db import init_db, get_or_create_user, create_chat

    init_db()
    get_or_create_user("test_user2", "test2@test.com", "Test2")
    chat_id = str(uuid.uuid4())
    create_chat(chat_id, "test_user2", title="New Chat")

    q1 = "what causes migraines"
    print("Q1:", q1)
    for piece in generate_answer_stream(q1, chat_id):
        print(piece, end="", flush=True)
    print()