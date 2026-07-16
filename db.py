import os
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def get_connection():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
            title TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            chat_id TEXT NOT NULL REFERENCES chats(id),
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS upload_chunks (
            id TEXT PRIMARY KEY,
            chat_id TEXT NOT NULL REFERENCES chats(id),
            file_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding vector(384)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id TEXT PRIMARY KEY,
            chat_id TEXT NOT NULL REFERENCES chats(id),
            filename TEXT NOT NULL,
            status TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()


# ---------- USER FUNCTIONS ----------

def get_or_create_user(user_id, email, name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
    existing = cursor.fetchone()

    if not existing:
        cursor.execute(
            "INSERT INTO users (id, email, name, created_at) VALUES (%s, %s, %s, %s)",
            (user_id, email, name, datetime.now().isoformat())
        )
        conn.commit()

    cursor.close()
    conn.close()


# ---------- CHAT FUNCTIONS ----------

def create_chat(chat_id, user_id, title="New Chat"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chats (id, user_id, title, created_at) VALUES (%s, %s, %s, %s)",
        (chat_id, user_id, title, datetime.now().isoformat())
    )
    conn.commit()
    cursor.close()
    conn.close()

def get_user_chats(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, created_at FROM chats WHERE user_id = %s ORDER BY created_at DESC",
        (user_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]

def update_chat_title(chat_id, title):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE chats SET title = %s WHERE id = %s", (title, chat_id))
    conn.commit()
    cursor.close()
    conn.close()

def delete_chat(chat_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE chat_id = %s", (chat_id,))
    cursor.execute("DELETE FROM files WHERE chat_id = %s", (chat_id,))
    cursor.execute("DELETE FROM upload_chunks WHERE chat_id = %s", (chat_id,))  # add this line
    cursor.execute("DELETE FROM chats WHERE id = %s", (chat_id,))
    conn.commit()
    cursor.close()
    conn.close()


# ---------- MESSAGE FUNCTIONS ----------

def save_message(chat_id, role, content):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (chat_id, role, content, timestamp) VALUES (%s, %s, %s, %s)",
        (chat_id, role, content, datetime.now().isoformat())
    )
    conn.commit()
    cursor.close()
    conn.close()

def get_chat_history(chat_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM messages WHERE chat_id = %s ORDER BY id ASC",
        (chat_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]


# ---------- FILE FUNCTIONS ----------

def create_file_record(file_id, chat_id, filename):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO files (id, chat_id, filename, status, created_at) VALUES (%s, %s, %s, %s, %s)",
        (file_id, chat_id, filename, "processing", datetime.now().isoformat())
    )
    conn.commit()
    cursor.close()
    conn.close()

def update_file_status(file_id, status, chunk_count=0):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE files SET status = %s, chunk_count = %s WHERE id = %s",
        (status, chunk_count, file_id)
    )
    conn.commit()
    cursor.close()
    conn.close()

def get_chat_files(chat_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, filename, status, chunk_count FROM files WHERE chat_id = %s ORDER BY created_at ASC",
        (chat_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"id": r[0], "filename": r[1], "status": r[2], "chunk_count": r[3]} for r in rows]


def add_upload_chunks(chat_id, file_id, filename, chunks, embeddings):
    conn = get_connection()
    cursor = conn.cursor()

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{file_id}_{i}"
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        cursor.execute(
            """INSERT INTO upload_chunks (id, chat_id, file_id, filename, chunk_index, chunk_text, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s::vector)""",
            (chunk_id, chat_id, file_id, filename, i, chunk, embedding_str)
        )

    conn.commit()
    cursor.close()
    conn.close()

def query_upload_chunks(chat_id, query_embedding, top_k=5):
    conn = get_connection()
    cursor = conn.cursor()

    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    cursor.execute(
        """SELECT chunk_text FROM upload_chunks
           WHERE chat_id = %s
           ORDER BY embedding <=> %s::vector
           LIMIT %s""",
        (chat_id, embedding_str, top_k)
    )

    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [r[0] for r in rows]

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully")