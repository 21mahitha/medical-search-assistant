# Medical Search Assistant

A RAG-powered medical Q&A chatbot — a "medical Perplexity" that answers questions grounded in trusted NIH medical sources (MedQuAD), lets users upload their own documents to query, and keeps per-user, per-chat conversation history.

Built as part of an internship project exploring RAG pipeline design end-to-end.

---

## Features

- **Grounded medical Q&A** — answers sourced from MedQuAD (16,407 Q&A pairs across 12 NIH sources: MedlinePlus, NIDDK, GARD, NINDS, and more)
- **Document upload** — upload a PDF/DOCX/TXT and ask questions about it directly; the chat prioritizes the document, falls back to MedQuAD, then to general knowledge — and says which one it used
- **Conversation memory** — follow-up questions ("what about treatment?") are understood in context of the current chat
- **Google login** — each user has their own separate, persistent chat history
- **Multiple chats per user** — new chat, chat switching, auto-generated titles, delete chats
- **Streaming responses** — answers appear word-by-word

---

## Tech stack

| Layer | Tool |
|---|---|
| Backend | Flask |
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector search — MedQuAD | ChromaDB |
| Vector search — uploaded files | Postgres + pgvector (Supabase) |
| Relational data | Postgres (Supabase) |
| Auth | Google OAuth (Authlib) |
| Frontend | HTML/CSS/vanilla JS |

---

## Project structure

```
chatbot/
├── app.py                  # Flask routes: auth, chats, ask, file upload
├── auth.py                 # Google OAuth setup
├── db.py                   # All Postgres (Supabase) reads/writes
├── retrieve.py              # MedQuAD ChromaDB retrieval
├── generate.py              # Query reformulation, retrieval routing, Groq streaming
├── embedding_model.py       # Shared SentenceTransformer instance (loaded once)
├── extract.py               # Text extraction (PDF/DOCX/TXT)
├── process_upload.py        # Chunk + embed + store uploaded files
├── templates/
│   ├── index.html           # Main chat UI
│   └── login.html           # Login page
├── data/
│   ├── parse_medquad.py     # One-time: MedQuAD XML → qa_pairs.json
│   ├── chunk_data.py        # One-time: qa_pairs.json → chunked_data.json
│   ├── embed_chunks.py      # One-time: chunked_data.json → embedded_chunks.json
│   ├── store_chroma.py      # One-time: embedded_chunks.json → ChromaDB
│   └── chroma_db/           # MedQuAD's ChromaDB collection (baked into app)
├── requirements.txt
├── Dockerfile
└── .env                     # Local secrets (never committed)
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/21mahitha/medical-search-assistant.git
cd medical-search-assistant
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key
GOOGLE_CLIENT_ID=your_google_oauth_client_id
GOOGLE_CLIENT_SECRET=your_google_oauth_client_secret
FLASK_SECRET_KEY=generate_with_python_secrets_token_hex_32
DATABASE_URL=your_supabase_postgres_session_pooler_url
```

- **Groq**: https://console.groq.com/keys
- **Google OAuth**: https://console.cloud.google.com/ → APIs & Services → Credentials → OAuth Client ID (Web application). Add `http://127.0.0.1:5000/auth/callback` as an authorized redirect URI.
- **Supabase**: https://supabase.com → new project → enable the `vector` extension (Database → Extensions) → grab the **session pooler** connection string (Connect → Direct → Session pooler) — the direct connection string is IPv6-only and will hang on most networks.
- **Flask secret key**: `python -c "import secrets; print(secrets.token_hex(32))"`

### 3. Build the MedQuAD knowledge base (first-time only)

This only needs to run once — it produces `data/chroma_db`, which then ships with the app.

```bash
git clone https://github.com/abachaa/MedQuAD.git data/MedQuAD
python data/parse_medquad.py
python data/chunk_data.py
python data/embed_chunks.py
python data/store_chroma.py
```

### 4. Initialize the database

```bash
python db.py
```

This creates the `users`, `chats`, `messages`, `files`, and `upload_chunks` tables in Supabase.

### 5. Run the app

```bash
python app.py
```

Visit `http://127.0.0.1:5000`.

---

## How it works (short version)

1. **MedQuAD** gets parsed, chunked (~150 words per chunk), embedded, and stored in ChromaDB — once, offline.
2. **Login**: Google confirms identity → Flask saves the user to Supabase → a signed session cookie tracks who's logged in on every request.
3. **Asking a question**: the query is reformulated using the last 6 messages in the current chat → retrieval tries the chat's uploaded document first, then MedQuAD, then falls back to general knowledge → Groq streams a grounded answer → both messages get saved to Postgres under this chat's ID.
4. **Uploading a file**: text is extracted, chunked, embedded, and stored in Postgres (via pgvector), tagged with the chat's ID so it's only ever searched within that specific chat.

For the full detailed walkthrough, see `medical_search_assistant_notes.md`.

---

## Known limitations / not yet done

- Not currently deployed (see notes on Render free-tier memory limits and Gemini embedding rate limits encountered during the deployment attempt)
- Citations from retrieved sources aren't shown in the UI (removed for simplicity; sources are tracked internally but not surfaced)
- No automated tests

---

## Credits

- Data: [MedQuAD](https://github.com/abachaa/MedQuAD) — Ben Abacha, A., Demner-Fushman, D.
