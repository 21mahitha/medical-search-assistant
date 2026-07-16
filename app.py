from flask import Flask, request, jsonify, render_template, session, Response, redirect, url_for
from generate import generate_answer_stream
from db import init_db, get_or_create_user, get_user_chats, create_chat, get_chat_history, update_chat_title, delete_chat
from auth import oauth, init_oauth
import uuid
from dotenv import load_dotenv
import os
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-later")

init_db()
init_oauth(app)

from process_upload import process_uploaded_file
from db import get_chat_files
import os

DATA_DIR = os.getenv("DATA_DIR", "data")
UPLOAD_FOLDER = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    if "user" not in session:
        return render_template("login.html")
    return render_template("index.html", user=session["user"])

@app.route("/login")
def login():
    redirect_uri = url_for("auth_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route("/auth/callback")
def auth_callback():
    token = oauth.google.authorize_access_token()
    user_info = token.get("userinfo")

    user_id = user_info["sub"]
    email = user_info["email"]
    name = user_info.get("name", email)

    get_or_create_user(user_id, email, name)
    session["user"] = {"id": user_id, "email": email, "name": name}

    return redirect(url_for("home"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/chats", methods=["GET"])
def list_chats():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    user_id = session["user"]["id"]
    chats = get_user_chats(user_id)
    return jsonify({"chats": chats})

@app.route("/chats/new", methods=["POST"])
def new_chat():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    user_id = session["user"]["id"]
    chat_id = str(uuid.uuid4())
    create_chat(chat_id, user_id, title="New Chat")

    return jsonify({"chat_id": chat_id})

@app.route("/chats/<chat_id>/messages", methods=["GET"])
def chat_messages(chat_id):
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    history = get_chat_history(chat_id)
    return jsonify({"messages": history})

@app.route("/chats/<chat_id>", methods=["DELETE"])
def remove_chat(chat_id):
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    delete_chat(chat_id)
    return jsonify({"success": True})

@app.route("/ask", methods=["POST"])
def ask():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    query = data.get("question", "")

    if not query:
        return jsonify({"error": "No question provided"}), 400

    chat_id = data.get("chat_id")

    return Response(generate_answer_stream(query, chat_id), mimetype="text/plain")


@app.route("/chats/<chat_id>/upload", methods=["POST"])
def upload_file(chat_id):
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed_extensions = (".pdf", ".docx", ".txt")
    if not file.filename.lower().endswith(allowed_extensions):
        return jsonify({"error": "Unsupported file type. Use PDF, DOCX, or TXT"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{file.filename}")
    file.save(save_path)

    result = process_uploaded_file(save_path, file.filename, chat_id)

    return jsonify(result)

@app.route("/chats/<chat_id>/files", methods=["GET"])
def list_chat_files(chat_id):
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    files = get_chat_files(chat_id)
    return jsonify({"files": files})

if __name__ == "__main__":
    app.run(debug=True)