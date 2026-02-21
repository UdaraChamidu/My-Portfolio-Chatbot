# main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json
import numpy as np
import faiss
import dotenv
import google.genai as genai
from typing import Optional  # <-- use Optional for 3.9

dotenv.load_dotenv()

app = FastAPI()

# --- CORS ---
def normalize_origin(value: str) -> str:
    # Normalize small formatting mistakes from env vars (quotes, trailing slash).
    return value.strip().strip("'").strip('"').rstrip("/")

default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://udara-chamidu-portfolio.vercel.app",
    "https://udarachamidu.site",
    "https://www.udarachamidu.site",
]
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
origins_from_env = [
    normalize_origin(o) for o in allowed_origins_env.split(",") if normalize_origin(o)
]
origins = list(dict.fromkeys(origins_from_env or default_origins))
origin_regex = os.getenv("ALLOWED_ORIGIN_REGEX", "").strip() or None
print("CORS allow_origins:", origins)
if origin_regex:
    print("CORS allow_origin_regex:", origin_regex)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini client ---
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embedding-001")

# --- Vector store ---
STORE_DIR = os.getenv("STORE_DIR", "store")
INDEX_PATH = os.path.join(STORE_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(STORE_DIR, "chunks.json")
META_PATH = os.path.join(STORE_DIR, "meta.json")

index = None
chunks = []
meta = {}

def load_store():
    global index, chunks, meta
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
        raise RuntimeError("Vector store not found. Run ingest.py first.")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks[:] = json.load(f)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta.update(json.load(f))
    print("Loaded vector store:", len(chunks), "chunks")

load_store()

# --- In-memory chat history ---
SESSION_HISTORY = {}
MAX_TURNS = 10  # keep last N user-bot turns

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # <-- fixed for Python 3.9

def embed_query(text: str) -> np.ndarray:
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[{"role": "user", "parts": [{"text": text}]}],
    )
    vec = np.array(resp.embeddings[0].values, dtype="float32")
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

def retrieve_context(query: str, top_k=5):
    if index is None:
        return []
    q = embed_query(query)
    D, I = index.search(q.reshape(1, -1), top_k)
    ctx = []
    for idx in I[0]:
        if 0 <= idx < len(chunks):
            ctx.append(chunks[idx])
    return ctx

SYSTEM_PROMPT = (
    "You are Udara Herath’s personal portfolio assistant.\n"
    "- behave politely and in a user friendly way.\n"
    "- use points, paragraphs and other ways to answers.\n"
    "- if you have a history, check it also. sometimes it will be needed to answer.\n"
    "- most of times user query based on a person called udara. when user asks details about udara, no need to provide all the details. according to the user query you can decide how much things user needed. most of times users need data briefly.\n"
    "- if user ask general question that is not related to me, you can answer freely.\n"
    "- for a user asked question, if the provided document context is not enough, you can add more things to it. but be sure to indicate what is from the context and what is not."
    "- do not give too long answers. answer briefly.\n"
)


def build_prompt(context_chunks: list, history: list, user_msg: str) -> str:  # <-- 3.9 compatible
    context_text = "\n\n---\n".join(context_chunks) if context_chunks else "No extra context."
    hist_text = ""
    for turn in history[-MAX_TURNS:]:
        role = turn["from"]
        text = turn["text"]
        hist_text += f"{role.upper()}: {text}\n"
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"CHAT HISTORY:\n{hist_text}\n"
        f"USER: {user_msg}\n\n"
        f"ASSISTANT:"
    )
    return prompt

@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    origin = request.headers.get("origin")
    try:
        session_id = req.session_id or "default"
        history = SESSION_HISTORY.setdefault(session_id, [])

        context_chunks = retrieve_context(req.message, top_k=5)
        prompt = build_prompt(context_chunks, history, req.message)

        resp = client.models.generate_content(
            model=GEN_MODEL,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )

        answer = resp.text.strip() if hasattr(resp, "text") else "Sorry, I couldn't generate a response."

        history.append({"from": "user", "text": req.message})
        history.append({"from": "bot", "text": answer})

        if len(history) > 2 * MAX_TURNS:
            SESSION_HISTORY[session_id] = history[-2 * MAX_TURNS :]

        return {
            "response": answer,
            "session_id": session_id,
            "context_used": context_chunks,
        }
    except Exception as e:
        print(f"/api/chat failed. origin={origin} error={repr(e)}")
        raise HTTPException(status_code=500, detail="Internal chatbot error")

@app.get("/api/reset")
def reset(session_id: Optional[str] = None):
    sid = session_id or "default"
    SESSION_HISTORY.pop(sid, None)
    return {"ok": True}

@app.get("/api/health")
def health():
    return {"ok": True}








