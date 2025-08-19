# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json
import numpy as np
import faiss
import dotenv
import google.genai as genai

dotenv.load_dotenv()

app = FastAPI()

# --- CORS ---
origins = [
    "http://localhost:5173",
    "https://udara-chamidu-portfolio.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # or ["*"] while testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini client ---
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

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

# --- In-memory chat history (per session) ---
# For production, move this to Redis/DB.
SESSION_HISTORY = {}
MAX_TURNS = 10  # keep last N user-bot turns

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # frontend can send a stable id

def embed_query(text: str) -> np.ndarray:
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[{"role": "user", "parts": [{"text": text}]}],
    )
    vec = np.array(resp.embeddings[0].values, dtype="float32")
    # normalize for IP similarity
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
    "- Use the provided context (retrieved from Udara’s documents) to answer.\n"
    "- If the answer is not in the context, say you don’t know or something like that in a friendly way.\n"
    "- if user ask general question that is not related to me, you can answer freely."
    "- Be concise, professional, and friendly, also do greetings when need.\n"
    "- Prefer bullet points for lists. Include links only if provided in context.\n"
    "- if you have a history, check it also. somethimes it will be needed to answer."
    "- for a user asked question, if the provided document context is not enough, you can add more things to it. but be sure to indicate what is from the context and what is not."
)
def build_prompt(context_chunks: list[str], history: list[dict], user_msg: str) -> str:
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
async def chat(req: ChatRequest):
    session_id = req.session_id or "default"
    history = SESSION_HISTORY.setdefault(session_id, [])

    # 1) Retrieve relevant chunks
    context_chunks = retrieve_context(req.message, top_k=5)

    # 2) Build prompt with context + history
    prompt = build_prompt(context_chunks, history, req.message)

    # 3) Generate answer with Gemini
    resp = client.models.generate_content(
        model=GEN_MODEL,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
    )

    answer = resp.text.strip() if hasattr(resp, "text") else "Sorry, I couldn't generate a response."

    # 4) Update history
    history.append({"from": "user", "text": req.message})
    history.append({"from": "bot", "text": answer})

    # Keep only the last MAX_TURNS
    if len(history) > 2 * MAX_TURNS:
        SESSION_HISTORY[session_id] = history[-2 * MAX_TURNS :]

    return {
        "response": answer,                 # ✅ fixed
        "session_id": session_id,
        "context_used": context_chunks,     # ✅ fixed
    }

@app.get("/api/reset")
def reset(session_id: str | None = None):
    sid = session_id or "default"
    SESSION_HISTORY.pop(sid, None)
    return {"ok": True}

@app.get("/api/health")
def health():
    return {"ok": True}
