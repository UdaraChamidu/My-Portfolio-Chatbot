# ingest.py
import os
import json
import numpy as np
import faiss
from pypdf import PdfReader
import dotenv
import google.genai as genai

dotenv.load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "data/cv.pdf")
STORE_DIR = os.getenv("STORE_DIR", "store")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/embedding-001")

os.makedirs(STORE_DIR, exist_ok=True)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        texts.append(txt)
    return "\n".join(texts)

def chunk_text(text: str, chunk_size=700, chunk_overlap=120):
    # simple word-based chunker
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return [c.strip() for c in chunks if c.strip()]

def embed_texts(batch):
    # Google GenAI embeddings — returns a vector per input
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[{"role": "user", "parts": [{"text": t}]} for t in batch],
    )
    # resp.embeddings is a list of objects with .values
    vecs = [np.array(e.values, dtype="float32") for e in resp.embeddings]
    return np.stack(vecs, axis=0)

def main():
    print("Loading PDF:", PDF_PATH)
    text = load_pdf(PDF_PATH)
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    # embed in batches to be safe
    all_vecs = []
    BATCH = 32
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i+BATCH]
        vecs = embed_texts(batch)
        all_vecs.append(vecs)
        print(f"Embedded {i+len(batch)}/{len(chunks)}")

    vecs = np.vstack(all_vecs)
    dim = vecs.shape[1]
    print("Embedding dim:", dim)

    # build FAISS index
    index = faiss.IndexFlatIP(dim)  # cosine-like with normalized vectors
    # normalize vectors
    faiss.normalize_L2(vecs)
    index.add(vecs)

    # save index & chunks
    faiss.write_index(index, os.path.join(STORE_DIR, "faiss.index"))
    with open(os.path.join(STORE_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    meta = {"dim": dim, "model": EMBED_MODEL}
    with open(os.path.join(STORE_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print("Ingestion complete. Files saved to:", STORE_DIR)

if __name__ == "__main__":
    main()
