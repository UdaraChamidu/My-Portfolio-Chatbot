# Portfolio Chatbot Backend

FastAPI backend for the portfolio chatbot.

## Endpoints

- `POST /api/chat`
- `GET /api/reset?session_id=<id>`
- `GET /api/health`

`POST /api/chat` request body:

```json
{
  "message": "Tell me about Udara",
  "session_id": "user-123"
}
```

## Environment Variables

- `GEMINI_API_KEY` (required)
- `GEN_MODEL` (optional, default: `gemini-2.5-flash`)
- `EMBED_MODEL` (optional, default: `embedding-001`)
- `STORE_DIR` (optional, default: `store`)
- `ALLOWED_ORIGINS` (optional, comma-separated list)
  - Example: `http://localhost:5173,https://udara-chamidu-portfolio.vercel.app,https://udarachamidu.site,https://www.udarachamidu.site`
  - Do not wrap the value in quotes.
  - Do not add trailing `/` to each origin.
- `ALLOWED_ORIGIN_REGEX` (optional)
  - Example: `^https://([a-z0-9-]+\\.)?udarachamidu\\.site$`

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## Production Frontend Connection

1. Deploy this backend (Railway/Render/Fly/other).
2. Set frontend env var to backend URL, for example:
   - `VITE_CHATBOT_API_BASE=https://your-backend-domain.com`
3. Call the backend from your frontend:

```js
const API_BASE = import.meta.env.VITE_CHATBOT_API_BASE;

const res = await fetch(`${API_BASE}/api/chat`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message, session_id }),
});
```

4. On backend, set `ALLOWED_ORIGINS` to your frontend domain(s).
5. If you use custom domains with and without `www`, include both or use `ALLOWED_ORIGIN_REGEX`.
 
