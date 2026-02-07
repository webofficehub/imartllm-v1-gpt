import os
import logging
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf-space-proxy")

# Config (set these in Render / locally)
HF_SPACE = os.getenv("HF_SPACE", "weboffice/imartllm-v1-gpt")  # owner/repo
HF_API_NAME = os.getenv("HF_API_NAME", "chat")                # api name exposed by the Space
HF_API_TOKEN = os.getenv("HF_API_TOKEN")                      # optional (private spaces)
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60.0"))

# CORS - adjust to your Vite dev URL and production origin
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "http://localhost:5173").split(",")

app = FastAPI(title="HF Space Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOW_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Models
# ---------------------------
class ChatRequest(BaseModel):
    message: str
    system_message: Optional[str] = "You are a friendly Chatbot."
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95

class ChatResponse(BaseModel):
    reply: Optional[str] = None
    raw: Optional[dict] = None

# ---------------------------
# Helpers
# ---------------------------
def space_base_url(space: str) -> str:
    parts = space.split("/")
    if len(parts) != 2:
        raise ValueError("HF_SPACE must be 'owner/repo'")
    owner, repo = parts
    return f"https://{owner}-{repo}.hf.space"

# Re-usable headers for HF Space calls
def hf_headers():
    h = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        h["Authorization"] = f"Bearer {HF_API_TOKEN}"
    return h

# Build gradio-style payload for Spaces
def gradio_payload_for_chat(message: str, system_message: str, max_tokens: int, temperature: float, top_p: float):
    return {"data": [message, system_message, max_tokens, temperature, top_p]}

# ---------------------------
# Health
# ---------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------------------------
# Existing chat endpoint (kept mostly as-is)
# ---------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    base = space_base_url(HF_SPACE)
    api_name_clean = HF_API_NAME.lstrip("/")
    urls_to_try = [
        f"{base}/gradio_api/call/{api_name_clean}",
        f"{base}/run/{api_name_clean}",
    ]

    payload = gradio_payload_for_chat(req.message, req.system_message, req.max_tokens, req.temperature, req.top_p)
    last_exc = None

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for url in urls_to_try:
            try:
                r = await client.post(url, json=payload, headers=hf_headers())
            except httpx.RequestError as e:
                last_exc = e
                logger.warning("RequestError for %s: %s", url, e)
                continue

            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    return ChatResponse(reply=r.text, raw={"status_code": r.status_code, "text": r.text})

                reply = None
                if isinstance(data, dict) and "data" in data:
                    arr = data["data"]
                    if isinstance(arr, list) and len(arr) > 0:
                        candidate = arr[0]
                        if isinstance(candidate, (str, int, float)):
                            reply = str(candidate)
                else:
                    if isinstance(data, (str, int, float)):
                        reply = str(data)

                if reply is None:
                    return ChatResponse(reply=None, raw=data)
                return ChatResponse(reply=reply, raw=data)

            else:
                body = r.text if r.text else "<no body>"
                logger.warning("Space returned %s for %s: %s", r.status_code, url, body)
                last_exc = r

    if isinstance(last_exc, Exception):
        raise HTTPException(status_code=502, detail=str(last_exc))
    raise HTTPException(status_code=502, detail="Failed to call Space API; see server logs")

# ---------------------------
# If run directly (useful for local dev)
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
