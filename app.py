# app.py
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

# Request body from frontend
class ChatRequest(BaseModel):
    message: str
    system_message: Optional[str] = "You are a friendly Chatbot."
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95

class ChatResponse(BaseModel):
    reply: Optional[str] = None
    raw: Optional[dict] = None

def space_base_url(space: str) -> str:
    """
    Convert "owner/repo" to "<owner>-<repo>.hf.space" and return base.
    """
    parts = space.split("/")
    if len(parts) != 2:
        raise ValueError("HF_SPACE must be 'owner/repo'")
    owner, repo = parts
    return f"https://{owner}-{repo}.hf.space"

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    base = space_base_url(HF_SPACE)

    # prefer the documented gradio API path; fallback attempt to /run/<api>
    api_name_clean = HF_API_NAME.lstrip("/")
    urls_to_try = [
        f"{base}/gradio_api/call/{api_name_clean}",     # typical Gradio “call” endpoint for Spaces
        f"{base}/run/{api_name_clean}",                 # alternate used by some clients
    ]

    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    # Gradio expects inputs as an array "data": [ ... ] (order matches function parameters).
    # Based on the example you pasted, parameters are: message, system_message, max_tokens, temperature, top_p
    payload = {"data": [req.message, req.system_message, req.max_tokens, req.temperature, req.top_p]}

    last_exc = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        for url in urls_to_try:
            try:
                r = await client.post(url, json=payload, headers=headers)
            except httpx.RequestError as e:
                last_exc = e
                logger.warning("RequestError for %s: %s", url, e)
                continue

            # 200 OK -> try to parse returned structure
            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    # fallback to text
                    return ChatResponse(reply=r.text, raw={"status_code": r.status_code, "text": r.text})

                # Spaces/Gradio call endpoints usually return {"data": [...], ...}
                reply = None
                if isinstance(data, dict) and "data" in data:
                    arr = data["data"]
                    # the first element is typically the actual returned value
                    if isinstance(arr, list) and len(arr) > 0:
                        # Normalize various return shapes:
                        # - simple string
                        # - dict/list returned by your Space
                        candidate = arr[0]
                        if isinstance(candidate, (str, int, float)):
                            reply = str(candidate)
                        else:
                            # if it's a dict/list, stringify it and also include in raw
                            reply = None
                else:
                    # Sometimes Spaces return a raw value (not wrapped)
                    if isinstance(data, (str, int, float)):
                        reply = str(data)

                # If we didn't manage to extract a single string reply, put the whole JSON as raw
                if reply is None:
                    return ChatResponse(reply=None, raw=data)
                return ChatResponse(reply=reply, raw=data)

            else:
                # non-200 -> try next URL (maybe different endpoint)
                logger.warning("Space returned %s for %s: %s", r.status_code, url, (await r.aread()).decode(errors="ignore") if hasattr(r, "aread") else r.text)
                last_exc = r

    # If we get here nothing worked
    if isinstance(last_exc, Exception):
        raise HTTPException(status_code=502, detail=str(last_exc))
    raise HTTPException(status_code=502, detail="Failed to call Space API; see server logs")
