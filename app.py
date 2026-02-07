# app.py
import os
import logging
import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# NOTE: gradio_client is required (add to requirements.txt: gradio_client>=0.4.0)
from gradio_client import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf-space-proxy")

# Config (set these in Render / locally)
HF_SPACE = os.getenv("HF_SPACE", "weboffice/imartllm-v1-gpt")  # owner/repo
HF_API_NAME = os.getenv("HF_API_NAME", "chat")                 # api name exposed by the Space (e.g. "/chat")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")                       # optional (private spaces)
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60.0"))

# CORS - adjust to your Vite dev URL and production origin
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "http://localhost:5173").split(",")

app = FastAPI(title="HF Space Proxy (gradio_client)")

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
# Gradio client (lazy singleton)
# ---------------------------
_gradio_client: Optional[Client] = None

def get_gradio_client() -> Client:
    global _gradio_client
    if _gradio_client is None:
        # Pass hf_token if provided (for private Spaces)
        if HF_API_TOKEN:
            logger.info("Creating gradio_client.Client for %s with token", HF_SPACE)
            _gradio_client = Client(HF_SPACE, hf_token=HF_API_TOKEN)
        else:
            logger.info("Creating gradio_client.Client for %s (no token)", HF_SPACE)
            _gradio_client = Client(HF_SPACE)
    return _gradio_client

# ---------------------------
# Health
# ---------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------------------------
# Chat endpoint (uses gradio_client.predict)
# ---------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Calls the target Hugging Face Space using gradio_client.Client.predict
    (this handles queued Jobs and returns the final output rather than an event_id).
    The gradio_client.predict is a blocking call so we run it in a thread via asyncio.to_thread.
    """
    client = get_gradio_client()

    # Prepare positional args in the same order the Space expects
    predict_args = (
        req.message,
        req.system_message or "You are a friendly Chatbot.",
        req.max_tokens or 1024,
        req.temperature or 0.7,
        req.top_p or 0.95,
    )

    try:
        # run blocking predict in a thread
        result = await asyncio.to_thread(
            client.predict,
            *predict_args,
            api_name=f"/{HF_API_NAME.lstrip('/')}",
        )
    except Exception as e:
        logger.exception("Error calling gradio_client.predict: %s", e)
        raise HTTPException(status_code=502, detail=f"Failed to call upstream Space: {e}")

    # result may be str, list, number, dict, etc. Try to extract a friendly 'reply'
    reply = None
    raw_payload = {"result": result}

    if isinstance(result, (str, int, float)):
        reply = str(result)
    elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], (str, int, float)):
        reply = str(result[0])
    else:
        # If result is dict or complex structure, try to find a top-level string field
        if isinstance(result, dict):
            for k in ("response", "reply", "data", "output"):
                if k in result and isinstance(result[k], (str, int, float)):
                    reply = str(result[k])
                    break

    return ChatResponse(reply=reply, raw=raw_payload)

# ---------------------------
# If run directly (useful for local dev)
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
