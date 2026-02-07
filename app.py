# app.py
import os
import logging
import asyncio
from typing import Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# require gradio_client in requirements (version can vary; this code is compatible with both hf_token and token names)
from gradio_client import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf-space-proxy")

# Config (set these in Render / locally)
HF_SPACE = os.getenv("HF_SPACE", "weboffice/imartllm-v1-gpt")  # owner/repo
HF_API_NAME = os.getenv("HF_API_NAME", "chat")                 # api name exposed by the Space (e.g. "chat" or "/chat")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")                       # optional (private spaces)
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60.0"))

# CORS - adjust to your frontend origin(s)
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "http://localhost:5173").split(",")

app = FastAPI(title="HF Space Proxy (gradio_client robust)")

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
# Gradio client (lazy singleton) with compatibility for different gradio_client versions
# ---------------------------
_gradio_client: Optional[Client] = None

def get_gradio_client() -> Client:
    """
    Attempt to construct Client with the appropriate token kwarg depending on gradio_client version.
    Tries hf_token, then token, then no token. Logs helpful info.
    """
    global _gradio_client
    if _gradio_client is not None:
        return _gradio_client

    # Try hf_token first (older client versions), fallback to token (newer), else no-token
    if HF_API_TOKEN:
        try:
            logger.info("Creating gradio_client.Client for %s using hf_token", HF_SPACE)
            _gradio_client = Client(HF_SPACE, hf_token=HF_API_TOKEN)
            return _gradio_client
        except TypeError:
            logger.info("hf_token not supported by installed gradio_client; trying 'token' kwarg")
        except Exception as e:
            logger.warning("Unexpected error constructing Client with hf_token: %s", e)

        try:
            logger.info("Creating gradio_client.Client for %s using token", HF_SPACE)
            _gradio_client = Client(HF_SPACE, token=HF_API_TOKEN)
            return _gradio_client
        except TypeError:
            logger.warning("token kwarg not supported either; falling back to no-token Client (may fail for private Spaces)")
        except Exception as e:
            logger.warning("Unexpected error constructing Client with token: %s", e)

    # Last resort: instantiate without token
    try:
        logger.info("Creating gradio_client.Client for %s without token", HF_SPACE)
        _gradio_client = Client(HF_SPACE)
        return _gradio_client
    except Exception as e:
        logger.exception("Failed to construct gradio_client.Client: %s", e)
        raise

# ---------------------------
# Health
# ---------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------------------------
# Chat endpoint
# ---------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Uses gradio_client.Client.predict to call the HF Space API and waits (with timeout) for the final reply.
    Runs the blocking predict in a thread via asyncio.to_thread and enforces REQUEST_TIMEOUT.
    """
    client = None
    try:
        client = get_gradio_client()
    except Exception as e:
        logger.exception("Could not initialize gradio client: %s", e)
        raise HTTPException(status_code=500, detail="Server misconfiguration: failed to initialize gradio client")

    # prepare positional args in same order the Space expects
    predict_args = (
        req.message,
        req.system_message or "You are a friendly Chatbot.",
        req.max_tokens or 1024,
        req.temperature or 0.7,
        req.top_p or 0.95,
    )

    api_name_clean = f"/{HF_API_NAME.lstrip('/')}"

    try:
        # run blocking predict in a thread and enforce a timeout so requests don't hang indefinitely
        result = await asyncio.wait_for(
            asyncio.to_thread(client.predict, *predict_args, api_name=api_name_clean),
            timeout=REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Upstream predict timed out after %s seconds", REQUEST_TIMEOUT)
        raise HTTPException(status_code=504, detail="Upstream Space predict timed out")
    except Exception as e:
        logger.exception("Error calling gradio_client.predict: %s", e)
        raise HTTPException(status_code=502, detail=f"Failed to call upstream Space: {e}")

    # result can be many shapes; try to extract simple reply
    reply = None
    if isinstance(result, (str, int, float)):
        reply = str(result)
    elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], (str, int, float)):
        reply = str(result[0])
    elif isinstance(result, dict):
        # try common keys
        for k in ("response", "reply", "data", "output", "result"):
            if k in result and isinstance(result[k], (str, int, float)):
                reply = str(result[k])
                break

    return ChatResponse(reply=reply, raw={"result": result})

# ---------------------------
# Local dev runner
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
