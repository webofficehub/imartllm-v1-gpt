# app.py
import os
import logging
import asyncio
import traceback
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# gradio_client is required in requirements.txt
from gradio_client import Client  # type: ignore

# -------- CONFIG & LOGGING --------
HF_SPACE = os.getenv("HF_SPACE", "weboffice/imartllm-v1-gpt")
HF_API_NAME = os.getenv("HF_API_NAME", "chat")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # secret token stored in Render env
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60.0"))
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "http://localhost:5173").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hf-space-proxy")

# -------- APP --------
app = FastAPI(title="HF Space Proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOW_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- MODELS --------
class ChatRequest(BaseModel):
    message: str
    system_message: Optional[str] = "You are a friendly Chatbot."
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95

class ChatResponse(BaseModel):
    reply: Optional[str] = None
    raw: Optional[dict] = None

# -------- gradio client singleton + diagnostics --------
_gradio_client: Optional[Client] = None
_gradio_client_meta: Dict[str, Any] = {"init_attempted": False, "kwarg_used": None, "error": None}

def get_gradio_client() -> Client:
    """
    Construct Client robustly across gradio_client versions:
    - prefer 'token' kwarg (newer versions), fall back to 'hf_token', else no-token.
    Saves a small diagnostic in _gradio_client_meta.
    """
    global _gradio_client, _gradio_client_meta
    if _gradio_client is not None:
        return _gradio_client

    _gradio_client_meta["init_attempted"] = True
    _gradio_client_meta["kwarg_used"] = None
    _gradio_client_meta["error"] = None

    if HF_API_TOKEN:
        # try most-likely kwarg names
        for kw in ("token", "hf_token"):
            try:
                logger.info("Attempting Client(HF_SPACE, %s=...)", kw)
                _gradio_client = Client(HF_SPACE, **{kw: HF_API_TOKEN})
                _gradio_client_meta["kwarg_used"] = kw
                logger.info("gradio_client created using %s", kw)
                return _gradio_client
            except TypeError as e:
                logger.debug("%s kwarg not supported: %s", kw, e)
            except Exception as e:
                logger.warning("Unexpected error constructing Client with %s: %s", kw, e)
                _gradio_client_meta["error"] = str(e)

    # last resort: no token
    try:
        logger.info("Attempting Client(HF_SPACE) without token")
        _gradio_client = Client(HF_SPACE)
        _gradio_client_meta["kwarg_used"] = "none"
        logger.info("gradio_client created without token")
        return _gradio_client
    except Exception as e:
        logger.exception("Failed to construct gradio_client.Client: %s", e)
        _gradio_client_meta["error"] = traceback.format_exc()
        raise

# -------- HEALTH --------
@app.get("/health")
async def health():
    return {"status": "ok"}

# -------- CHAT endpoint --------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    logger.info("Received /chat request: message_len=%d system_message_len=%d", len(req.message or ""), len(req.system_message or ""))

    try:
        client = get_gradio_client()
    except Exception as e:
        logger.exception("Failed to initialize gradio_client: %s", e)
        raise HTTPException(status_code=500, detail="Server misconfiguration: failed to initialize gradio client")

    # positional args that match the Space inputs
    predict_args = (
        req.message,
        req.system_message or "You are a friendly Chatbot.",
        req.max_tokens or 1024,
        req.temperature or 0.7,
        req.top_p or 0.95,
    )

    # try both api_name variants to be defensive
    api_candidates = [f"/{HF_API_NAME.lstrip('/')}", HF_API_NAME.lstrip('/')]
    last_exc: Optional[Exception] = None
    result: Any = None
    api_used: Optional[str] = None

    for api_name_try in api_candidates:
        try:
            logger.debug("Trying predict with api_name=%s", api_name_try)
            result = await asyncio.wait_for(
                asyncio.to_thread(client.predict, *predict_args, api_name=api_name_try),
                timeout=REQUEST_TIMEOUT,
            )
            api_used = api_name_try
            logger.info("Predict succeeded with api_name=%s; result_type=%s", api_used, type(result).__name__)
            break
        except asyncio.TimeoutError:
            logger.warning("Predict timed out for api_name=%s after %s seconds", api_name_try, REQUEST_TIMEOUT)
            last_exc = asyncio.TimeoutError(f"timeout for api_name={api_name_try}")
        except Exception as e:
            logger.warning("Predict failed for api_name=%s: %s", api_name_try, e)
            last_exc = e

    if result is None:
        logger.exception("All predict attempts failed")
        # give a clean error to the client but include some raw debug details (no token leakage)
        detail = {"error": "upstream-predict-failed", "client_meta": {"kwarg_used": _gradio_client_meta.get("kwarg_used")}}
        if LOG_LEVEL == "DEBUG" and last_exc is not None:
            detail["traceback"] = traceback.format_exc()
        raise HTTPException(status_code=502, detail=detail)

    # Try to extract a simple reply
    reply: Optional[str] = None
    if isinstance(result, (str, int, float)):
        reply = str(result)
    elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], (str, int, float)):
        reply = str(result[0])
    elif isinstance(result, dict):
        for k in ("response", "reply", "data", "output", "result"):
            if k in result and isinstance(result[k], (str, int, float)):
                reply = str(result[k])
                break

    raw_out = {
        "result": result,
        "client_meta": {"kwarg_used": _gradio_client_meta.get("kwarg_used"), "api_used": api_used}
    }

    return ChatResponse(reply=reply, raw=raw_out)

# -------- LOCAL DEV RUNNER --------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level=LOG_LEVEL.lower())
