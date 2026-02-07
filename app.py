# app.py (debug-enhanced)
import os
import logging
import asyncio
import traceback
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# require gradio_client in requirements
from gradio_client import Client  # type: ignore

# -------- CONFIG & LOGGING --------
HF_SPACE = os.getenv("HF_SPACE", "weboffice/imartllm-v1-gpt")
HF_API_NAME = os.getenv("HF_API_NAME", "chat")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # DO NOT print this value
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60.0"))
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "http://localhost:5173").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hf-space-proxy")

# -------- APP --------
app = FastAPI(title="HF Space Proxy (debug)")
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
    Construct Client robustly across versions:
    - try hf_token kwarg
    - try token kwarg
    - try no token
    Store diagnostics in _gradio_client_meta for /debug/gradio to inspect.
    """
    global _gradio_client, _gradio_client_meta
    if _gradio_client is not None:
        return _gradio_client

    _gradio_client_meta["init_attempted"] = True
    _gradio_client_meta["kwarg_used"] = None
    _gradio_client_meta["error"] = None

    # If token is provided, try ways to pass it
    if HF_API_TOKEN:
        try:
            logger.info("Attempting Client(HF_SPACE, hf_token=...)")
            _gradio_client = Client(HF_SPACE, hf_token=HF_API_TOKEN)
            _gradio_client_meta["kwarg_used"] = "hf_token"
            logger.info("gradio_client created using hf_token")
            return _gradio_client
        except TypeError as e:
            logger.debug("hf_token kwarg not supported: %s", e)
        except Exception as e:
            logger.warning("Unexpected error with hf_token attempt: %s", e)
            _gradio_client_meta["error"] = str(e)

        try:
            logger.info("Attempting Client(HF_SPACE, token=...)")
            _gradio_client = Client(HF_SPACE, token=HF_API_TOKEN)
            _gradio_client_meta["kwarg_used"] = "token"
            logger.info("gradio_client created using token")
            return _gradio_client
        except TypeError as e:
            logger.debug("token kwarg not supported: %s", e)
        except Exception as e:
            logger.warning("Unexpected error with token attempt: %s", e)
            _gradio_client_meta["error"] = str(e)

    # Last resort: no token
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

# -------- DEBUG: env --------
@app.get("/debug/env")
async def debug_env():
    """Return basic environment diagnostics (does NOT reveal token)."""
    return {
        "hf_space": HF_SPACE,
        "hf_api_name": HF_API_NAME,
        "hf_api_token_present": bool(HF_API_TOKEN),
        "request_timeout": REQUEST_TIMEOUT,
        "log_level": LOG_LEVEL,
    }

# -------- DEBUG: gradio --------
@app.get("/debug/gradio")
async def debug_gradio():
    """Report installed gradio_client info and client init diagnostics."""
    info: Dict[str, Any] = {"hf_space": HF_SPACE, "hf_api_name": HF_API_NAME}
    try:
        import gradio_client  # type: ignore
        info["gradio_client_version"] = getattr(gradio_client, "__version__", "unknown")
    except Exception as e:
        info["gradio_client_version"] = f"import-failed: {e}"

    # attempt to instantiate client (non-blocking check)
    try:
        # If we already built it, report that metadata
        if _gradio_client is not None:
            info["client_init"] = "already_instantiated"
            info["client_meta"] = _gradio_client_meta
        else:
            # try to construct but do not persist if fails (we reuse get_gradio_client for persistent construction)
            meta = {}
            try:
                c = None
                try:
                    c = Client(HF_SPACE, hf_token=HF_API_TOKEN)
                    meta["attempt"] = "hf_token"
                except TypeError:
                    try:
                        c = Client(HF_SPACE, token=HF_API_TOKEN)
                        meta["attempt"] = "token"
                    except Exception:
                        c = Client(HF_SPACE)
                        meta["attempt"] = "none"
                meta["init_success"] = True
            except Exception as e:
                meta["init_success"] = False
                meta["error"] = traceback.format_exc()
            info["client_init"] = meta
    except Exception as e:
        info["client_init_error"] = traceback.format_exc()

    # Also include our internal diagnostics (if available)
    info["internal_client_meta"] = _gradio_client_meta
    return info

# -------- DEBUG: predict (quick test) --------
@app.post("/debug/predict")
async def debug_predict(req: ChatRequest, request: Request):
    """
    Run a short test predict using the same call as /chat, but with a shorter timeout.
    Useful to reproduce token/auth errors while keeping predict time small.
    WARNING: this calls the upstream Space and may be queued or billed per Space settings.
    """
    try:
        client = get_gradio_client()
    except Exception as e:
        logger.exception("get_gradio_client failed in /debug/predict: %s", e)
        raise HTTPException(status_code=500, detail={"error": "client-init-failed", "trace": traceback.format_exc()})

    api_name_clean = f"/{HF_API_NAME.lstrip('/')}"
    predict_args = (
        req.message,
        req.system_message or "You are a friendly Chatbot.",
        req.max_tokens or 128,
        req.temperature or 0.2,
        req.top_p or 0.9,
    )

    try:
        # use a short timeout for debug predict
        short_timeout = min(15.0, REQUEST_TIMEOUT)
        logger.info("Running debug predict (timeout=%s) api=%s args=(message_len=%d)", short_timeout, api_name_clean, len(predict_args[0]))
        result = await asyncio.wait_for(asyncio.to_thread(client.predict, *predict_args, api_name=api_name_clean), timeout=short_timeout)
        logger.info("Debug predict returned type=%s", type(result).__name__)
        return {"ok": True, "result": result}
    except asyncio.TimeoutError:
        logger.warning("Debug predict timed out after %s seconds", short_timeout)
        raise HTTPException(status_code=504, detail="Debug predict timed out")
    except Exception as e:
        logger.exception("Debug predict failed: %s", e)
        raise HTTPException(status_code=502, detail={"error": str(e), "trace": traceback.format_exc()})

# -------- CHAT endpoint (improved logging & debug info) --------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    logger.info("Received /chat request: message_len=%d system_message_len=%d", len(req.message or ""), len(req.system_message or ""))
    try:
        client = get_gradio_client()
    except Exception as e:
        logger.exception("Could not initialize gradio client: %s", e)
        raise HTTPException(status_code=500, detail="Server misconfiguration: failed to initialize gradio client")

    api_name_clean = f"/{HF_API_NAME.lstrip('/')}"
    predict_args = (
        req.message,
        req.system_message or "You are a friendly Chatbot.",
        req.max_tokens or 1024,
        req.temperature or 0.7,
        req.top_p or 0.95,
    )

    logger.debug("Calling predict api=%s args_preview=%s", api_name_clean, {"message_preview": (req.message[:120] + "...") if len(req.message) > 120 else req.message, "max_tokens": req.max_tokens})

    try:
        result = await asyncio.wait_for(asyncio.to_thread(client.predict, *predict_args, api_name=api_name_clean), timeout=REQUEST_TIMEOUT)
        logger.info("Upstream predict finished; result_type=%s", type(result).__name__)
    except asyncio.TimeoutError:
        logger.warning("Upstream predict timed out after %s seconds", REQUEST_TIMEOUT)
        # include minimal debug info for client
        raise HTTPException(status_code=504, detail="Upstream Space predict timed out (increase REQUEST_TIMEOUT if needed)")
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Error calling gradio_client.predict: %s", e)
        # When debugging, include the exception text in the raw payload if LOG_LEVEL=DEBUG
        raw_debug = {"error": str(e)}
        if LOG_LEVEL == "DEBUG":
            raw_debug["traceback"] = tb
            raw_debug["client_meta"] = _gradio_client_meta
        raise HTTPException(status_code=502, detail=raw_debug)

    # attempt a friendly reply extraction
    reply = None
    if isinstance(result, (str, int, float)):
        reply = str(result)
    elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], (str, int, float)):
        reply = str(result[0])
    elif isinstance(result, dict):
        for k in ("response", "reply", "data", "output", "result"):
            if k in result and isinstance(result[k], (str, int, float)):
                reply = str(result[k])
                break

    # return result with some debug meta
    raw_out = {"result": result, "client_meta": _gradio_client_meta if LOG_LEVEL == "DEBUG" else {"kwarg_used": _gradio_client_meta.get("kwarg_used")}}
    return ChatResponse(reply=reply, raw=raw_out)

# -------- LOCAL DEV RUNNER --------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")
