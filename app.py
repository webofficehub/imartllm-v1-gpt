# app.py (robust proxy with HTTP /run/ fallback)
import os
import logging
import asyncio
import traceback
import json
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# gradio_client is required in requirements.txt
from gradio_client import Client  # type: ignore

# requests is used for direct HTTP fallback to HF Space /run/<api>
import requests

# -------- CONFIG & LOGGING --------
HF_SPACE = os.getenv("HF_SPACE", "weboffice/imartllm-v1-gpt")  # owner/repo
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
        # try the likely kwarg names
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

# -------- Helper: call HF Space /run/<api> directly --------
def call_space_run_api_via_http(message: str, system_message: str, max_tokens: int, temperature: float, top_p: float) -> Dict[str, Any]:
    """
    Fallback: call the HF Space's HTTP /run/<api> endpoint directly.
    This constructs the host from HF_SPACE: owner/repo -> owner-repo.hf.space
    and posts {"data": [ ... ]} which is what Gradio run endpoints expect.
    Requires HF_API_TOKEN to be set for private spaces.
    Returns parsed JSON if 200, else raises.
    """
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set; cannot call Space /run/<api> endpoint")

    owner_repo = HF_SPACE.replace("/", "-")
    url = f"https://{owner_repo}.hf.space/run/{HF_API_NAME.lstrip('/')}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}
    payload = {"data": [message, system_message, max_tokens, temperature, top_p]}

    logger.info("Attempting HTTP POST to Space run endpoint: %s (owner_repo=%s)", url, owner_repo)
    resp = requests.post(url, headers=headers, json=payload, timeout=min(REQUEST_TIMEOUT, 60.0))
    # Try to parse JSON body for diagnostics
    text = resp.text
    try:
        j = resp.json()
    except Exception:
        j = {"raw_text": text}

    if resp.status_code != 200:
        err = {"status_code": resp.status_code, "body": j}
        logger.warning("Space /run/ returned non-200: %s", err)
        raise RuntimeError(f"Space /run/ returned {resp.status_code}: {json.dumps(j)[:2000]}")

    # success -> return parsed body
    logger.info("Space /run/ HTTP call succeeded")
    return j

# -------- CHAT endpoint (with fallback) --------
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

    api_candidates = [f"/{HF_API_NAME.lstrip('/')}", HF_API_NAME.lstrip('/')]
    last_exc: Optional[Exception] = None
    result: Any = None
    api_used: Optional[str] = None

    # 1) Primary: try gradio_client.predict (may fail with upstream errors)
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

    # 2) If predict failed, try direct HF Space /run/<api> HTTP fallback (only if token present)
    http_fallback_result = None
    if result is None:
        try:
            logger.info("Predict failed; attempting HTTP /run/ fallback to HF Space")
            http_resp = await asyncio.to_thread(
                call_space_run_api_via_http, predict_args[0], predict_args[1], predict_args[2], predict_args[3], predict_args[4]
            )
            # many spaces return {"data": [...] } — normalize to result
            if isinstance(http_resp, dict) and "data" in http_resp:
                # if data is a list, prefer first element
                data = http_resp.get("data")
                if isinstance(data, list) and len(data) == 1:
                    http_fallback_result = data[0]
                else:
                    http_fallback_result = data
            else:
                http_fallback_result = http_resp
            result = http_fallback_result
            api_used = "http:/run/<api> fallback"
            logger.info("HTTP /run/ fallback returned type=%s", type(result).__name__)
        except Exception as e:
            logger.warning("HTTP /run/ fallback failed: %s", e)
            # attach this info to last_exc for diagnostics
            if last_exc is None:
                last_exc = e
            else:
                # combine messages
                last_exc = RuntimeError(f"predict_error: {last_exc} ; http_fallback_error: {e}")

    # If still no result, surface a helpful 502 including diagnostics (but no token)
    if result is None:
        logger.exception("All predict attempts (gradio_client + http fallback) failed")
        detail = {"error": "upstream-predict-failed", "client_meta": {"kwarg_used": _gradio_client_meta.get("kwarg_used")}}
        # include last exception trace if debug
        if LOG_LEVEL == "DEBUG" and last_exc is not None:
            detail["traceback"] = traceback.format_exc()
            try:
                detail["last_exception"] = str(last_exc)
            except Exception:
                pass
        raise HTTPException(status_code=502, detail=detail)

    # Try to extract a simple reply (same logic you had)
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
