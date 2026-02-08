# app.py -- Render-ready proxy for a streaming HF Space (robust to sync/async gradio_client versions)
import os
import logging
import asyncio
import time
import traceback
import inspect
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gradio_client import Client  # type: ignore

# -------- CONFIG & LOGGING --------
HF_SPACE = os.getenv("HF_SPACE", "weboffice/imartllm-v1-gpt")
HF_API_NAME = os.getenv("HF_API_NAME", "chat")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60.0"))
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "http://localhost:5173").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "0.5"))

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
    global _gradio_client, _gradio_client_meta
    if _gradio_client is not None:
        return _gradio_client

    _gradio_client_meta["init_attempted"] = True
    _gradio_client_meta["kwarg_used"] = None
    _gradio_client_meta["error"] = None

    if HF_API_TOKEN:
        for kw in ("token", "hf_token"):
            try:
                logger.info("Attempting Client(%s, %s=...)", HF_SPACE, kw)
                _gradio_client = Client(HF_SPACE, **{kw: HF_API_TOKEN})
                _gradio_client_meta["kwarg_used"] = kw
                logger.info("gradio_client created using %s", kw)
                return _gradio_client
            except TypeError as e:
                logger.debug("%s kwarg not supported: %s", kw, e)
            except Exception as e:
                logger.warning("Unexpected error constructing Client with %s: %s", kw, e)
                _gradio_client_meta["error"] = str(e)

    try:
        logger.info("Attempting Client(%s) without token", HF_SPACE)
        _gradio_client = Client(HF_SPACE)
        _gradio_client_meta["kwarg_used"] = "none"
        logger.info("gradio_client created without token")
        return _gradio_client
    except Exception as e:
        logger.exception("Failed to construct gradio_client.Client: %s", e)
        _gradio_client_meta["error"] = traceback.format_exc()
        raise

# -------- Helpers for sync/async callables --------
async def _call_flexible(func, *args, **kwargs):
    """
    Call `func(*args, **kwargs)` in a way that supports:
      - synchronous functions (call inside asyncio.to_thread)
      - coroutine functions (await directly)
    Also handles cases where a synchronous call returns a coroutine (await that).
    """
    try:
        if inspect.iscoroutinefunction(func):
            # func is defined async def -> await
            return await func(*args, **kwargs)
        # call in thread so we don't block the event loop
        res = await asyncio.to_thread(func, *args, **kwargs)
        if asyncio.iscoroutine(res):
            # the function returned a coroutine object (some libraries do that) -> await it
            return await res
        return res
    except Exception:
        # re-raise so caller can handle/log
        raise

async def _call_attr_flexible(obj, attr_name: str, *args, **kwargs):
    """
    Helper to call obj.attr_name(...) whether attr is sync or async (or returns coroutine).
    """
    attr = getattr(obj, attr_name)
    return await _call_flexible(attr, *args, **kwargs)

# extract reply
def _extract_reply_from_result(result: Any) -> Optional[str]:
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
    return reply

# -------- Health & debug --------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/_debug/client-meta")
async def client_meta():
    return _gradio_client_meta

# -------- CHAT endpoint (robust) --------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    logger.info("Received /chat request: message_len=%d system_message_len=%d", len(req.message or ""), len(req.system_message or ""))

    try:
        client = get_gradio_client()
    except Exception as e:
        logger.exception("Failed to initialize gradio_client: %s", e)
        raise HTTPException(status_code=500, detail="Server misconfiguration: failed to initialize gradio client")

    predict_args = (
        req.message,
        req.system_message or "You are a friendly Chatbot.",
        req.max_tokens or 1024,
        req.temperature or 0.7,
        req.top_p or 0.95,
    )

    # broaden candidate api names (leading slash variants and common names)
    base_name = HF_API_NAME.lstrip("/")
    api_candidates = []
    for n in (HF_API_NAME, base_name, "/predict", "predict", "/run", "run", "/chat", "chat"):
        if n not in api_candidates:
            api_candidates.append(n)

    last_exc: Optional[Exception] = None
    final_result: Any = None
    api_used: Optional[str] = None

    # Try with retries/backoff
    for attempt in range(1, MAX_RETRIES + 2):
        for api_name_try in api_candidates:
            try:
                logger.debug("Attempt %d: submitting job with api_name=%s", attempt, api_name_try)
                # Use flexible caller to support sync or async client.submit
                job = await _call_flexible(client.submit, *predict_args, api_name=api_name_try)
                api_used = api_name_try

                # wait for job to complete with timeout: use job.done() whether sync or async
                try:
                    start = time.time()
                    while True:
                        # 'done' may be sync or async
                        done = await _call_attr_flexible(job, "done")
                        if done:
                            break
                        if time.time() - start > REQUEST_TIMEOUT:
                            # try cancel if available
                            try:
                                await _call_attr_flexible(job, "cancel")
                            except Exception:
                                pass
                            raise asyncio.TimeoutError("Timeout waiting for job to finish")
                        await asyncio.sleep(0.1)
                except asyncio.TimeoutError:
                    logger.warning("Job timed out for api_name=%s (attempt=%d)", api_name_try, attempt)
                    raise

                # collect outputs (outputs() or result())
                try:
                    outputs = await _call_attr_flexible(job, "outputs")
                except Exception:
                    outputs = None

                if outputs:
                    final_result = outputs[-1]
                else:
                    final_result = await _call_attr_flexible(job, "result")

                logger.info("Job completed for api_name=%s on attempt %d", api_name_try, attempt)
                break

            except asyncio.TimeoutError as e:
                last_exc = e
                logger.warning("Predict job timed out for api_name=%s after %s seconds (attempt=%d)", api_name_try, REQUEST_TIMEOUT, attempt)
            except Exception as e:
                last_exc = e
                # quiet common "not found" messages at INFO/WARN level but keep details in DEBUG
                logger.warning("Submit/Job failed for api_name=%s on attempt %d: %s", api_name_try, attempt, e)
                logger.debug(traceback.format_exc())

        if final_result is not None:
            break

        if attempt <= MAX_RETRIES:
            backoff = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            logger.info("Retrying after %.2fs backoff (attempt %d)", backoff, attempt + 1)
            await asyncio.sleep(backoff)

    if final_result is None:
        logger.exception("All predict attempts failed after retries")
        detail = {"error": "upstream-predict-failed", "client_meta": {"kwarg_used": _gradio_client_meta.get("kwarg_used"), "api_used": api_used}}
        if LOG_LEVEL == "DEBUG" and last_exc is not None:
            detail["traceback"] = traceback.format_exc()
        raise HTTPException(status_code=502, detail=detail)

    reply = _extract_reply_from_result(final_result)
    raw_out = {
        "result": final_result,
        "client_meta": {"kwarg_used": _gradio_client_meta.get("kwarg_used"), "api_used": api_used}
    }

    return ChatResponse(reply=reply, raw=raw_out)

# -------- LOCAL DEV RUNNER --------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level=LOG_LEVEL.lower())
