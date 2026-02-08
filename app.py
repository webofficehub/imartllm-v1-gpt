# app.py -- Render proxy that POSTS directly to a Hugging Face Space /chat endpoint
import os
import logging
import asyncio
import httpx
import traceback
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -------- Config & logging --------
HF_SPACE_HOST = os.getenv("HF_SPACE_HOST", "https://weboffice-imartllm-v1-gpt.hf.space").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60.0"))
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "http://localhost:5173").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "0.5"))
HF_API_TOKEN = os.getenv("HF_API_TOKEN", None)  # optional: include as Bearer header if needed

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hf-space-proxy-http")

# -------- FastAPI app --------
app = FastAPI(title="HF Space /chat HTTP proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOW_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Models --------
class ChatRequest(BaseModel):
    message: str
    # optional history if you later want to pass conversation context
    history: Optional[List[Dict[str, str]]] = None
    system_message: Optional[str] = "You are a friendly Chatbot."
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95

class ChatResponse(BaseModel):
    reply: Optional[str] = None
    raw: Optional[dict] = None

# -------- Helpers --------
def _build_payload(req: ChatRequest) -> dict:
    """
    Build the JSON body the Space expects for its /chat endpoint.
    Matches the Gradio inputs order: [message, history, system_message, max_tokens, temperature, top_p]
    """
    history = req.history if req.history is not None else []
    data = [req.message, history, req.system_message or "You are a friendly Chatbot.", req.max_tokens or 1024, req.temperature or 0.7, req.top_p or 0.95]
    return {"data": data}

async def _post_chat(path: str, json_body: dict, timeout: float) -> httpx.Response:
    headers = {"Content-Type": "application/json"}
    # optionally include HF API token as Authorization if your Space is protected
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
        return await client.post(path, json=json_body, headers=headers)

def _extract_reply_from_space_json(j: dict) -> Optional[str]:
    """
    Gradio /predict-style responses normally return {"data": [<reply>, ...], ...}
    For /chat top-level we expect similar shapes; be defensive.
    """
    if not isinstance(j, dict):
        return None
    # common: {"data": ["final reply", ...], ...}
    if "data" in j and isinstance(j["data"], list) and len(j["data"]) > 0:
        first = j["data"][0]
        if isinstance(first, (str, int, float)):
            return str(first)
        # sometimes nested structures: {"data": [{"reply": "..."}]}
        if isinstance(first, dict):
            for k in ("reply", "response", "output", "text"):
                if k in first and isinstance(first[k], (str, int, float)):
                    return str(first[k])
            # try stringify
            try:
                return str(first)
            except Exception:
                pass
    # fallback: direct "detail" or "message"
    for k in ("reply", "response", "result", "message", "detail"):
        if k in j and isinstance(j[k], (str, int, float)):
            return str(j[k])
    return None

# -------- Health & debug --------
@app.get("/health")
async def health():
    return {"status": "ok", "space_host": HF_SPACE_HOST}

@app.get("/_debug/config")
async def debug_config():
    return {"hf_space_host": HF_SPACE_HOST, "request_timeout": REQUEST_TIMEOUT, "max_retries": MAX_RETRIES, "kw_token": bool(HF_API_TOKEN)}

# -------- Chat endpoint (calls HF Space /chat directly) --------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    logger.info("Received /chat request: message_len=%d history_len=%d", len(req.message or ""), len(req.history or []))
    json_body = _build_payload(req)
    url = f"{HF_SPACE_HOST}/chat"  # you said the real endpoint is /chat

    last_exc = None
    final_json = None
    status_code = None

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            logger.debug("POSTing to HF Space %s (attempt %d)", url, attempt)
            resp = await _post_chat(url, json_body, timeout=REQUEST_TIMEOUT)
            status_code = resp.status_code
            text = resp.text
            logger.debug("HF response HTTP %d (len=%d)", status_code, len(text or ""))

            if resp.status_code != 200:
                # try to parse JSON error detail if available
                try:
                    err_j = resp.json()
                    logger.warning("HF Space returned non-200: %s", err_j)
                    last_exc = HTTPException(status_code=502, detail={"error": "upstream-non-200", "status": resp.status_code, "body": err_j})
                except Exception:
                    last_exc = HTTPException(status_code=502, detail={"error": "upstream-non-200", "status": resp.status_code, "body": resp.text})
                raise last_exc

            # parse JSON
            try:
                j = resp.json()
            except Exception as e:
                logger.exception("Failed to parse JSON from HF Space: %s", e)
                last_exc = e
                raise HTTPException(status_code=502, detail={"error": "invalid-upstream-json", "body": resp.text})

            final_json = j
            break

        except Exception as e:
            last_exc = e
            logger.warning("Attempt %d failed: %s", attempt, str(e))
            # backoff if we will retry
            if attempt <= MAX_RETRIES:
                backoff = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.info("Retrying after %.2fs backoff (attempt %d)", backoff, attempt + 1)
                await asyncio.sleep(backoff)
            else:
                logger.exception("All attempts failed; last exception: %s", e)

    if final_json is None:
        # return a helpful 502 with last error details
        detail = {"error": "upstream-unavailable", "last_exception": str(last_exc)}
        if LOG_LEVEL == "DEBUG":
            detail["traceback"] = traceback.format_exc()
        raise HTTPException(status_code=502, detail=detail)

    reply = _extract_reply_from_space_json(final_json)
    raw_out = {"upstream_status": status_code, "upstream_json": final_json}
    return ChatResponse(reply=reply, raw=raw_out)

# -------- Local dev runner --------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level=LOG_LEVEL.lower())
