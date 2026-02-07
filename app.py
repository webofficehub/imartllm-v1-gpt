# app.py
import os
import logging
import json
from typing import Optional, AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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

class ConvertRequest(BaseModel):
    from_currency: str
    to_currency: str
    amount: Optional[float] = 1.0
    rate: Optional[float] = None
    # optional override for system/prompt
    system_message: Optional[str] = "You are an assistant that provides short, actionable currency conversion insights."
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.9

# ---------------------------
# Helpers
# ---------------------------
def space_base_url(space: str) -> str:
    parts = space.split("/")
    if len(parts) != 2:
        raise ValueError("HF_SPACE must be 'owner/repo'")
    owner, repo = parts
    return f"https://{owner}-{repo}.hf.space"

def sse_event(data: str, event: Optional[str] = None) -> str:
    """
    Format a Server-Sent Event block. `data` may contain multiple lines; we
    prefix each line with `data: ` per SSE spec.
    """
    out = []
    if event:
        out.append(f"event: {event}")
    # ensure string
    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False)
    for line in data.splitlines() or [""]:
        out.append(f"data: {line}")
    out.append("")  # trailing blank line to terminate the event
    return "\n".join(out) + "\n"

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
# Streaming convert-insight endpoint
# ---------------------------
@app.post("/convert-insight")
async def convert_insight_endpoint(req: ConvertRequest, request: Request):
    """
    Streams SSE events:
      - event: summary  -> immediate small JSON { summary, meta }
      - event: insight  -> one or many events containing partial text { text }
      - event: done     -> final event when completed
      - event: error    -> error details

    The endpoint will attempt to proxy to the configured HF Space. If the Space response
    is chunked, it will forward chunks as `insight` events. Otherwise it sends a single `insight` event.
    """
    base = space_base_url(HF_SPACE)
    api_name_clean = HF_API_NAME.lstrip("/")
    urls_to_try = [
        f"{base}/gradio_api/call/{api_name_clean}",
        f"{base}/run/{api_name_clean}",
    ]
    # Build a short conversion summary locally
    def build_summary() -> dict:
        rate_display = None
        converted_display = None
        if req.rate is not None:
            try:
                rate_display = float(req.rate)
                converted_display = float(req.amount) * float(req.rate)
            except Exception:
                rate_display = None
                converted_display = None
        return {
            "summary": f"Converted {req.amount} {req.from_currency} → {req.to_currency} = "
                       + (f"{converted_display:.6f}" if converted_display is not None else "—"),
            "meta": {
                "from": req.from_currency,
                "to": req.to_currency,
                "amount": req.amount,
                "rate": req.rate,
            },
        }

    # Prepare the assistant prompt: include the summary and ask for concise insight
    summary_obj = build_summary()
    # Compose the model message: include summary + instruction
    model_message = (
        f"Conversion summary:\n{summary_obj['summary']}\n\n"
        f"Provide a short, actionable insight for this conversion (2-5 bullets or 2-3 sentences). "
        f"Include potential cautions (fees, rounding) and a final short suggestion for display formatting."
    )

    # SSE generator
    async def event_generator() -> AsyncGenerator[str, None]:
        # If client disconnects, we stop early
        client_disconnected = False

        # Immediately yield summary event
        yield sse_event(summary_obj, event="summary")

        payload = gradio_payload_for_chat(
            message=model_message,
            system_message=req.system_message or "You are an assistant that provides short, actionable currency conversion insights.",
            max_tokens=req.max_tokens or 512,
            temperature=req.temperature or 0.2,
            top_p=req.top_p or 0.9,
        )

        headers = hf_headers()

        last_exception = None
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            for url in urls_to_try:
                try:
                    # Try streaming with httpx. If the Space returns a streaming/chunked body, we'll forward chunks.
                    # Note: many Gradio Spaces are not streaming; in that case we will receive a single full response.
                    async with client.stream("POST", url, json=payload, headers=headers) as resp:
                        status = resp.status_code
                        ct = resp.headers.get("content-type", "")
                        logger.info("Proxying to %s (status=%s, content-type=%s)", url, status, ct)

                        if status != 200:
                            # read body for logging (small bodies only)
                            try:
                                text_body = await resp.aread()
                                text_body = text_body.decode(errors="ignore") if isinstance(text_body, (bytes, bytearray)) else str(text_body)
                            except Exception:
                                text_body = "<failed to read body>"
                            last_exception = Exception(f"Space returned status {status}: {text_body}")
                            logger.warning("Non-200 from space %s: %s", url, text_body)
                            continue

                        # If content-type suggests SSE or text/event-stream, forward as-is by parsing SSE lines.
                        if "text/event-stream" in ct or "event-stream" in ct:
                            # forward each line-block as an insight event
                            async for raw_chunk in resp.aiter_bytes():
                                if await client_is_disconnected(request):
                                    client_disconnected = True
                                    break
                                try:
                                    text = raw_chunk.decode("utf-8", errors="replace")
                                except Exception:
                                    text = str(raw_chunk)
                                # Forward chunk as insight event (we trim whitespace)
                                if text.strip():
                                    yield sse_event({"text": text}, event="insight")
                            if client_disconnected:
                                break
                            # finished streaming from this url
                            yield sse_event({"status": "complete"}, event="done")
                            return

                        # Otherwise, attempt to stream raw bytes/chunks and forward decoded partial text
                        # This helps when upstream is chunked but not SSE.
                        streamed = False
                        async for raw_chunk in resp.aiter_bytes():
                            if await client_is_disconnected(request):
                                client_disconnected = True
                                break
                            try:
                                text = raw_chunk.decode("utf-8", errors="replace")
                            except Exception:
                                text = str(raw_chunk)
                            if text.strip():
                                streamed = True
                                yield sse_event({"text": text}, event="insight")
                        if client_disconnected:
                            break
                        if streamed:
                            yield sse_event({"status": "complete"}, event="done")
                            return

                        # If we reached here there were no chunked bytes or upstream returned full JSON at once.
                        # Read it fully and emit a final insight event.
                        try:
                            full = await resp.json()
                            # Try to extract the first data element like gradio does
                            candidate_text = None
                            if isinstance(full, dict) and "data" in full:
                                arr = full["data"]
                                if isinstance(arr, list) and len(arr) > 0:
                                    cand = arr[0]
                                    # if cand is JSON-able, stringify intelligently
                                    if isinstance(cand, (str, int, float)):
                                        candidate_text = str(cand)
                                    else:
                                        candidate_text = json.dumps(cand, ensure_ascii=False)
                            else:
                                if isinstance(full, (str, int, float)):
                                    candidate_text = str(full)
                                else:
                                    candidate_text = json.dumps(full, ensure_ascii=False)

                            if candidate_text:
                                yield sse_event({"text": candidate_text}, event="insight")
                                yield sse_event({"status": "complete"}, event="done")
                                return
                            else:
                                # no usable candidate — forward raw text
                                raw_text = json.dumps(full, ensure_ascii=False)
                                yield sse_event({"text": raw_text}, event="insight")
                                yield sse_event({"status": "complete"}, event="done")
                                return

                        except Exception as e:
                            # final fallback: try reading text
                            try:
                                txt = await resp.aread()
                                txt = txt.decode("utf-8", errors="replace") if isinstance(txt, (bytes, bytearray)) else str(txt)
                                if txt:
                                    yield sse_event({"text": txt}, event="insight")
                                    yield sse_event({"status": "complete"}, event="done")
                                    return
                            except Exception as ex:
                                last_exception = ex
                                logger.exception("Failed to parse space response: %s", ex)
                                continue

                except httpx.RequestError as e:
                    last_exception = e
                    logger.warning("RequestError for %s: %s", url, e)
                    continue
                except Exception as e:
                    last_exception = e
                    logger.exception("Unhandled exception when calling %s: %s", url, e)
                    continue

        # If we got here, none of the URLs provided a usable response
        err_msg = {"error": "Failed to obtain insight from HF Space"}
        if last_exception:
            err_msg["detail"] = str(last_exception)
        yield sse_event(err_msg, event="error")
        return

    # Return streaming response
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Helpers to detect client disconnect (fastapi starlette)
async def client_is_disconnected(request: Request) -> bool:
    """
    Attempts to detect whether the client disconnected.
    If it's not supported, returns False and streaming continues.
    """
    try:
        return await request.is_disconnected()
    except Exception:
        return False

# ---------------------------
# If run directly (useful for local dev)
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
