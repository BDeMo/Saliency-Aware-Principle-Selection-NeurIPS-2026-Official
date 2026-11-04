# run/instruct_clean.py
import os
import re
import time
import json
import asyncio
import threading
from typing import Any, Dict, Optional, List

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# --------------------------
# Upstream
# --------------------------
UPSTREAM_BASE = os.environ.get("UPSTREAM_BASE", "http://127.0.0.1:8001").rstrip("/")
UPSTREAM_PATH = os.environ.get("UPSTREAM_PATH", "/v1/chat/completions").strip()
UPSTREAM_MODEL = os.environ.get("UPSTREAM_MODEL", "Qwen3-VL-8B-Instruct").strip()

# Proxy model name
PROXY_MODEL = os.environ.get("PROXY_MODEL", "instruct-clean").strip()

# --------------------------
# Timeouts + concurrency
# --------------------------
CONCURRENCY = int(os.environ.get("CONCURRENCY", "32"))
SEM = asyncio.Semaphore(CONCURRENCY)

TIMEOUT_CONNECT_S = float(os.environ.get("TIMEOUT_CONNECT_S", "10"))
TIMEOUT_READ_S    = float(os.environ.get("TIMEOUT_READ_S", "900"))
TIMEOUT_WRITE_S   = float(os.environ.get("TIMEOUT_WRITE_S", "30"))
TIMEOUT_POOL_S    = float(os.environ.get("TIMEOUT_POOL_S", "10"))

UPSTREAM_MAX_RETRIES = int(os.environ.get("UPSTREAM_MAX_RETRIES", "2"))
UPSTREAM_RETRY_BACKOFF_S = float(os.environ.get("UPSTREAM_RETRY_BACKOFF_S", "0.6"))

HTTPX_MAX_CONNECTIONS = int(os.environ.get("HTTPX_MAX_CONNECTIONS", str(CONCURRENCY)))
HTTPX_MAX_KEEPALIVE = int(os.environ.get("HTTPX_MAX_KEEPALIVE", str(min(CONCURRENCY, 64))))

# --------------------------
# Output safety
# --------------------------
EMPTY_FALLBACK = os.environ.get("EMPTY_FALLBACK", "unknown").strip() or "unknown"
MIN_CONTENT_CHARS = int(os.environ.get("MIN_CONTENT_CHARS", "1"))

BENCHMARK_SAFE = os.environ.get("BENCHMARK_SAFE", "1").strip() in ("1", "true", "True")
BENCHMARK_SAFE_MODE = os.environ.get("BENCHMARK_SAFE_MODE", "auto").strip().lower()  # auto|choice_only|clean_text
MAX_FINAL_CHARS = int(os.environ.get("MAX_FINAL_CHARS", "128"))

# --------------------------
# Debug JSONL
# --------------------------
DEBUG_INSTRUCT_CLEAN = os.environ.get("DEBUG_INSTRUCT_CLEAN", "0").strip() in ("1", "true", "True")
DEBUG_INSTRUCT_CLEAN_MAXLEN = int(os.environ.get("DEBUG_INSTRUCT_CLEAN_MAXLEN", "4000"))
DEBUG_INSTRUCT_CLEAN_LOGFILE = os.environ.get("DEBUG_INSTRUCT_CLEAN_LOGFILE", "").strip()
DEBUG_INSTRUCT_CLEAN_LOG_REQUEST = os.environ.get("DEBUG_INSTRUCT_CLEAN_LOG_REQUEST", "1").strip() in ("1", "true", "True")
DEBUG_INSTRUCT_CLEAN_LOG_RESPONSE = os.environ.get("DEBUG_INSTRUCT_CLEAN_LOG_RESPONSE", "1").strip() in ("1", "true", "True")

_LOG_LOCK = threading.Lock()

# markdown/latex cleanup
_CODE_FENCE_RE = re.compile(r"```.*?```", flags=re.DOTALL)
_LATEX_BLOCK_RE = re.compile(r"\$\$.*?\$\$", flags=re.DOTALL)
_LATEX_INLINE_RE = re.compile(r"\$[^$]*\$")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+.*$", flags=re.MULTILINE)
_MD_HR_RE = re.compile(r"^\s*---+\s*$", flags=re.MULTILINE)
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
_EDGE_QUOTES_RE = re.compile(r'^[\'"“”‘’]+|[\'"“”‘’]+$')

_CHOICE_LETTER_RE = re.compile(r"(?:^|\b)(?:option\s*)?([A-E])(?:\b|[\.\:\)\]])", flags=re.IGNORECASE)
_CHOICE_PAREN_RE = re.compile(r"[\(\[]\s*([A-E])\s*[\)\]]", flags=re.IGNORECASE)

app = FastAPI(title="instruct-clean: OpenAI proxy with safe output + retries + debug jsonl")
_CLIENT: Optional[httpx.AsyncClient] = None


def _now_unix() -> int:
    return int(time.time())


def _truncate(x: Any, maxlen: int) -> str:
    try:
        s = json.dumps(x, ensure_ascii=False)
    except Exception:
        s = repr(x)
    if len(s) > maxlen:
        return s[:maxlen] + "...<truncated>"
    return s


def _dbg(title: str, obj: Any):
    if not DEBUG_INSTRUCT_CLEAN:
        return
    print(f"\n[DEBUG_INSTRUCT_CLEAN] {title}\n{_truncate(obj, DEBUG_INSTRUCT_CLEAN_MAXLEN)}\n")


def _log_jsonl(event: str, payload: Dict[str, Any]):
    if (not DEBUG_INSTRUCT_CLEAN) or (not DEBUG_INSTRUCT_CLEAN_LOGFILE):
        return
    rec = {"ts": time.time(), "event": event, "payload": payload}
    line = json.dumps(rec, ensure_ascii=False)

    path = DEBUG_INSTRUCT_CLEAN_LOGFILE
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    except Exception as e:
        print(f"[DEBUG_INSTRUCT_CLEAN][WARN] mkdir failed for {path}: {e}")
        print(line)
        return

    with _LOG_LOCK:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"[DEBUG_INSTRUCT_CLEAN][WARN] write failed for {path}: {e}")
            print(line)


def _force_upstream_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})
    out["model"] = UPSTREAM_MODEL
    # ensure non-streaming for eval stability
    out["stream"] = False
    return out


def _ensure_nonempty(s: Any) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip()
    if len(s) >= MIN_CONTENT_CHARS:
        return s
    return EMPTY_FALLBACK


def _extract_choice_letter(text: str) -> Optional[str]:
    if not text:
        return None
    hits: List[str] = []
    for m in _CHOICE_PAREN_RE.finditer(text):
        hits.append(m.group(1).upper())
    for m in _CHOICE_LETTER_RE.finditer(text):
        hits.append(m.group(1).upper())
    return hits[-1] if hits else None


def _clean_text(text: str) -> str:
    t = text or ""
    t = _CODE_FENCE_RE.sub(" ", t)
    t = _LATEX_BLOCK_RE.sub(" ", t)
    t = _LATEX_INLINE_RE.sub(" ", t)
    t = _MD_HEADING_RE.sub(" ", t)
    t = _MD_HR_RE.sub(" ", t)
    t = _NON_ASCII_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _EDGE_QUOTES_RE.sub("", t).strip()
    if len(t) > MAX_FINAL_CHARS:
        t = t[:MAX_FINAL_CHARS].rstrip()
    return _ensure_nonempty(t)


def _benchmark_safe(text: str) -> str:
    mode = BENCHMARK_SAFE_MODE if BENCHMARK_SAFE_MODE in ("auto", "choice_only", "clean_text") else "auto"
    t = (text or "").strip()

    if mode in ("auto", "choice_only"):
        c = _extract_choice_letter(t)
        if c:
            return c
        if mode == "choice_only":
            return EMPTY_FALLBACK

    return _clean_text(t)


def _force_min_openai() -> Dict[str, Any]:
    t = _now_unix()
    return {
        "id": f"chatcmpl-{t}",
        "object": "chat.completion",
        "created": t,
        "model": PROXY_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": EMPTY_FALLBACK}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sanitize_openai(resp: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(resp, dict):
        return _force_min_openai()

    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return _force_min_openai()

    for c in choices:
        if not isinstance(c, dict):
            continue
        msg = c.get("message")
        if not isinstance(msg, dict):
            c["message"] = {"role": "assistant", "content": EMPTY_FALLBACK}
            msg = c["message"]

        content = msg.get("content", "")
        content = _ensure_nonempty(content)
        if BENCHMARK_SAFE:
            content = _benchmark_safe(content)
        msg["content"] = content

        if "finish_reason" not in c:
            c["finish_reason"] = "stop"

    # normalize model name
    resp["model"] = PROXY_MODEL
    return resp


@app.on_event("startup")
async def _startup():
    global _CLIENT
    timeout = httpx.Timeout(connect=TIMEOUT_CONNECT_S, read=TIMEOUT_READ_S, write=TIMEOUT_WRITE_S, pool=TIMEOUT_POOL_S)
    limits = httpx.Limits(max_connections=HTTPX_MAX_CONNECTIONS, max_keepalive_connections=HTTPX_MAX_KEEPALIVE)
    _CLIENT = httpx.AsyncClient(timeout=timeout, limits=limits)


@app.on_event("shutdown")
async def _shutdown():
    global _CLIENT
    if _CLIENT is not None:
        await _CLIENT.aclose()
    _CLIENT = None


async def _call_upstream(payload: Dict[str, Any]) -> Dict[str, Any]:
    async with SEM:
        global _CLIENT
        if _CLIENT is None:
            timeout = httpx.Timeout(connect=TIMEOUT_CONNECT_S, read=TIMEOUT_READ_S, write=TIMEOUT_WRITE_S, pool=TIMEOUT_POOL_S)
            limits = httpx.Limits(max_connections=HTTPX_MAX_CONNECTIONS, max_keepalive_connections=HTTPX_MAX_KEEPALIVE)
            _CLIENT = httpx.AsyncClient(timeout=timeout, limits=limits)

        last_exc: Optional[Exception] = None

        for attempt in range(UPSTREAM_MAX_RETRIES + 1):
            try:
                r = await _CLIENT.post(f"{UPSTREAM_BASE}{UPSTREAM_PATH}", json=payload)

                try:
                    body = r.json()
                except Exception:
                    body = {"_non_json": r.text}

                if DEBUG_INSTRUCT_CLEAN and DEBUG_INSTRUCT_CLEAN_LOG_RESPONSE:
                    _log_jsonl("upstream_response", {"status_code": int(r.status_code), "body": _truncate(body, DEBUG_INSTRUCT_CLEAN_MAXLEN)})

                if r.status_code >= 400:
                    return {"_upstream_error": True, "status_code": r.status_code, "body": body}

                return body

            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPError) as e:
                last_exc = e
                if DEBUG_INSTRUCT_CLEAN:
                    _log_jsonl("upstream_exception", {"attempt": attempt, "type": type(e).__name__, "message": str(e)})
                if attempt < UPSTREAM_MAX_RETRIES:
                    await asyncio.sleep(UPSTREAM_RETRY_BACKOFF_S * (2 ** attempt))
                    continue
                break

        return {
            "_upstream_error": True,
            "status_code": 504,
            "body": {
                "error": "upstream_timeout",
                "exception": type(last_exc).__name__ if last_exc else "unknown",
                "message": str(last_exc) if last_exc else "unknown",
            },
        }


@app.post("/v1/chat/completions")
async def chat(req: Request):
    payload = await req.json()

    if DEBUG_INSTRUCT_CLEAN and DEBUG_INSTRUCT_CLEAN_LOG_REQUEST:
        _log_jsonl(
            "request",
            {
                "model_in": payload.get("model"),
                "forced_upstream_model": UPSTREAM_MODEL,
                "messages_len": len(payload.get("messages") or []),
                "max_tokens": payload.get("max_tokens"),
                "temperature": payload.get("temperature"),
                "benchmark_safe": BENCHMARK_SAFE,
                "benchmark_safe_mode": BENCHMARK_SAFE_MODE,
            },
        )

    payload2 = _force_upstream_model(payload)
    upstream_resp = await _call_upstream(payload2)

    if isinstance(upstream_resp, dict) and upstream_resp.get("_upstream_error"):
        out = {
            "error": {
                "message": "Upstream error",
                "type": "upstream_error",
                "upstream_base": UPSTREAM_BASE,
                "upstream_path": UPSTREAM_PATH,
                "upstream_body": upstream_resp.get("body"),
            }
        }
        if DEBUG_INSTRUCT_CLEAN and DEBUG_INSTRUCT_CLEAN_LOG_RESPONSE:
            _log_jsonl("proxy_error_response", {"status_code": int(upstream_resp.get("status_code", 502)), "body": _truncate(out, DEBUG_INSTRUCT_CLEAN_MAXLEN)})
        return JSONResponse(status_code=int(upstream_resp.get("status_code", 502)), content=out)

    out = _sanitize_openai(upstream_resp)

    if DEBUG_INSTRUCT_CLEAN and DEBUG_INSTRUCT_CLEAN_LOG_RESPONSE:
        c0 = ""
        try:
            c0 = out.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            pass
        _log_jsonl("proxy_response", {"status_code": 200, "content": _truncate(c0, DEBUG_INSTRUCT_CLEAN_MAXLEN)})

    return JSONResponse(out)


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": PROXY_MODEL, "object": "model", "created": _now_unix(), "owned_by": "proxy"}],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "proxy_model": PROXY_MODEL,
        "benchmark_safe": BENCHMARK_SAFE,
        "benchmark_safe_mode": BENCHMARK_SAFE_MODE,
        "timeouts": {"connect": TIMEOUT_CONNECT_S, "read": TIMEOUT_READ_S, "write": TIMEOUT_WRITE_S, "pool": TIMEOUT_POOL_S},
        "retries": {"max_retries": UPSTREAM_MAX_RETRIES, "backoff_s": UPSTREAM_RETRY_BACKOFF_S},
        "upstream": {"base": UPSTREAM_BASE, "path": UPSTREAM_PATH, "upstream_model": UPSTREAM_MODEL},
        "debug": {"enabled": DEBUG_INSTRUCT_CLEAN, "logfile": DEBUG_INSTRUCT_CLEAN_LOGFILE},
    }
