# run/thinking_clean.py
import os
import re
import time
import json
import asyncio
import threading
from typing import Any, Dict, Optional, Tuple, List

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# ==========================================================
# Config
# ==========================================================
UPSTREAM_BASE = os.environ.get("UPSTREAM_BASE", "http://127.0.0.1:8002").rstrip("/")
UPSTREAM_PATH = os.environ.get("UPSTREAM_PATH", "/v1/chat/completions").strip()

UPSTREAM_MODEL = os.environ.get("UPSTREAM_MODEL", "Qwen3-VL-8B-Thinking").strip()
PROXY_MODEL = os.environ.get("PROXY_MODEL", "thinking-clean").strip()

# strip | keep | auto
STRIP_MODE = os.environ.get("STRIP_MODE", "auto").strip().lower()

TIMEOUT_S = float(os.environ.get("TIMEOUT_S", "180"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "128"))
SEM = asyncio.Semaphore(CONCURRENCY)

# Always return non-empty content (avoid "Failed API")
EMPTY_FALLBACK = os.environ.get("EMPTY_FALLBACK", "unknown").strip() or "unknown"
MIN_CONTENT_CHARS = int(os.environ.get("MIN_CONTENT_CHARS", "1"))

# ==========================================================
# Benchmark-safe output cleanup (IMPORTANT for MMMU_Pro_Vision)
# ==========================================================
BENCHMARK_SAFE = os.environ.get("BENCHMARK_SAFE", "1").strip() in ("1", "true", "True")
BENCHMARK_SAFE_MODE = os.environ.get("BENCHMARK_SAFE_MODE", "auto").strip().lower()
# auto: try choice-letter extraction; else clean text
# choice_only: only return A/B/C/D/E if found else fallback
# clean_text: skip letter extraction; just clean markdown/latex/etc

MAX_FINAL_CHARS = int(os.environ.get("MAX_FINAL_CHARS", "256"))  # keep answer short for parsers

# ==========================================================
# DEBUG (stdout + jsonl, same style as evo proxy)
# ==========================================================
DEBUG_THINKING_CLEAN = os.environ.get("DEBUG_THINKING_CLEAN", "0").strip() in ("1", "true", "True")
DEBUG_THINKING_CLEAN_MAXLEN = int(os.environ.get("DEBUG_THINKING_CLEAN_MAXLEN", "4000"))

DEBUG_THINKING_CLEAN_LOGFILE = os.environ.get("DEBUG_THINKING_CLEAN_LOGFILE", "").strip()
DEBUG_THINKING_CLEAN_LOG_REQUEST = os.environ.get("DEBUG_THINKING_CLEAN_LOG_REQUEST", "1").strip() in ("1", "true", "True")
DEBUG_THINKING_CLEAN_LOG_RESPONSE = os.environ.get("DEBUG_THINKING_CLEAN_LOG_RESPONSE", "1").strip() in ("1", "true", "True")

# ---- timeout granular (seconds) ----
TIMEOUT_CONNECT_S = float(os.environ.get("TIMEOUT_CONNECT_S", "10"))
TIMEOUT_READ_S    = float(os.environ.get("TIMEOUT_READ_S", "600"))   # 关键：读超时要大
TIMEOUT_WRITE_S   = float(os.environ.get("TIMEOUT_WRITE_S", "30"))
TIMEOUT_POOL_S    = float(os.environ.get("TIMEOUT_POOL_S", "10"))

# retries for transient issues
UPSTREAM_MAX_RETRIES = int(os.environ.get("UPSTREAM_MAX_RETRIES", "2"))
UPSTREAM_RETRY_BACKOFF_S = float(os.environ.get("UPSTREAM_RETRY_BACKOFF_S", "0.6"))

# connection limits (avoid too many sockets)
HTTPX_MAX_CONNECTIONS = int(os.environ.get("HTTPX_MAX_CONNECTIONS", str(CONCURRENCY)))
HTTPX_MAX_KEEPALIVE = int(os.environ.get("HTTPX_MAX_KEEPALIVE", str(min(CONCURRENCY, 64))))


_LOG_LOCK = threading.Lock()

app = FastAPI(title="Thinking-clean OpenAI-compatible proxy (strip+benchmark-safe+debug)")

_CLIENT: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
@app.on_event("startup")
async def _startup():
    global _CLIENT
    timeout = httpx.Timeout(
        connect=TIMEOUT_CONNECT_S,
        read=TIMEOUT_READ_S,
        write=TIMEOUT_WRITE_S,
        pool=TIMEOUT_POOL_S,
    )
    limits = httpx.Limits(
        max_connections=HTTPX_MAX_CONNECTIONS,
        max_keepalive_connections=HTTPX_MAX_KEEPALIVE,
    )
    _CLIENT = httpx.AsyncClient(timeout=timeout, limits=limits)



@app.on_event("shutdown")
async def _shutdown():
    global _CLIENT
    if _CLIENT is not None:
        await _CLIENT.aclose()
    _CLIENT = None


# ==========================================================
# Regex for robust stripping
# ==========================================================
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", flags=re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", flags=re.IGNORECASE)
_FINAL_MARK_RE = re.compile(r"\bfinal\s*:\s*", flags=re.IGNORECASE)

# Markdown / LaTeX cleanup
_CODE_FENCE_RE = re.compile(r"```.*?```", flags=re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_LATEX_BLOCK_RE = re.compile(r"\$\$.*?\$\$", flags=re.DOTALL)
_LATEX_INLINE_RE = re.compile(r"\$[^$]*\$")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+.*$", flags=re.MULTILINE)
_MD_BOLD_ITALIC_RE = re.compile(r"(\*\*|__|\*|_)(.*?)\1")
_MD_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", flags=re.MULTILINE)
_MD_HR_RE = re.compile(r"^\s*---+\s*$", flags=re.MULTILINE)

# choice extraction
_CHOICE_LETTER_RE = re.compile(
    r"(?:^|\b)(?:option\s*)?([A-E])(?:\b|[\.\:\)\]])",
    flags=re.IGNORECASE
)
# also tolerate like "**C.**" or "C)" or "(C)"
_CHOICE_PAREN_RE = re.compile(r"[\(\[]\s*([A-E])\s*[\)\]]", flags=re.IGNORECASE)

# remove emojis / non-ascii symbols (keep basic punctuation)
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")

# strip weird quotes at ends
_EDGE_QUOTES_RE = re.compile(r'^[\'"“”‘’]+|[\'"“”‘’]+$')


# ==========================================================
# Debug helpers
# ==========================================================
def _now_unix() -> int:
    return int(time.time())


def _dbg(title: str, obj: Any):
    if not DEBUG_THINKING_CLEAN:
        return
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = repr(obj)
    if len(s) > DEBUG_THINKING_CLEAN_MAXLEN:
        s = s[:DEBUG_THINKING_CLEAN_MAXLEN] + "\n...<truncated>..."
    print(f"\n[DEBUG_THINKING_CLEAN] {title}\n{s}\n")


def _truncate(x: Any, maxlen: int) -> Any:
    try:
        s = json.dumps(x, ensure_ascii=False)
    except Exception:
        s = repr(x)
    if len(s) > maxlen:
        return s[:maxlen] + "...<truncated>"
    return s


def _log_jsonl(event: str, payload: Dict[str, Any]):
    """
    Append one JSON line to DEBUG_THINKING_CLEAN_LOGFILE.
    Ensures parent directory exists.
    Never raises to request path.
    """
    if (not DEBUG_THINKING_CLEAN) or (not DEBUG_THINKING_CLEAN_LOGFILE):
        return

    rec = {"ts": time.time(), "event": event, "payload": payload}
    try:
        line = json.dumps(rec, ensure_ascii=False)
    except Exception:
        line = json.dumps({"ts": time.time(), "event": event, "payload": repr(payload)}, ensure_ascii=False)

    path = DEBUG_THINKING_CLEAN_LOGFILE

    # ensure dir exists
    try:
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"[DEBUG_THINKING_CLEAN][WARN] cannot create log dir for {path}: {e}")
        print(line)
        return

    with _LOG_LOCK:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"[DEBUG_THINKING_CLEAN][WARN] cannot write log file {path}: {e}")
            print(line)


# ==========================================================
# Thinking strip
# ==========================================================
def _should_strip(text: str) -> bool:
    if STRIP_MODE == "keep":
        return False
    if STRIP_MODE == "strip":
        return True
    tl = (text or "").lower()
    return ("<think>" in tl) or ("</think>" in tl) or ("final:" in tl)


def _strip_thinking(text: str) -> str:
    """
    Robust stripping strategy:

    1) If </think> exists, keep ONLY content AFTER the LAST </think>.
    2) Else remove <think>...</think> blocks.
    3) Else if 'Final:' exists, keep content after it.
    4) Else if stray <think>, keep content after it.
    5) Else return original.
    """
    t = (text or "")

    closes = list(_THINK_CLOSE_RE.finditer(t))
    if closes:
        tail = t[closes[-1].end():].strip()
        if tail:
            return tail
        t = _THINK_CLOSE_RE.sub("", t)

    t2 = _THINK_BLOCK_RE.sub("", t).strip()
    if t2 and t2 != t.strip():
        t = t2

    m_final = _FINAL_MARK_RE.search(t)
    if m_final:
        return t[m_final.end():].strip()

    m_open = _THINK_OPEN_RE.search(t)
    if m_open:
        return t[m_open.end():].strip()

    return t.strip()


# ==========================================================
# Benchmark-safe cleanup
# ==========================================================
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
    t = text.strip()

    # Prefer the last occurrence (models often mention multiple options)
    hits: List[str] = []
    for m in _CHOICE_PAREN_RE.finditer(t):
        hits.append(m.group(1).upper())
    for m in _CHOICE_LETTER_RE.finditer(t):
        hits.append(m.group(1).upper())

    if not hits:
        return None
    return hits[-1]


def _clean_markup_latex(text: str) -> str:
    t = text or ""

    # remove fenced code blocks
    t = _CODE_FENCE_RE.sub("", t)
    # remove latex blocks/inline
    t = _LATEX_BLOCK_RE.sub(" ", t)
    t = _LATEX_INLINE_RE.sub(" ", t)

    # remove headings, hr, blockquotes
    t = _MD_HEADING_RE.sub("", t)
    t = _MD_HR_RE.sub("", t)
    t = _MD_BLOCKQUOTE_RE.sub("", t)

    # remove inline code markers
    t = _INLINE_CODE_RE.sub(r"\1", t)

    # remove bold/italic markers but keep content
    t = _MD_BOLD_ITALIC_RE.sub(r"\2", t)

    # remove non-ascii blobs (emoji etc)
    t = _NON_ASCII_RE.sub(" ", t)

    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _benchmark_safe(text: str) -> str:
    """
    Transform model output into a parser-friendly short final answer.
    Priority:
      - auto / choice_only: if an A-E option letter exists, return it.
      - else: strip markdown/latex/emoji and keep short text.
    """
    t = (text or "").strip()

    mode = BENCHMARK_SAFE_MODE
    if mode not in ("auto", "choice_only", "clean_text"):
        mode = "auto"

    if mode in ("auto", "choice_only"):
        c = _extract_choice_letter(t)
        if c:
            return c
        if mode == "choice_only":
            return EMPTY_FALLBACK

    # clean_text
    t = _clean_markup_latex(t)

    # remove leading "Final Answer:" like prefixes
    t = re.sub(r"(?i)^\s*(final\s*answer|answer)\s*[:：]\s*", "", t).strip()

    # trim edge quotes
    t = _EDGE_QUOTES_RE.sub("", t).strip()

    # keep it short
    if len(t) > MAX_FINAL_CHARS:
        t = t[:MAX_FINAL_CHARS].rstrip()

    return _ensure_nonempty(t)


# ==========================================================
# Upstream call
# ==========================================================
def _force_upstream_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})
    out["model"] = UPSTREAM_MODEL
    return out


async def _call_upstream(payload: Dict[str, Any]) -> Dict[str, Any]:
    async with SEM:
        global _CLIENT
        if _CLIENT is None:
            timeout = httpx.Timeout(
                connect=TIMEOUT_CONNECT_S,
                read=TIMEOUT_READ_S,
                write=TIMEOUT_WRITE_S,
                pool=TIMEOUT_POOL_S,
            )
            limits = httpx.Limits(
                max_connections=HTTPX_MAX_CONNECTIONS,
                max_keepalive_connections=HTTPX_MAX_KEEPALIVE,
            )
            _CLIENT = httpx.AsyncClient(timeout=timeout, limits=limits)

        last_exc: Optional[Exception] = None

        for attempt in range(UPSTREAM_MAX_RETRIES + 1):
            try:
                r = await _CLIENT.post(f"{UPSTREAM_BASE}{UPSTREAM_PATH}", json=payload)

                try:
                    body = r.json()
                except Exception:
                    body = {"_non_json": r.text}

                _dbg("8003 -> upstream response", {"status_code": r.status_code, "body": body})
                if DEBUG_THINKING_CLEAN and DEBUG_THINKING_CLEAN_LOG_RESPONSE:
                    _log_jsonl(
                        "upstream_response",
                        {"status_code": int(r.status_code), "body": _truncate(body, DEBUG_THINKING_CLEAN_MAXLEN)},
                    )

                if r.status_code >= 400:
                    return {"_upstream_error": True, "status_code": r.status_code, "body": body}

                return body

            except httpx.ReadTimeout as e:
                last_exc = e
                if DEBUG_THINKING_CLEAN:
                    _log_jsonl("upstream_timeout", {"attempt": attempt, "type": "ReadTimeout", "message": str(e)})
                if attempt < UPSTREAM_MAX_RETRIES:
                    await asyncio.sleep(UPSTREAM_RETRY_BACKOFF_S * (2 ** attempt))
                    continue
                break

            except httpx.ConnectTimeout as e:
                last_exc = e
                if DEBUG_THINKING_CLEAN:
                    _log_jsonl("upstream_timeout", {"attempt": attempt, "type": "ConnectTimeout", "message": str(e)})
                if attempt < UPSTREAM_MAX_RETRIES:
                    await asyncio.sleep(UPSTREAM_RETRY_BACKOFF_S * (2 ** attempt))
                    continue
                break

            except httpx.HTTPError as e:
                # other transient network errors
                last_exc = e
                if DEBUG_THINKING_CLEAN:
                    _log_jsonl("upstream_http_error", {"attempt": attempt, "type": type(e).__name__, "message": str(e)})
                if attempt < UPSTREAM_MAX_RETRIES:
                    await asyncio.sleep(UPSTREAM_RETRY_BACKOFF_S * (2 ** attempt))
                    continue
                break

        # retries exhausted -> return structured upstream_error
        return {
            "_upstream_error": True,
            "status_code": 504,
            "body": {
                "error": "upstream_timeout",
                "exception": type(last_exc).__name__ if last_exc else "unknown",
                "message": str(last_exc) if last_exc else "unknown",
                "upstream_base": UPSTREAM_BASE,
                "upstream_path": UPSTREAM_PATH,
            },
        }



def _rewrite_model_name(resp: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(resp, dict):
        resp["model"] = PROXY_MODEL
    return resp


def _force_minimal_openai_response() -> Dict[str, Any]:
    t = _now_unix()
    return {
        "id": f"chatcmpl-{t}",
        "object": "chat.completion",
        "created": t,
        "model": PROXY_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": EMPTY_FALLBACK}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _process_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Guarantees:
      - OpenAI chat.completion schema
      - choices[0].message.content exists and is non-empty string
      - optional thinking stripping
      - optional benchmark-safe answer normalization
    """
    if not isinstance(resp, dict):
        out = _force_minimal_openai_response()
        return out

    choices = resp.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        out = _force_minimal_openai_response()
        return out

    for c in choices:
        if not isinstance(c, dict):
            continue

        msg = c.get("message")
        if not isinstance(msg, dict):
            c["message"] = {"role": "assistant", "content": EMPTY_FALLBACK}
            msg = c["message"]

        content = msg.get("content", "")
        original = content if isinstance(content, str) else ("" if content is None else str(content))

        # 1) strip thinking
        if isinstance(original, str) and _should_strip(original):
            stripped = _strip_thinking(original)
            if not stripped.strip():
                # salvage: remove only tags and keep rest
                salvage = _THINK_OPEN_RE.sub("", original)
                salvage = _THINK_CLOSE_RE.sub("", salvage)
                stripped = salvage.strip()
            original = stripped

        # 2) ensure non-empty
        original = _ensure_nonempty(original)

        # 3) benchmark-safe cleanup (MMMU-friendly)
        if BENCHMARK_SAFE:
            cleaned = _benchmark_safe(original)
            msg["content"] = cleaned
        else:
            msg["content"] = original

        # normalize finish_reason if missing
        if "finish_reason" not in c:
            c["finish_reason"] = "stop"

    return resp


# ==========================================================
# API
# ==========================================================
@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    payload = await req.json()

    if DEBUG_THINKING_CLEAN and DEBUG_THINKING_CLEAN_LOG_REQUEST:
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
        _dbg("8003 upstream error -> client", out)
        if DEBUG_THINKING_CLEAN and DEBUG_THINKING_CLEAN_LOG_RESPONSE:
            _log_jsonl(
                "proxy_error_response",
                {"status_code": int(upstream_resp.get("status_code", 502)), "body": _truncate(out, DEBUG_THINKING_CLEAN_MAXLEN)},
            )
        return JSONResponse(status_code=int(upstream_resp.get("status_code", 502)), content=out)

    out = _process_response(upstream_resp)
    out = _rewrite_model_name(out)

    # log final response summary
    if DEBUG_THINKING_CLEAN and DEBUG_THINKING_CLEAN_LOG_RESPONSE:
        try:
            c0 = out.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            c0 = ""
        _log_jsonl(
            "proxy_response",
            {"status_code": 200, "model": out.get("model"), "content": _truncate(c0, DEBUG_THINKING_CLEAN_MAXLEN)},
        )

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
        "strip_mode": STRIP_MODE,
        "timeout_s": TIMEOUT_S,
        "concurrency": CONCURRENCY,
        "empty_fallback": EMPTY_FALLBACK,
        "benchmark_safe": BENCHMARK_SAFE,
        "benchmark_safe_mode": BENCHMARK_SAFE_MODE,
        "max_final_chars": MAX_FINAL_CHARS,
        "debug": {
            "enabled": DEBUG_THINKING_CLEAN,
            "maxlen": DEBUG_THINKING_CLEAN_MAXLEN,
            "logfile": DEBUG_THINKING_CLEAN_LOGFILE,
            "log_request": DEBUG_THINKING_CLEAN_LOG_REQUEST,
            "log_response": DEBUG_THINKING_CLEAN_LOG_RESPONSE,
        },
        "upstream": {"base": UPSTREAM_BASE, "path": UPSTREAM_PATH, "upstream_model": UPSTREAM_MODEL},
    }
