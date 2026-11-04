# run/sap_clean.py
import os
import io
import json
import base64
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

SAP_BASE = os.environ.get("SAP_BASE", "http://127.0.0.1:8030").rstrip("/")
SAP_VQA_PATH = os.environ.get("SAP_VQA_PATH", "/vqa").strip()

PROXY_MODEL = os.environ.get("PROXY_MODEL", "SAPGroundedVQA-8040")

SAP_INIT_PRINCIPLES_N = int(os.environ.get("SAP_INIT_PRINCIPLES_N", "-1"))
SAP_RETURN_TRACES = int(os.environ.get("SAP_RETURN_TRACES", "0"))

INCLUDE_REASONS = int(os.environ.get("INCLUDE_REASONS", "0"))

DEBUG_SAP_CLEAN = int(os.environ.get("DEBUG_SAP_CLEAN", "0"))
DEBUG_SAP_CLEAN_MAXLEN = int(os.environ.get("DEBUG_SAP_CLEAN_MAXLEN", "1000"))
DEBUG_SAP_CLEAN_LOGFILE = os.environ.get("DEBUG_SAP_CLEAN_LOGFILE", "./tmp/sap_clean_debug.jsonl")
DEBUG_SAP_CLEAN_LOG_REQUEST = int(os.environ.get("DEBUG_SAP_CLEAN_LOG_REQUEST", "1"))
DEBUG_SAP_CLEAN_LOG_RESPONSE = int(os.environ.get("DEBUG_SAP_CLEAN_LOG_RESPONSE", "1"))

# multi-image stitch (optional)
STITCH_MAX_SIDE = int(os.environ.get("STITCH_MAX_SIDE", "1536"))
STITCH_BG = os.environ.get("STITCH_BG", "black")  # black|white
# if >0: when multiple images are provided, stitch into one image and send as single image
STITCH_MULTI_IMAGE = int(os.environ.get("STITCH_MULTI_IMAGE", "0"))

EMPTY_FALLBACK = os.environ.get("EMPTY_FALLBACK", "unknown")

BENCHMARK_SAFE = int(os.environ.get("BENCHMARK_SAFE", "1"))
MAX_FINAL_CHARS = int(os.environ.get("MAX_FINAL_CHARS", "128"))

CONCURRENCY = int(os.environ.get("CONCURRENCY", "128"))
TIMEOUT_CONNECT_S = float(os.environ.get("TIMEOUT_CONNECT_S", "10"))
TIMEOUT_READ_S = float(os.environ.get("TIMEOUT_READ_S", "900"))
TIMEOUT_WRITE_S = float(os.environ.get("TIMEOUT_WRITE_S", "30"))
TIMEOUT_POOL_S = float(os.environ.get("TIMEOUT_POOL_S", "10"))

UPSTREAM_MAX_RETRIES = int(os.environ.get("UPSTREAM_MAX_RETRIES", "2"))
UPSTREAM_RETRY_BACKOFF_S = float(os.environ.get("UPSTREAM_RETRY_BACKOFF_S", "0.6"))

app = FastAPI(title="sap_clean: OpenAI-compatible wrapper over SAP")
_SEM = asyncio.Semaphore(CONCURRENCY)
_CLIENT: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def _startup():
    global _CLIENT
    _CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=TIMEOUT_CONNECT_S,
            read=TIMEOUT_READ_S,
            write=TIMEOUT_WRITE_S,
            pool=TIMEOUT_POOL_S,
        )
    )


@app.on_event("shutdown")
async def _shutdown():
    global _CLIENT
    if _CLIENT is not None:
        await _CLIENT.aclose()
    _CLIENT = None


def _safe_trunc(s: str, n: int) -> str:
    s = (s or "").strip()
    if not s:
        return EMPTY_FALLBACK
    if n > 0 and len(s) > n:
        return s[:n]
    return s


def _dbg_write(obj: Dict[str, Any], tag: str):
    if DEBUG_SAP_CLEAN <= 0:
        return
    try:
        line = json.dumps({"tag": tag, **obj}, ensure_ascii=False)
        if DEBUG_SAP_CLEAN_MAXLEN > 0:
            line = line[:DEBUG_SAP_CLEAN_MAXLEN]
        os.makedirs(os.path.dirname(DEBUG_SAP_CLEAN_LOGFILE), exist_ok=True)
        with open(DEBUG_SAP_CLEAN_LOGFILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _extract_text_question(messages: List[Dict[str, Any]]) -> str:
    # take the last user text as question
    q = ""
    for m in messages or []:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            q = c
        elif isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    q = str(blk.get("text", ""))
    q = (q or "").strip()
    return q if q else "unknown"


def _extract_image_data_urls(messages: List[Dict[str, Any]]) -> List[str]:
    urls = []
    for m in messages or []:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, list):
            for blk in c:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") == "image_url":
                    u = blk.get("image_url", {}).get("url", "")
                    if isinstance(u, str) and u.startswith("data:image"):
                        urls.append(u)
    return urls


def _data_url_to_pil(data_url: str) -> Image.Image:
    # data:image/png;base64,...
    head, b64 = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = float(max_side) / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), resample=Image.BICUBIC)


def _stitch_images(images: List[Image.Image], max_side: int, bg: str) -> Image.Image:
    # simple horizontal stitch; scale each to same height by max_side cap
    imgs = [_resize_max_side(im, max_side=max_side) for im in images]
    heights = [im.size[1] for im in imgs]
    target_h = max(heights) if heights else 1

    resized = []
    for im in imgs:
        w, h = im.size
        if h == target_h:
            resized.append(im)
            continue
        scale = float(target_h) / float(max(1, h))
        nw = max(1, int(round(w * scale)))
        resized.append(im.resize((nw, target_h), resample=Image.BICUBIC))

    total_w = sum(im.size[0] for im in resized)
    bgc = (0, 0, 0) if bg.lower() == "black" else (255, 255, 255)
    canvas = Image.new("RGB", (total_w, target_h), color=bgc)

    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.size[0]

    # cap again
    canvas = _resize_max_side(canvas, max_side=max_side)
    return canvas


async def _call_sap(images_png_bytes: List[bytes], question: str, init_principles_n: int, return_traces: bool) -> Dict[str, Any]:
    global _CLIENT
    if _CLIENT is None:
        await _startup()

    data = {
        "question": question,
        "return_traces": "true" if return_traces else "false",
    }
    if init_principles_n is not None and int(init_principles_n) > 0:
        data["init_principles_n"] = str(int(init_principles_n))

    files = []
    if len(images_png_bytes) <= 1:
        files.append(("image", ("image.png", images_png_bytes[0], "image/png")))
    else:
        for k, b in enumerate(images_png_bytes):
            files.append(("images", (f"image_{k}.png", b, "image/png")))

    last_err = None
    for attempt in range(max(1, UPSTREAM_MAX_RETRIES + 1)):
        try:
            async with _SEM:
                r = await _CLIENT.post(f"{SAP_BASE}{SAP_VQA_PATH}", data=data, files=files)
            if r.status_code >= 400:
                raise RuntimeError(f"SAP HTTP {r.status_code}: {r.text[:4000]}")
            return r.json()
        except Exception as e:
            last_err = e
            if attempt >= UPSTREAM_MAX_RETRIES:
                break
            await asyncio.sleep(float(UPSTREAM_RETRY_BACKOFF_S))
    raise RuntimeError(f"Upstream SAP failed: {type(last_err).__name__}: {last_err}")


def _openai_response(content: str, model_name: str) -> Dict[str, Any]:
    # minimal OpenAI chat.completions response
    return {
        "id": "sap_clean_dummy",
        "object": "chat.completion",
        "created": 0,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "proxy": "sap_clean",
        "sap_base": SAP_BASE,
        "sap_vqa_path": SAP_VQA_PATH,
        "proxy_model": PROXY_MODEL,
        "stitch_multi_image": STITCH_MULTI_IMAGE,
        "stitch_max_side": STITCH_MAX_SIDE,
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    payload = await req.json()

    if DEBUG_SAP_CLEAN and DEBUG_SAP_CLEAN_LOG_REQUEST:
        _dbg_write({"req": payload}, tag="request")

    messages = payload.get("messages", []) or []
    question = _extract_text_question(messages)
    data_urls = _extract_image_data_urls(messages)

    if not data_urls:
        # no image: return fallback (or keep empty)
        out_txt = EMPTY_FALLBACK if BENCHMARK_SAFE else ""
        resp = _openai_response(out_txt, model_name=PROXY_MODEL)
        if DEBUG_SAP_CLEAN and DEBUG_SAP_CLEAN_LOG_RESPONSE:
            _dbg_write({"resp": resp}, tag="response_no_image")
        return JSONResponse(resp)

    # decode images
    pil_images = []
    for u in data_urls:
        try:
            pil_images.append(_data_url_to_pil(u))
        except Exception:
            continue

    if not pil_images:
        out_txt = EMPTY_FALLBACK if BENCHMARK_SAFE else ""
        resp = _openai_response(out_txt, model_name=PROXY_MODEL)
        if DEBUG_SAP_CLEAN and DEBUG_SAP_CLEAN_LOG_RESPONSE:
            _dbg_write({"resp": resp}, tag="response_decode_fail")
        return JSONResponse(resp)

    # stitch option
    if STITCH_MULTI_IMAGE > 0 and len(pil_images) > 1:
        stitched = _stitch_images(pil_images, max_side=STITCH_MAX_SIDE, bg=STITCH_BG)
        images_png = [_pil_to_png_bytes(stitched)]
    else:
        images_png = [_pil_to_png_bytes(_resize_max_side(im, STITCH_MAX_SIDE)) for im in pil_images]

    init_n = int(SAP_INIT_PRINCIPLES_N)
    if init_n <= 0:
        init_n = None

    return_traces = bool(SAP_RETURN_TRACES > 0)

    # call SAP
    try:
        sap_out = await _call_sap(images_png, question=question, init_principles_n=init_n, return_traces=return_traces)
        final_answer = str(sap_out.get("final_answer", "")).strip()
        final_unc = str(sap_out.get("final_uncertainty", "")).strip()
        final_reasons = sap_out.get("final_reasons", []) or []

        if not final_answer:
            final_answer = EMPTY_FALLBACK

        # formatting
        if INCLUDE_REASONS > 0 and isinstance(final_reasons, list) and final_reasons:
            rs = "; ".join([str(x).strip() for x in final_reasons if str(x).strip()][:4])
            out_txt = f"{final_answer}\nReasons: {rs}\nUncertainty: {final_unc or 'high'}"
        else:
            out_txt = final_answer

        out_txt = _safe_trunc(out_txt, MAX_FINAL_CHARS if BENCHMARK_SAFE else -1)
        resp = _openai_response(out_txt, model_name=PROXY_MODEL)

        if DEBUG_SAP_CLEAN and DEBUG_SAP_CLEAN_LOG_RESPONSE:
            _dbg_write({"sap_out": sap_out, "resp": resp}, tag="response_ok")

        return JSONResponse(resp)

    except Exception as e:
        out_txt = EMPTY_FALLBACK if BENCHMARK_SAFE else ""
        out_txt = _safe_trunc(out_txt, MAX_FINAL_CHARS if BENCHMARK_SAFE else -1)
        resp = _openai_response(out_txt, model_name=PROXY_MODEL)
        if DEBUG_SAP_CLEAN and DEBUG_SAP_CLEAN_LOG_RESPONSE:
            _dbg_write({"error": f"{type(e).__name__}: {e}", "resp": resp}, tag="response_err")
        return JSONResponse(resp)
