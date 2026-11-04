import io
import re
import time
import base64
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def now_unix() -> int:
    return int(time.time())


def ensure_str_list(x, fallback=None) -> List[str]:
    if fallback is None:
        fallback = []
    if isinstance(x, list):
        out = []
        for t in x:
            s = str(t).strip()
            if s:
                out.append(s)
        return out
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else fallback
    return fallback


def normalize_answer(a: str) -> str:
    a = (a or "").strip().lower()
    a = re.sub(r"\s+", " ", a)
    return a if a else "unknown"


def clamp_uncertainty(u: str) -> str:
    u = (u or "high").strip().lower()
    return u if u in ["low", "medium", "high"] else "high"


def uncertainty_rank(u: str) -> int:
    u = clamp_uncertainty(u)
    return {"low": 0, "medium": 1, "high": 2}[u]


def b64_to_pil(b64: str, mode="RGB") -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert(mode)


def pil_to_png_bytes(pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def pil_to_data_url(pil: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(pil_to_png_bytes(pil)).decode("utf-8")


def data_url_to_bytes(url: str) -> Optional[bytes]:
    if not isinstance(url, str):
        return None
    if not url.startswith("data:image"):
        return None
    try:
        _, b64 = url.split(",", 1)
        return base64.b64decode(b64)
    except Exception:
        return None


def mask_bbox(mask_l: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    m = np.array(mask_l) > 0
    ys, xs = np.where(m)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def pad_bbox(b: Tuple[int, int, int, int], w: int, h: int, pad: int = 12) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = b
    return max(0, x0 - pad), max(0, y0 - pad), min(w - 1, x1 + pad), min(h - 1, y1 + pad)


def masked_crop(img_rgb: Image.Image, mask_l: Image.Image, bbox_xyxy: Tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = bbox_xyxy
    crop = img_rgb.crop((x0, y0, x1 + 1, y1 + 1))
    mask_c = mask_l.crop((x0, y0, x1 + 1, y1 + 1))
    arr = np.array(crop).astype(np.float32)
    m = np.array(mask_c) > 0
    arr[~m] *= 0.15
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_FINAL_MARK_RE = re.compile(r"\bfinal\s*:\s*", flags=re.IGNORECASE)


def should_strip_thinking(text: str, mode: str) -> bool:
    mode = (mode or "auto").strip().lower()
    if mode == "keep":
        return False
    if mode == "strip":
        return True
    tl = (text or "").lower()
    return ("<think>" in tl) or ("final:" in tl)


def strip_thinking_text(text: str) -> str:
    t = text or ""
    t = _THINK_TAG_RE.sub("", t)
    m = _FINAL_MARK_RE.search(t)
    if m:
        t = t[m.end():]
    return t.strip()


def openai_models_list(model_id: str) -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model", "created": now_unix(), "owned_by": "proxy"}],
    }
