# run/proxy_qwen_sam.py
import os
import io
import re
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from .common import b64_to_pil, pil_to_data_url

SAM_BASE = os.environ.get("SAM_BASE", "http://127.0.0.1:8010").rstrip("/")
QWEN_BASE = os.environ.get("QWEN_BASE", "http://127.0.0.1:8001").rstrip("/")
QWEN_PATH = os.environ.get("QWEN_PATH", "/v1/chat/completions").strip()
QWEN_MODEL = os.environ.get("QWEN_MODEL", "").strip()

QWEN_TIMEOUT = float(os.environ.get("QWEN_TIMEOUT", "120"))
QWEN_TEMPERATURE = float(os.environ.get("QWEN_TEMPERATURE", "0.0"))
QWEN_MAX_TOKENS = int(os.environ.get("QWEN_MAX_TOKENS", "16"))

BBOX_PAD = int(os.environ.get("BBOX_PAD", "16"))
BG_MODE = os.environ.get("BG_MODE", "dark")  # dark|black|keep
LABEL_FALLBACK = os.environ.get("LABEL_FALLBACK", "object")
USE_MASKED_CROP = int(os.environ.get("USE_MASKED_CROP", "0"))

# keep API surface: batch | single
# - single: label each crop with one request (current behavior)
# - batch: keep the knob, but implementation still loops (safe, predictable)
QWEN_LABEL_MODE = os.environ.get("QWEN_LABEL_MODE", "single").strip().lower()
if QWEN_LABEL_MODE not in ("single", "batch"):
    QWEN_LABEL_MODE = "single"

app = FastAPI(title="proxy_qwen_sam: SAM(amg) + Qwen labels")
_session = requests.Session()

_label_re = re.compile(r"[^\w\s\-/+]+")


def clean_label(text: str, max_words: int = 3) -> str:
    s = (text or "").strip()
    s = s.splitlines()[0].strip()
    s = _label_re.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    parts = s.split(" ")
    return " ".join(parts[:max_words]).strip()


def pad_bbox_xyxy(b: Tuple[int, int, int, int], w: int, h: int, pad: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = b
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)
    return x0, y0, x1, y1


def masked_crop(orig_rgb: Image.Image, mask_l: Image.Image, bbox_xyxy: Tuple[int, int, int, int], bg_mode: str) -> Image.Image:
    x0, y0, x1, y1 = bbox_xyxy
    crop = orig_rgb.crop((x0, y0, x1 + 1, y1 + 1))
    m = mask_l.crop((x0, y0, x1 + 1, y1 + 1)).convert("L")

    crop_np = np.asarray(crop).astype(np.float32)
    mm = (np.asarray(m) > 0)

    if bg_mode == "dark":
        crop_np[~mm] *= 0.15
    elif bg_mode == "black":
        crop_np[~mm] = 0.0

    crop_np = np.clip(crop_np, 0, 255).astype(np.uint8)
    return Image.fromarray(crop_np, mode="RGB")


def call_sam_amg(img_bytes: bytes, points_per_batch: int, max_masks: int, sort_by: str) -> Dict[str, Any]:
    files = {"image": ("image.png", img_bytes, "application/octet-stream")}
    data = {
        "points_per_batch": str(int(points_per_batch)),
        "max_masks": str(int(max_masks)),
        "sort_by": str(sort_by),
    }
    r = _session.post(f"{SAM_BASE}/amg", files=files, data=data, timeout=300)
    r.raise_for_status()
    return r.json()


def call_sam_amg_batch(img_bytes_list: List[bytes], points_per_batch: int, max_masks: int, sort_by: str) -> Dict[str, Any]:
    files = []
    for k, b in enumerate(img_bytes_list):
        files.append(("images", (f"image_{k}.png", b, "application/octet-stream")))
    data = {
        "points_per_batch": str(int(points_per_batch)),
        "max_masks": str(int(max_masks)),
        "sort_by": str(sort_by),
    }
    r = _session.post(f"{SAM_BASE}/amg_batch", files=files, data=data, timeout=600)
    r.raise_for_status()
    return r.json()  # {"items":[...]}


def qwen_label_crop(crop_rgb: Image.Image) -> str:
    data_url = pil_to_data_url(crop_rgb)
    payload = {
        "model": QWEN_MODEL if QWEN_MODEL else "dummy",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify the main object in this image. Return ONLY a short noun phrase."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": float(QWEN_TEMPERATURE),
        "max_tokens": int(QWEN_MAX_TOKENS),
    }
    r = _session.post(f"{QWEN_BASE}{QWEN_PATH}", json=payload, timeout=QWEN_TIMEOUT)
    r.raise_for_status()
    out = r.json()
    txt = out["choices"][0]["message"]["content"]
    return clean_label(txt)


def _label_one_image(img_bytes: bytes, points_per_batch: int, max_masks: int, sort_by: str) -> Dict[str, Any]:
    sam_json = call_sam_amg(img_bytes, points_per_batch, max_masks, sort_by)

    masks_b64 = sam_json.get("masks_png_base64", None)
    idx = sam_json.get("returned_indices", None)
    bboxes_all = sam_json.get("bboxes_xyxy", None)

    if not isinstance(masks_b64, list) or len(masks_b64) == 0:
        sam_json["qwen_labels"] = []
        sam_json["qwen_label_meta"] = {
            "bbox_pad": int(BBOX_PAD),
            "bg_mode": BG_MODE,
            "qwen_model": QWEN_MODEL,
            "use_masked_crop": int(USE_MASKED_CROP),
            "qwen_label_mode": QWEN_LABEL_MODE,
        }
        return sam_json

    try:
        orig = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        orig = b64_to_pil(sam_json["orig_png_base64"], "RGB")

    w, h = orig.size

    have_bboxes = isinstance(bboxes_all, list) and isinstance(idx, list) and len(idx) == len(masks_b64)
    qwen_labels: List[str] = []

    if have_bboxes:
        for j, i in enumerate(idx):
            try:
                bb = bboxes_all[i]
                if not (isinstance(bb, list) and len(bb) == 4):
                    qwen_labels.append(LABEL_FALLBACK)
                    continue

                x0, y0, x1, y1 = map(int, bb)
                x0, y0, x1, y1 = pad_bbox_xyxy((x0, y0, x1, y1), w, h, int(BBOX_PAD))

                if USE_MASKED_CROP and BG_MODE in ("dark", "black"):
                    mask_l = b64_to_pil(masks_b64[j], "L")
                    crop = masked_crop(orig, mask_l, (x0, y0, x1, y1), bg_mode=BG_MODE)
                else:
                    crop = orig.crop((x0, y0, x1 + 1, y1 + 1))

                lab = qwen_label_crop(crop)
                qwen_labels.append(lab if lab else LABEL_FALLBACK)
            except Exception:
                qwen_labels.append(LABEL_FALLBACK)
    else:
        # fallback: bbox from decoded mask (slow)
        for mb in masks_b64:
            try:
                mask_l = b64_to_pil(mb, "L")
                m = (np.asarray(mask_l.convert("L")) > 0)
                ys, xs = np.where(m)
                if xs.size == 0:
                    qwen_labels.append(LABEL_FALLBACK)
                    continue
                bb = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                x0, y0, x1, y1 = pad_bbox_xyxy(bb, w, h, int(BBOX_PAD))

                if USE_MASKED_CROP and BG_MODE in ("dark", "black"):
                    crop = masked_crop(orig, mask_l, (x0, y0, x1, y1), bg_mode=BG_MODE)
                else:
                    crop = orig.crop((x0, y0, x1 + 1, y1 + 1))

                lab = qwen_label_crop(crop)
                qwen_labels.append(lab if lab else LABEL_FALLBACK)
            except Exception:
                qwen_labels.append(LABEL_FALLBACK)

    sam_json["qwen_labels"] = qwen_labels
    # trimmed meta (keep only what's useful for debugging / reproducibility)
    sam_json["qwen_label_meta"] = {
        "bbox_pad": int(BBOX_PAD),
        "bg_mode": BG_MODE,
        "use_masked_crop": int(USE_MASKED_CROP),
        "qwen_model": QWEN_MODEL,
        "qwen_label_mode": QWEN_LABEL_MODE,
    }
    return sam_json


@app.get("/health")
def health():
    return {
        "status": "ok",
        "sam_base": SAM_BASE,
        "qwen_base": QWEN_BASE,
        "qwen_model": QWEN_MODEL,
        "qwen_label_mode": QWEN_LABEL_MODE,
    }


@app.post("/amg")
async def amg_proxy(
    image: UploadFile = File(...),
    points_per_batch: int = Form(64),
    max_masks: int = Form(12),
    sort_by: str = Form("area"),
):
    img_bytes = await image.read()
    out = _label_one_image(img_bytes, int(points_per_batch), int(max_masks), str(sort_by))
    return JSONResponse(out)


@app.post("/amg_batch")
async def amg_proxy_batch(
    images: List[UploadFile] = File(...),
    points_per_batch: int = Form(64),
    max_masks: int = Form(12),
    sort_by: str = Form("area"),
):
    """
    Batch proxy:
      input: images[]
      output: { "items": [ <same-as-/amg>, ... ] }
    """
    img_bytes_list = [await im.read() for im in images]

    # AMG batch from SAM (optional speed), then label each item.
    # Note: labeling still needs per-crop Qwen calls; this endpoint batches only the SAM stage + HTTP overhead.
    try:
        sam_batch = call_sam_amg_batch(img_bytes_list, int(points_per_batch), int(max_masks), str(sort_by))
        items = sam_batch.get("items", None)
        if not isinstance(items, list) or len(items) != len(img_bytes_list):
            raise RuntimeError("SAM /amg_batch returned unexpected shape.")
        out_items = []
        # reuse per-item labeling but avoid second SAM call by injecting sam_json
        for k, (img_bytes, sam_json) in enumerate(zip(img_bytes_list, items)):
            # patch path: run labeling using the already-produced sam_json
            # (we re-run minimal parts of _label_one_image but without calling SAM again)
            masks_b64 = sam_json.get("masks_png_base64", None)
            idx = sam_json.get("returned_indices", None)
            bboxes_all = sam_json.get("bboxes_xyxy", None)

            if not isinstance(masks_b64, list) or len(masks_b64) == 0:
                sam_json["qwen_labels"] = []
                sam_json["qwen_label_meta"] = {
                    "bbox_pad": int(BBOX_PAD),
                    "bg_mode": BG_MODE,
                    "qwen_model": QWEN_MODEL,
                    "use_masked_crop": int(USE_MASKED_CROP),
                    "qwen_label_mode": QWEN_LABEL_MODE,
                }
                out_items.append(sam_json)
                continue

            try:
                orig = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                orig = b64_to_pil(sam_json["orig_png_base64"], "RGB")
            w, h = orig.size

            have_bboxes = isinstance(bboxes_all, list) and isinstance(idx, list) and len(idx) == len(masks_b64)
            qwen_labels: List[str] = []

            if have_bboxes:
                for j, i in enumerate(idx):
                    try:
                        bb = bboxes_all[i]
                        if not (isinstance(bb, list) and len(bb) == 4):
                            qwen_labels.append(LABEL_FALLBACK)
                            continue
                        x0, y0, x1, y1 = map(int, bb)
                        x0, y0, x1, y1 = pad_bbox_xyxy((x0, y0, x1, y1), w, h, int(BBOX_PAD))
                        if USE_MASKED_CROP and BG_MODE in ("dark", "black"):
                            mask_l = b64_to_pil(masks_b64[j], "L")
                            crop = masked_crop(orig, mask_l, (x0, y0, x1, y1), bg_mode=BG_MODE)
                        else:
                            crop = orig.crop((x0, y0, x1 + 1, y1 + 1))
                        lab = qwen_label_crop(crop)
                        qwen_labels.append(lab if lab else LABEL_FALLBACK)
                    except Exception:
                        qwen_labels.append(LABEL_FALLBACK)
            else:
                for mb in masks_b64:
                    try:
                        mask_l = b64_to_pil(mb, "L")
                        m = (np.asarray(mask_l.convert("L")) > 0)
                        ys, xs = np.where(m)
                        if xs.size == 0:
                            qwen_labels.append(LABEL_FALLBACK)
                            continue
                        bb = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                        x0, y0, x1, y1 = pad_bbox_xyxy(bb, w, h, int(BBOX_PAD))
                        if USE_MASKED_CROP and BG_MODE in ("dark", "black"):
                            crop = masked_crop(orig, mask_l, (x0, y0, x1, y1), bg_mode=BG_MODE)
                        else:
                            crop = orig.crop((x0, y0, x1 + 1, y1 + 1))
                        lab = qwen_label_crop(crop)
                        qwen_labels.append(lab if lab else LABEL_FALLBACK)
                    except Exception:
                        qwen_labels.append(LABEL_FALLBACK)

            sam_json["qwen_labels"] = qwen_labels
            sam_json["qwen_label_meta"] = {
                "bbox_pad": int(BBOX_PAD),
                "bg_mode": BG_MODE,
                "use_masked_crop": int(USE_MASKED_CROP),
                "qwen_model": QWEN_MODEL,
                "qwen_label_mode": QWEN_LABEL_MODE,
            }
            out_items.append(sam_json)

        return JSONResponse({"items": out_items})

    except Exception:
        # conservative fallback: call /amg logic per image (still returns batch shape)
        out_items = []
        for b in img_bytes_list:
            out_items.append(_label_one_image(b, int(points_per_batch), int(max_masks), str(sort_by)))
        return JSONResponse({"items": out_items})
