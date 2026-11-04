# run/sam_amg.py
import io
import os
import base64
import time
from typing import Any, Optional, List, Dict

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from transformers import pipeline

MODEL_ID = os.environ.get("MODEL_ID", "checkpoints/sam2.1-hiera-large")
DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))
DEFAULT_POINTS_PER_BATCH = int(os.environ.get("POINTS_PER_BATCH", "64"))

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

gen = pipeline("mask-generation", model=MODEL_ID, device=DEVICE_ID)
app = FastAPI(title="SAM 2.1 FastAPI (AMG returns masks + bboxes)")


# -----------------------------
# IO helpers
# -----------------------------
def to_numpy_uint8_hwc(x: Any) -> np.ndarray:
    if isinstance(x, Image.Image):
        arr = np.array(x)
    elif torch.is_tensor(x):
        t = x.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):
            t = t.permute(1, 2, 0)
        arr = t.numpy()
    elif isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
    else:
        raise TypeError(f"Unsupported image type: {type(x)}")

    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D image array, got shape {arr.shape}")

    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.floating):
        mx = float(np.nanmax(arr)) if arr.size else 0.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        if mx <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def to_pil_image(x: Any, mode: Optional[str] = None) -> Image.Image:
    if isinstance(x, Image.Image):
        img = x
    else:
        arr = to_numpy_uint8_hwc(x)
        if arr.shape[-1] == 1:
            img = Image.fromarray(arr[..., 0], mode="L")
        else:
            img = Image.fromarray(arr)
    if mode is not None:
        img = img.convert(mode)
    return img


def image_to_png_bytes(x: Any) -> bytes:
    img = to_pil_image(x)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def image_to_b64_png(x: Any) -> str:
    return base64.b64encode(image_to_png_bytes(x)).decode("utf-8")


def _ensure_mask_matches_image(mask_pil: Image.Image, img_pil: Image.Image) -> Image.Image:
    mask_pil = mask_pil.convert("L")
    if mask_pil.size != img_pil.size:
        mask_pil = mask_pil.resize(img_pil.size, resample=Image.NEAREST)
    return mask_pil


# -----------------------------
# AMG core
# -----------------------------
def _scores_to_list(scores: Any, n: int) -> Optional[List[float]]:
    if scores is None:
        return None
    try:
        if torch.is_tensor(scores):
            s = scores.detach().cpu().float().numpy()
        else:
            s = np.asarray(scores, dtype=np.float32)
        s = np.ravel(s)
        if s.size == 0:
            return None
        m = min(int(s.size), int(n))
        out = [float(x) for x in s[:m].tolist()]
        if m < n:
            out += [float("nan")] * (n - m)
        return out
    except Exception:
        return None


def mask_area(mask: Any, ref_size: Optional[tuple] = None) -> int:
    m_pil = to_pil_image(mask)
    if ref_size is not None:
        m_pil = _ensure_mask_matches_image(m_pil, Image.new("RGB", ref_size))
    else:
        m_pil = m_pil.convert("L")
    m = np.array(m_pil)
    return int((m > 0).sum())


def mask_to_bbox_xyxy(mask: Any, img_pil: Image.Image) -> Optional[List[int]]:
    m_pil = to_pil_image(mask)
    m_pil = _ensure_mask_matches_image(m_pil, img_pil)
    m = (np.asarray(m_pil) > 0)
    ys, xs = np.where(m)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def pick_top(scores: Any, n_masks: int) -> int:
    if n_masks <= 0:
        return 0
    if scores is None:
        return 0
    try:
        if torch.is_tensor(scores):
            s = scores.detach().cpu().float().numpy()
        else:
            s = np.asarray(scores, dtype=np.float32)
    except Exception:
        return 0

    s = np.ravel(s)
    if s.size == 0:
        return 0

    m = min(int(s.size), int(n_masks))
    if m <= 0:
        return 0

    top = int(np.argmax(s[:m]))
    if top < 0:
        top = 0
    if top >= n_masks:
        top = n_masks - 1
    return top


def run_amg(img: Image.Image, points_per_batch: int):
    img = to_pil_image(img, mode="RGB")
    with torch.inference_mode():
        out = gen(img, points_per_batch=int(points_per_batch))
    masks = out.get("masks", []) or []
    scores = out.get("scores", None)
    labels = out.get("labels", None)
    top = pick_top(scores, n_masks=len(masks))
    return masks, scores, labels, top


def _amg_from_pil(img_pil: Image.Image, points_per_batch: int, max_masks: int, sort_by: str) -> Dict[str, Any]:
    t0 = time.time()
    img_pil = img_pil.convert("RGB")

    masks, scores_raw, labels_raw, top = run_amg(img_pil, points_per_batch=int(points_per_batch))
    bboxes_all: List[Optional[List[int]]] = [mask_to_bbox_xyxy(m, img_pil) for m in masks]

    if len(masks) == 0:
        return {
            "model": MODEL_ID,
            "device_id": DEVICE_ID,
            "points_per_batch": int(points_per_batch),
            "num_masks_total": 0,
            "num_masks_returned": 0,
            "returned_indices": [],
            "sort_by": sort_by,
            "top_index_original": None,
            "top_score_original": None,
            "orig_png_base64": image_to_b64_png(img_pil),
            "masks_png_base64": [],
            "areas": [],
            "scores": None,
            "labels": None,
            "bboxes_xyxy": [],
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    scores_list = _scores_to_list(scores_raw, n=len(masks))
    areas = [mask_area(m, ref_size=img_pil.size) for m in masks]

    idx = list(range(len(masks)))
    if sort_by == "score" and scores_list is not None:
        def sget(i):
            v = scores_list[i]
            return float(v) if np.isfinite(v) else float("-inf")
        idx.sort(key=sget, reverse=True)
    elif sort_by == "area":
        idx.sort(key=lambda i: areas[i], reverse=True)

    if max_masks is not None and int(max_masks) > 0:
        idx = idx[: int(max_masks)]

    masks_b64 = []
    for i in idx:
        m_pil = to_pil_image(masks[i])
        m_pil = _ensure_mask_matches_image(m_pil, img_pil)
        masks_b64.append(image_to_b64_png(m_pil))

    areas_out = [areas[i] for i in idx]
    scores_out = None
    if scores_list is not None:
        scores_out = []
        for i in idx:
            v = scores_list[i]
            scores_out.append(float(v) if np.isfinite(v) else float("nan"))

    labels_out = [labels_raw[i] for i in idx] if isinstance(labels_raw, list) else labels_raw

    top_score_original = None
    if scores_list is not None and 0 <= top < len(scores_list):
        v = scores_list[top]
        top_score_original = float(v) if np.isfinite(v) else None

    return {
        "model": MODEL_ID,
        "device_id": DEVICE_ID,
        "points_per_batch": int(points_per_batch),
        "num_masks_total": len(masks),
        "num_masks_returned": len(idx),
        "returned_indices": idx,
        "sort_by": sort_by,
        "top_index_original": int(top),
        "top_score_original": top_score_original,
        "orig_png_base64": image_to_b64_png(img_pil),
        "masks_png_base64": masks_b64,
        "areas": areas_out,
        "scores": scores_out,
        "labels": labels_out,
        "bboxes_xyxy": bboxes_all,   # full list, original order
        "elapsed_ms": int((time.time() - t0) * 1000),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device_id": DEVICE_ID,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


@app.post("/amg")
async def amg(
    image: UploadFile = File(...),
    points_per_batch: int = Form(DEFAULT_POINTS_PER_BATCH),
    max_masks: int = Form(-1),
    sort_by: str = Form("score"),
):
    """
    Single-image AMG.
    Compatible fields for downstream proxy:
      - masks_png_base64 (returned order)
      - returned_indices (indices into original masks list)
      - bboxes_xyxy (full list, original order)
      - orig_png_base64
    """
    img_bytes = await image.read()
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    resp = _amg_from_pil(img_pil, points_per_batch=int(points_per_batch), max_masks=int(max_masks), sort_by=str(sort_by))
    return JSONResponse(resp)


@app.post("/amg_batch")
async def amg_batch(
    images: List[UploadFile] = File(...),
    points_per_batch: int = Form(DEFAULT_POINTS_PER_BATCH),
    max_masks: int = Form(-1),
    sort_by: str = Form("score"),
):
    """
    Batch wrapper.
    Returns:
      { "items": [ <single-amg-json>, ... ] }
    """
    items = []
    for im in images:
        b = await im.read()
        img_pil = Image.open(io.BytesIO(b)).convert("RGB")
        items.append(_amg_from_pil(img_pil, points_per_batch=int(points_per_batch), max_masks=int(max_masks), sort_by=str(sort_by)))
    return JSONResponse({"items": items})
