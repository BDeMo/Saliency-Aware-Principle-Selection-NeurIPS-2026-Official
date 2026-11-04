# run/sap.py
#
# SAP Proxy v3 (Saliency-Aware Principle Selection) — DISCRETE LEVELS (MULTI-IMAGE)
# - Grounding: calls proxy_qwen_sam (/amg or /amg_batch) to get masks + qwen_labels + orig_png_base64 + bboxes.
# - LLM input: ONLY (grounding_summary + ORIGINAL images). No montage. No obj lines. No crops.
# - Evidence format: img#k_obj#j(label)
#
# Discretization:
#   consensus_level, diversity_level, uncertainty, evidence_level in {'low','medium','high'}.
#   Fitness maps levels internally (for sorting elites); traces expose levels only.

import io
import os
import json
import base64
import asyncio
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import httpx

from .template import (
    build_init_principles_prompt,
    build_nextgen_principles_prompt,
    build_single_call_multipath_prompt_multi,
    build_final_aggregator_prompt,
)

# ==========================================================
# Env config
# ==========================================================
GroundingMaskWithLabel_URL = os.environ.get("GroundingMaskWithLabel_URL", "http://127.0.0.1:8020").rstrip("/")

QWEN_URL = os.environ.get("QWEN_URL", "http://127.0.0.1:8001").rstrip("/")
QWEN_CHAT_PATH = os.environ.get("QWEN_CHAT_PATH", "/v1/chat/completions").strip()
QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen3-VL-8B-Instruct").strip()

POINTS_PER_BATCH = int(os.environ.get("POINTS_PER_BATCH", "64"))
MAX_MASKS = int(os.environ.get("MAX_MASKS", "12"))

MU_PLUS_LAMBDA = int(os.environ.get("MU_PLUS_LAMBDA", os.environ.get("PRINCIPLES_N", "8")))
MU = int(os.environ.get("MU", os.environ.get("EVO_ELITES", "2")))
LAMBDA = int(os.environ.get("LAMBDA", str(max(0, MU_PLUS_LAMBDA - MU))))

EVO_GENS = int(os.environ.get("EVO_GENS", "2"))
TAU = int(os.environ.get("TAU", "2"))

QWEN_TIMEOUT = float(os.environ.get("QWEN_TIMEOUT", "120"))
QWEN_CONCURRENCY = int(os.environ.get("QWEN_CONCURRENCY", "14"))

CONS_W = int(float(os.environ.get("CONS_W", "1")))
DIV_W = int(float(os.environ.get("DIV_W", "1")))
UNC_W = int(float(os.environ.get("UNC_W", "1")))
EVID_W = int(float(os.environ.get("EVID_W", "1")))

# fitness-side diagnostics only (NOT sent to LLM)
MAX_OBJECT_TEXT = int(os.environ.get("MAX_OBJECT_TEXT", "24"))
MAX_IMAGES_TOPK = int(os.environ.get("MAX_IMAGES_TOPK", "8"))
BBOX_DECIMALS = int(os.environ.get("BBOX_DECIMALS", "3"))

USE_FINAL_AGGREGATOR = int(os.environ.get("USE_FINAL_AGGREGATOR", "1"))

# multi-image grounding mode: "single" (call /amg per image) or "batch" (call /amg_batch once)
GROUNDING_IMAGE_MODE = os.environ.get("GROUNDING_IMAGE_MODE", "single").strip().lower()
if GROUNDING_IMAGE_MODE not in ("single", "batch"):
    GROUNDING_IMAGE_MODE = "single"

app = FastAPI(title="SAP Proxy (discrete levels, multi-image)")
QWEN_SEM = asyncio.Semaphore(QWEN_CONCURRENCY)

_QWEN_CLIENT: Optional[httpx.AsyncClient] = None
_GROUND_CLIENT: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def _startup():
    global _QWEN_CLIENT, _GROUND_CLIENT
    _QWEN_CLIENT = httpx.AsyncClient(timeout=QWEN_TIMEOUT)
    _GROUND_CLIENT = httpx.AsyncClient(timeout=300)


@app.on_event("shutdown")
async def _shutdown():
    global _QWEN_CLIENT, _GROUND_CLIENT
    if _QWEN_CLIENT is not None:
        await _QWEN_CLIENT.aclose()
    if _GROUND_CLIENT is not None:
        await _GROUND_CLIENT.aclose()
    _QWEN_CLIENT = None
    _GROUND_CLIENT = None


# ==========================================================
# Helpers
# ==========================================================
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


def clamp_level(x: str, default: str = "low") -> str:
    t = (x or default).strip().lower()
    if t in ("median", "med", "mid"):
        t = "medium"
    return t if t in ("low", "medium", "high") else default


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def round_list(xs: List[float], nd: int) -> List[float]:
    return [round(float(x), nd) for x in xs]


def ratio_to_level(r: float) -> str:
    if r >= 0.75:
        return "high"
    if r >= 0.50:
        return "medium"
    return "low"


# ==========================================================
# Image utils
# ==========================================================
def b64_to_pil(b64: str, mode="RGB") -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert(mode)


def pil_png_bytes(pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def data_url_from_png_bytes(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def mask_bbox(mask_l: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    m = np.array(mask_l) > 0
    ys, xs = np.where(m)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ==========================================================
# HTTP calls
# ==========================================================
async def call_qwen(payload: Dict[str, Any]) -> Dict[str, Any]:
    global _QWEN_CLIENT
    async with QWEN_SEM:
        if _QWEN_CLIENT is None:
            _QWEN_CLIENT = httpx.AsyncClient(timeout=QWEN_TIMEOUT)
        r = await _QWEN_CLIENT.post(f"{QWEN_URL}{QWEN_CHAT_PATH}", json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"Qwen HTTP {r.status_code}: {r.text}")
        content = r.json()["choices"][0]["message"]["content"]

    txt = (content or "").strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        if "{" in txt and "}" in txt:
            txt = txt[txt.find("{") : txt.rfind("}") + 1]
    try:
        return json.loads(txt)
    except Exception:
        return {"_raw": (content or "").strip()}


async def call_grounding_amg_single(image_bytes: bytes) -> Dict[str, Any]:
    global _GROUND_CLIENT
    if _GROUND_CLIENT is None:
        _GROUND_CLIENT = httpx.AsyncClient(timeout=300)
    r = await _GROUND_CLIENT.post(
        f"{GroundingMaskWithLabel_URL}/amg",
        files={"image": ("img.png", image_bytes)},
        data={
            "points_per_batch": str(POINTS_PER_BATCH),
            "max_masks": str(MAX_MASKS),
            "sort_by": "area",
        },
    )
    r.raise_for_status()
    return r.json()


async def call_grounding_amg_batch(images_bytes: List[bytes]) -> List[Dict[str, Any]]:
    global _GROUND_CLIENT
    if _GROUND_CLIENT is None:
        _GROUND_CLIENT = httpx.AsyncClient(timeout=300)
    files = []
    for k, b in enumerate(images_bytes):
        files.append(("images", (f"img_{k}.png", b, "application/octet-stream")))
    r = await _GROUND_CLIENT.post(
        f"{GroundingMaskWithLabel_URL}/amg_batch",
        files=files,
        data={
            "points_per_batch": str(POINTS_PER_BATCH),
            "max_masks": str(MAX_MASKS),
            "sort_by": "area",
        },
    )
    r.raise_for_status()
    out = r.json()
    items = out.get("items", None)
    if not isinstance(items, list):
        raise RuntimeError("Grounding /amg_batch returned invalid shape (missing items list).")
    return items


# ==========================================================
# Grounding extraction (multi-image)
# ==========================================================
def extract_grounding_objects_single(
    grounding: Dict[str, Any],
    orig_rgb: Image.Image,
    img_idx: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    W, H = orig_rgb.size
    masks_b64 = grounding.get("masks_png_base64", []) or []
    scores = grounding.get("scores", None)
    areas = grounding.get("areas", None)
    qwen_labels = grounding.get("qwen_labels", None)

    bboxes_all = grounding.get("bboxes_xyxy", None)
    returned_idx = grounding.get("returned_indices", None)

    debug = {
        "img_idx": img_idx,
        "n_masks": len(masks_b64),
        "has_scores": isinstance(scores, list),
        "has_areas": isinstance(areas, list),
        "has_qwen_labels": isinstance(qwen_labels, list),
        "qwen_labels_len": (len(qwen_labels) if isinstance(qwen_labels, list) else None),
        "has_bboxes_xyxy": isinstance(bboxes_all, list),
        "has_returned_indices": isinstance(returned_idx, list),
    }

    objs: List[Dict[str, Any]] = []
    have_fast_bbox = (
        isinstance(bboxes_all, list)
        and isinstance(returned_idx, list)
        and len(returned_idx) == len(masks_b64)
    )

    for j, mb in enumerate(masks_b64):
        bb = None
        if have_fast_bbox:
            try:
                ii = int(returned_idx[j])
                bb0 = bboxes_all[ii]
                if isinstance(bb0, list) and len(bb0) == 4:
                    x0, y0, x1, y1 = map(int, bb0)
                    x0 = max(0, min(W - 1, x0))
                    y0 = max(0, min(H - 1, y0))
                    x1 = max(0, min(W - 1, x1))
                    y1 = max(0, min(H - 1, y1))
                    if x1 >= x0 and y1 >= y0:
                        bb = (x0, y0, x1, y1)
            except Exception:
                bb = None

        if bb is None:
            try:
                mask_l = b64_to_pil(mb, "L")
            except Exception:
                continue
            bb = mask_bbox(mask_l)
            if bb is None:
                continue

        bbox_xyxy = [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]

        area_px = None
        if isinstance(areas, list) and j < len(areas):
            area_px = areas[j]
        if area_px is None:
            try:
                mask_l = b64_to_pil(mb, "L")
                area_px = int((np.array(mask_l) > 0).sum())
            except Exception:
                area_px = int(
                    max(
                        1,
                        (bbox_xyxy[2] - bbox_xyxy[0] + 1)
                        * (bbox_xyxy[3] - bbox_xyxy[1] + 1),
                    )
                )

        area_ratio = float(area_px) / float(W * H)
        score = scores[j] if isinstance(scores, list) and j < len(scores) else None

        lab = None
        if isinstance(qwen_labels, list) and j < len(qwen_labels):
            lab = str(qwen_labels[j]).strip() or None
        if lab is None:
            lab = "object"

        bbox_norm = [
            bbox_xyxy[0] / W,
            bbox_xyxy[1] / H,
            bbox_xyxy[2] / W,
            bbox_xyxy[3] / H,
        ]

        objs.append(
            {
                "img_index": img_idx,
                "object_index": len(objs),  # per-image index
                "label": lab,
                "score": score,
                "bbox_xyxy": bbox_xyxy,
                "bbox_norm": bbox_norm,
                "area_px": int(area_px) if area_px is not None else None,
                "area_ratio": area_ratio,
                "mask_png_base64": mb,
            }
        )

    return objs, debug


def build_grounding_summary_text_multi(objs_by_img: List[List[Dict[str, Any]]], sizes: List[Tuple[int, int]]) -> str:
    parts = []
    total = 0
    for k, objs in enumerate(objs_by_img):
        w, h = sizes[k]
        parts.append(f"img#{k}: {len(objs)} objects; size={w}x{h}")
        total += len(objs)
    return "Grounding summary (optional/noisy): " + "; ".join(parts) + f"; total_objects={total}."


def build_obj_label_map_multi(objs_by_img: List[List[Dict[str, Any]]]) -> Dict[Tuple[int, int], str]:
    m: Dict[Tuple[int, int], str] = {}
    for k, objs in enumerate(objs_by_img):
        for o in objs:
            try:
                j = int(o.get("object_index", -1))
            except Exception:
                continue
            lab = str(o.get("label", "object")).strip() or "object"
            m[(k, j)] = lab
    return m


def compact_object_line_multi(o: Dict[str, Any], nd: int) -> str:
    ik = int(o.get("img_index", 0))
    ij = int(o.get("object_index", 0))
    lab = str(o.get("label", "object"))
    bb = o.get("bbox_norm", [0, 0, 0, 0])
    bb = round_list(bb, nd)
    ar = o.get("area_ratio", None)
    sc = o.get("score", None)
    ar_s = f"{float(ar):.4f}" if ar is not None else "na"
    sc_s = f"{float(sc):.3f}" if sc is not None else "na"
    return f"img#{ik}_obj#{ij} {lab} bb={bb} a={ar_s} s={sc_s}"


def select_topk_for_images_multi(objs_by_img: List[List[Dict[str, Any]]], k: int) -> List[Tuple[int, int]]:
    if k <= 0:
        return []
    items = []
    for ik, objs in enumerate(objs_by_img):
        for o in objs:
            ar = safe_float(o.get("area_ratio", 0.0))
            sc = safe_float(o.get("score", 0.0))
            ij = int(o.get("object_index", 0))
            items.append((ar, sc, ik, ij))
    items.sort(reverse=True)
    out = [(ik, ij) for _, _, ik, ij in items[:k]]
    return out


# ==========================================================
# Parse outputs
# ==========================================================
def _norm_path_item(x: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(x, dict):
        return None
    try:
        pid = int(x.get("path_id", 0))
    except Exception:
        pid = 0

    tag = str(x.get("angle_tag", "")).strip() or "uncertainty"
    rtxt = str(x.get("r", "")).strip()

    s = x.get("s", None)
    if not isinstance(s, dict):
        s = {}

    final_answer = str(s.get("final_answer", "")).strip() or "unknown"
    reasons = ensure_str_list(s.get("reasons", []))
    uncertainty = clamp_level(s.get("uncertainty", "high"), default="high")
    evidence_used = ensure_str_list(s.get("evidence_used", []))
    grounding_note = str(s.get("grounding_note", "")).strip()

    return {
        "path_id": pid,
        "angle_tag": tag,
        "r": rtxt,
        "s": {
            "final_answer": final_answer,
            "reasons": reasons,
            "uncertainty": uncertainty,
            "evidence_used": evidence_used,
            "grounding_note": grounding_note,
        },
    }


def normalize_multipath_out(raw: Dict[str, Any], tau: int) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"paths": [], "diversity_level": "low", "_raw_fallback": True}

    if "_raw" in raw and not any(k in raw for k in ["paths", "diversity_level"]):
        txt = str(raw.get("_raw", "")).strip()
        one = {
            "path_id": 0,
            "angle_tag": "uncertainty",
            "r": "",
            "s": {
                "final_answer": txt if txt else "unknown",
                "reasons": [],
                "uncertainty": "high",
                "evidence_used": [],
                "grounding_note": "raw_fallback",
            },
        }
        return {"paths": [one], "diversity_level": "low", "_raw_fallback": True}

    paths = raw.get("paths", [])
    if not isinstance(paths, list):
        paths = []

    out_paths = []
    for it in paths:
        p = _norm_path_item(it)
        if p is not None:
            out_paths.append(p)

    k = max(1, int(tau))
    if not out_paths:
        out_paths = [
            {
                "path_id": 0,
                "angle_tag": "uncertainty",
                "r": "",
                "s": {
                    "final_answer": "unknown",
                    "reasons": [],
                    "uncertainty": "high",
                    "evidence_used": [],
                    "grounding_note": "empty_paths",
                },
            }
        ]

    out_paths.sort(key=lambda z: int(z.get("path_id", 0)))
    fixed = []
    for i in range(min(k, len(out_paths))):
        z = out_paths[i]
        z["path_id"] = i
        fixed.append(z)
    while len(fixed) < k:
        z = dict(fixed[-1])
        z["path_id"] = len(fixed)
        fixed.append(z)

    div_level = clamp_level(raw.get("diversity_level", "low"), default="low")
    return {"paths": fixed[:k], "diversity_level": div_level}


def pick_representative_path(paths: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not paths:
        return {
            "path_id": 0,
            "angle_tag": "uncertainty",
            "r": "",
            "s": {
                "final_answer": "unknown",
                "reasons": [],
                "uncertainty": "high",
                "evidence_used": [],
                "grounding_note": "no_paths",
            },
        }

    ord_u = {"low": 0, "medium": 1, "high": 2}

    def key_fn(p):
        s = p.get("s", {}) if isinstance(p.get("s"), dict) else {}
        u = clamp_level(s.get("uncertainty", "high"), default="high")
        ans = normalize_answer(str(s.get("final_answer", "unknown")))
        bad = 1 if ans in ("unknown", "uncertain") else 0
        return (ord_u[u], bad)

    return min(paths, key=key_fn)


# ==========================================================
# evidence_used formatting: img#k_obj#j(label)
# ==========================================================
_EVID_RE = re.compile(r"img#\s*(\d+)\s*_obj#\s*(\d+)(?:\((.*?)\))?$", re.IGNORECASE)


def normalize_evidence_used_multi(items: List[str], obj_label_map: Dict[Tuple[int, int], str]) -> Tuple[List[str], Dict[str, Any]]:
    out = []
    stat = {"valid_obj_refs": 0, "invalid_obj_refs": 0, "unknown_label_filled": 0}
    for t in (items or []):
        s = str(t).strip()
        if not s:
            continue
        m = _EVID_RE.match(s)
        if not m:
            out.append(s)
            continue
        ik = int(m.group(1))
        ij = int(m.group(2))
        lab = obj_label_map.get((ik, ij), None)
        if lab is None:
            stat["invalid_obj_refs"] += 1
            out.append(f"img#{ik}_obj#{ij}(unknown)")
            stat["unknown_label_filled"] += 1
        else:
            stat["valid_obj_refs"] += 1
            out.append(f"img#{ik}_obj#{ij}({lab})")

    seen = set()
    dedup = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        dedup.append(s)
    return dedup[:6], stat


def evidence_stat_to_level(valid_refs: int, invalid_refs: int, empty: bool) -> str:
    if empty:
        return "low"
    tot = max(1, valid_refs + invalid_refs)
    return ratio_to_level(float(valid_refs) / float(tot))


# ==========================================================
# Global consensus over representative answers
# ==========================================================
def consensus_global(rep_calls: List[Dict[str, Any]]) -> Tuple[str, str, Dict[str, Any]]:
    if not rep_calls:
        return "unknown", "low", {"final_answer": "unknown", "uncertainty": "high"}

    groups = defaultdict(list)
    for o in rep_calls:
        ans = normalize_answer(o.get("final_answer", "unknown"))
        groups[ans].append(o)

    items = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    best_ans, best_list = items[0]
    total = sum(len(v) for _, v in items)
    frac = float(len(best_list)) / float(max(1, total))
    consensus_level = ratio_to_level(frac)

    ord_u = {"low": 0, "medium": 1, "high": 2}
    rep = min(best_list, key=lambda x: ord_u[clamp_level(x.get("uncertainty", "high"), default="high")])
    return best_ans, consensus_level, rep


# ==========================================================
# Fitness (levels -> internal mapping)
# ==========================================================
_LEVEL_SCORE = {"low": 0, "medium": 1, "high": 2}
_UNC_PEN = {"low": 0, "medium": 1, "high": 2}


def fitness_from_levels(cons_level: str, div_level: str, unc_level: str, evid_level: str) -> int:
    cons_level = clamp_level(cons_level, "low")
    div_level = clamp_level(div_level, "low")
    unc_level = clamp_level(unc_level, "high")
    evid_level = clamp_level(evid_level, "low")
    return (
        CONS_W * _LEVEL_SCORE[cons_level]
        + DIV_W * _LEVEL_SCORE[div_level]
        + EVID_W * _LEVEL_SCORE[evid_level]
        - UNC_W * _UNC_PEN[unc_level]
    )


def compute_principle_fitness_discrete(x_pop: List[str], call_records: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, Any]]:
    P = [str(p).strip() for p in (x_pop or []) if str(p).strip()]
    per_sum = {p: 0 for p in P}
    per_cnt = {p: 0 for p in P}

    hist = {
        "consensus_level": Counter(),
        "diversity_level": Counter(),
        "uncertainty": Counter(),
        "evidence_level": Counter(),
    }

    for o in (call_records or []):
        p = str(o.get("_principle", "")).strip()
        if p not in per_sum:
            continue

        cons_level = clamp_level(o.get("consensus_level", "low"), "low")
        div_level = clamp_level(o.get("diversity_level", "low"), "low")
        unc_level = clamp_level(o.get("uncertainty", "high"), "high")
        evid_level = clamp_level(o.get("evidence_level", "low"), "low")

        hist["consensus_level"][cons_level] += 1
        hist["diversity_level"][div_level] += 1
        hist["uncertainty"][unc_level] += 1
        hist["evidence_level"][evid_level] += 1

        score = fitness_from_levels(cons_level, div_level, unc_level, evid_level)
        per_sum[p] += int(score)
        per_cnt[p] += 1

    fitness = {}
    for p in P:
        fitness[p] = 0 if per_cnt[p] == 0 else int(round(per_sum[p] / max(1, per_cnt[p])))

    stats = {
        "level_hist": {k: dict(v) for k, v in hist.items()},
        "per_principle_usage_cnt": per_cnt,
        "fitness_weights_internal": {"CONS_W": CONS_W, "DIV_W": DIV_W, "UNC_W": UNC_W, "EVID_W": EVID_W},
    }
    return fitness, stats


def select_elite_principles(fitness: Dict[str, int], k: int) -> Tuple[List[str], List[str]]:
    items = sorted(fitness.items(), key=lambda kv: kv[1], reverse=True)
    selected = [p for p, _ in items[:k]]
    dropped = [p for p, _ in items[k:]]
    return selected, dropped


def enforce_population_size(principles: List[str], n: int) -> List[str]:
    ps = [str(p).strip() for p in (principles or []) if str(p).strip()]
    seen = set()
    out = []
    for p in ps:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= n:
            break

    if n <= 0:
        return out

    pad_bank = [
        "Prefer visible evidence; avoid speculation",
        "State uncertainty explicitly when cues are weak",
        "Cross-check multiple cues before concluding",
        "Avoid over-trusting a single detail; seek corroboration",
        "Consider plausible alternatives briefly then decide",
        "If unsure, answer conservatively and justify uncertainty",
    ]
    for p in pad_bank:
        if len(out) >= n:
            break
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:n]


# ==========================================================
# API
# ==========================================================
@app.post("/vqa")
async def vqa(
    # backward compat
    image: Optional[UploadFile] = File(None),
    # multi-image
    images: Optional[List[UploadFile]] = File(None),
    question: str = Form(...),
    init_principles_n: int = Form(MU_PLUS_LAMBDA),
    return_traces: bool = Form(True),
):
    # collect images
    imgs: List[bytes] = []
    if images:
        for im in images:
            imgs.append(await im.read())
    elif image is not None:
        imgs.append(await image.read())

    if len(imgs) == 0:
        return JSONResponse({"error": "No image(s) provided."}, status_code=400)

    # 1) grounding per image (single or batch)
    if GROUNDING_IMAGE_MODE == "batch" and len(imgs) > 1:
        groundings = await call_grounding_amg_batch(imgs)
    else:
        groundings = await asyncio.gather(*[call_grounding_amg_single(b) for b in imgs])

    orig_pils: List[Image.Image] = []
    orig_data_urls: List[str] = []
    sizes: List[Tuple[int, int]] = []
    objs_by_img: List[List[Dict[str, Any]]] = []
    grounding_parse_debug: List[Dict[str, Any]] = []

    for k, g in enumerate(groundings):
        orig = b64_to_pil(g["orig_png_base64"], "RGB")
        orig_pils.append(orig)
        sizes.append(orig.size)
        orig_data_urls.append(data_url_from_png_bytes(pil_png_bytes(orig)))

        objs, dbg = extract_grounding_objects_single(g, orig, img_idx=k)
        objs_by_img.append(objs)
        grounding_parse_debug.append(dbg)

    grounding_summary = build_grounding_summary_text_multi(objs_by_img, sizes)
    obj_label_map = build_obj_label_map_multi(objs_by_img)

    total_objs = sum(len(x) for x in objs_by_img)
    if total_objs == 0:
        return JSONResponse({
            "question": question,
            "final_answer": "uncertain",
            "final_reasons": ["No objects were grounded."],
            "final_uncertainty": "high",
            "grounding_summary_optional": grounding_summary,
            "grounding_parse_debug": grounding_parse_debug,
            "meta": {"objects_total": 0, "proxy": "SAP"},
        })

    # 2) fitness-side diagnostics only (NOT sent to LLM)
    obj_lines_all: List[str] = []
    for objs in objs_by_img:
        for o in objs:
            obj_lines_all.append(compact_object_line_multi(o, nd=BBOX_DECIMALS))
    obj_lines_cap = obj_lines_all[: max(1, min(MAX_OBJECT_TEXT, len(obj_lines_all)))]
    topk_obj_pairs = select_topk_for_images_multi(objs_by_img, k=min(MAX_IMAGES_TOPK, total_objs))

    # 3) init principles (TEMPLATED)
    init_payload = build_init_principles_prompt(
        model=QWEN_MODEL,
        p=question,
        n=int(init_principles_n),
        grounding_summary=grounding_summary,
    )
    init_out = await call_qwen(init_payload)
    x_pop = ensure_str_list(init_out.get("principles", []), fallback=["Use visible evidence only", "State uncertainty when unsure"])
    x_pop = enforce_population_size(x_pop, MU_PLUS_LAMBDA)

    traces = []
    last_gen_call_records: List[Dict[str, Any]] = []
    last_gen_consensus_rep: Dict[str, Any] = {}
    last_gen_consensus_norm: str = "unknown"
    last_gen_consensus_level: str = "low"
    last_fit_stats: Dict[str, Any] = {}
    last_fit: Dict[str, int] = {}

    tau = max(1, int(TAU))

    # NOTE: chunking removed in multi-image version; LLM input never uses obj lines.
    for gen in range(max(1, EVO_GENS)):
        call_records: List[Dict[str, Any]] = []
        rep_calls_for_consensus: List[Dict[str, Any]] = []

        tasks = []
        meta = []
        for pidx, x_i in enumerate(x_pop):
            payload = build_single_call_multipath_prompt_multi(
                model=QWEN_MODEL,
                p=question,
                grounding_summary=grounding_summary,
                x_i=x_i,
                orig_image_data_urls=orig_data_urls,
                tau=tau,
            )
            tasks.append(call_qwen(payload))
            meta.append((x_i, pidx))

        outs = await asyncio.gather(*tasks)

        parsed_per_principle = []
        for i, raw in enumerate(outs):
            x_i, pidx = meta[i]
            norm = normalize_multipath_out(raw, tau=tau)
            paths = norm["paths"]
            div_level = clamp_level(norm["diversity_level"], "low")

            # normalize evidence_used to img#k_obj#j(label)
            for pp in paths:
                ss = pp.get("s", {}) if isinstance(pp.get("s"), dict) else {}
                ev_raw = ensure_str_list(ss.get("evidence_used", []))
                ev_norm, _ = normalize_evidence_used_multi(ev_raw, obj_label_map)
                ss["evidence_used"] = ev_norm
                ss["uncertainty"] = clamp_level(ss.get("uncertainty", "high"), "high")
                pp["s"] = ss

            rep_path = pick_representative_path(paths)
            rep_s = rep_path.get("s", {}) if isinstance(rep_path.get("s"), dict) else {}

            rep_unc = clamp_level(rep_s.get("uncertainty", "high"), "high")
            rep_item = {
                "_principle": x_i,
                "_principle_idx": pidx,
                "final_answer": str(rep_s.get("final_answer", "unknown")).strip() or "unknown",
                "uncertainty": rep_unc,
                "reasons": ensure_str_list(rep_s.get("reasons", [])),
                "evidence_used": ensure_str_list(rep_s.get("evidence_used", [])),
            }
            parsed_per_principle.append((x_i, pidx, paths, div_level, rep_path, rep_s))
            rep_calls_for_consensus.append(rep_item)

        # global consensus (over representative answers)
        consensus_norm, consensus_level, consensus_rep = consensus_global(rep_calls_for_consensus)

        # finalize call_records with evidence_level + per-call consensus_level (match/non-match)
        for (x_i, pidx, paths, div_level, rep_path, rep_s) in parsed_per_principle:
            rep_ans = str(rep_s.get("final_answer", "unknown")).strip() or "unknown"
            rep_unc = clamp_level(rep_s.get("uncertainty", "high"), "high")
            ev = ensure_str_list(rep_s.get("evidence_used", []))

            # evidence validity stats for discretized evidence_level
            valid_refs = 0
            invalid_refs = 0
            for t in ev:
                m = _EVID_RE.match(str(t).strip())
                if not m:
                    continue
                ik = int(m.group(1))
                ij = int(m.group(2))
                if (ik, ij) in obj_label_map:
                    valid_refs += 1
                else:
                    invalid_refs += 1

            evidence_level = evidence_stat_to_level(valid_refs=valid_refs, invalid_refs=invalid_refs, empty=(len(ev) == 0))

            # per-call consensus: binary -> low/high
            rep_match = (normalize_answer(rep_ans) == consensus_norm)
            per_call_cons_level = "high" if rep_match else "low"

            rec = {
                "_gen": gen,
                "_principle": x_i,
                "_principle_idx": pidx,

                "tau": tau,
                "paths": paths,

                # DISCRETE LEVELS
                "diversity_level": clamp_level(div_level, "low"),
                "uncertainty": rep_unc,
                "consensus_level": per_call_cons_level,
                "evidence_level": evidence_level,

                # representative
                "final_answer": rep_ans,
                "reasons": ensure_str_list(rep_s.get("reasons", [])),
                "evidence_used": ev,
                "grounding_note": str(rep_s.get("grounding_note", "")).strip(),

                "_evid_valid_refs": int(valid_refs),
                "_evid_invalid_refs": int(invalid_refs),

                "rep_path": {
                    "path_id": int(rep_path.get("path_id", 0)),
                    "angle_tag": str(rep_path.get("angle_tag", "")).strip(),
                    "r": str(rep_path.get("r", "")).strip(),
                },
            }
            call_records.append(rec)

        # keep last-gen
        last_gen_call_records = call_records
        last_gen_consensus_norm = consensus_norm
        last_gen_consensus_level = consensus_level
        last_gen_consensus_rep = consensus_rep

        # fitness
        fit, fit_stats = compute_principle_fitness_discrete(x_pop=x_pop, call_records=call_records)
        last_fit = fit
        last_fit_stats = fit_stats

        elite_k = max(1, min(int(MU), len(x_pop)))
        elites, dropped = select_elite_principles(fit, k=elite_k)

        if return_traces:
            fit_list = [{"principle": p, "fitness": int(fit.get(p, 0))} for p in x_pop]
            fit_list.sort(key=lambda x: x["fitness"], reverse=True)

            call_summ = []
            for o in call_records[: min(48, len(call_records))]:
                pv_paths = []
                for pp in (o.get("paths") or [])[:2]:
                    ss = pp.get("s", {}) if isinstance(pp.get("s"), dict) else {}
                    pv_paths.append({
                        "path_id": pp.get("path_id"),
                        "angle_tag": pp.get("angle_tag"),
                        "final_answer": ss.get("final_answer"),
                        "uncertainty": clamp_level(ss.get("uncertainty", "high"), "high"),
                        "evidence_used": (ss.get("evidence_used") or [])[:4],
                    })

                call_summ.append({
                    "principle": o.get("_principle"),
                    "rep_final_answer": o.get("final_answer"),
                    "levels": {
                        "consensus_level": o.get("consensus_level"),
                        "diversity_level": o.get("diversity_level"),
                        "uncertainty": o.get("uncertainty"),
                        "evidence_level": o.get("evidence_level"),
                    },
                    "paths_preview": pv_paths,
                })

            traces.append({
                "gen": gen,
                "x_pop": x_pop,
                "calls_expected_this_gen": len(x_pop),
                "tau": tau,

                "global_consensus_norm": consensus_norm,
                "global_consensus_level": consensus_level,
                "global_consensus_rep": consensus_rep,

                "fitness_top": fit_list[: min(30, len(fit_list))],
                "elites": elites,
                "dropped": dropped,
                "fitness_stats": fit_stats,
                "trace_calls_preview": call_summ,
            })

        # evolve (TEMPLATED)
        if gen < EVO_GENS - 1:
            known_list = [{"principle": p, "fitness": int(fit.get(p, 0))} for p in x_pop]
            known_list.sort(key=lambda x: x["fitness"], reverse=True)

            n_new = max(0, MU_PLUS_LAMBDA - len(elites))
            payload_next = build_nextgen_principles_prompt(
                model=QWEN_MODEL,
                p=question,
                grounding_summary=grounding_summary,
                known_principles_with_scores=known_list,
                selected=elites,
                dropped=dropped,
                n_new=n_new,
            )
            nxt = await call_qwen(payload_next)
            new_ps = ensure_str_list(nxt.get("new_principles", []), fallback=[])

            next_pop = []
            seen = set()
            for p in elites + new_ps:
                p = str(p).strip()
                if not p or p in seen:
                    continue
                seen.add(p)
                next_pop.append(p)

            x_pop = enforce_population_size(next_pop, MU_PLUS_LAMBDA)

    # 6) final output (TEMPLATED)
    if USE_FINAL_AGGREGATOR > 0:
        summaries = []
        for o in last_gen_call_records:
            summaries.append({
                "principle": o.get("_principle"),
                "rep_final_answer": o.get("final_answer"),
                "rep_reasons": (o.get("reasons") or [])[:4],
                "rep_uncertainty": o.get("uncertainty"),
                "diversity_level": o.get("diversity_level"),
                "consensus_level": o.get("consensus_level"),
                "evidence_level": o.get("evidence_level"),
                "tau": o.get("tau"),
            })
        summaries = summaries[: min(96, len(summaries))]

        try:
            agg_payload = build_final_aggregator_prompt(
                model=QWEN_MODEL,
                p=question,
                grounding_summary=grounding_summary,
                call_summaries=summaries,
            )
            agg = await call_qwen(agg_payload)
            final_answer = str(agg.get("final_answer", "")).strip()
            final_reasons = ensure_str_list(agg.get("reasons", []))
            final_uncertainty = clamp_level(agg.get("uncertainty", "high"), "high")
            if not final_answer:
                raise RuntimeError("Aggregator empty final_answer")
            final_meta = {"final_mode": "final_aggregator"}
        except Exception as e:
            final_answer = str(last_gen_consensus_rep.get("final_answer", "")).strip() or "unknown"
            final_reasons = ensure_str_list(last_gen_consensus_rep.get("reasons", []))
            final_reasons = (final_reasons[:4] if final_reasons else []) + [f"Fallback: {type(e).__name__}"]
            final_uncertainty = "high"
            final_meta = {"final_mode": "consensus_rep_fallback"}
    else:
        final_answer = str(last_gen_consensus_rep.get("final_answer", "")).strip() or "unknown"
        final_reasons = ensure_str_list(last_gen_consensus_rep.get("reasons", []))
        final_uncertainty = clamp_level(last_gen_consensus_rep.get("uncertainty", "high"), "high")
        final_meta = {"final_mode": "consensus_rep"}

    resp = {
        "question": question,
        "final_answer": final_answer,
        "final_reasons": final_reasons,
        "final_uncertainty": final_uncertainty,

        "final_consensus_level": last_gen_consensus_level,
        "final_consensus_answer_norm": last_gen_consensus_norm,

        "grounding_summary_optional": grounding_summary,
        "grounding_parse_debug": grounding_parse_debug,

        "evidence_preview": {
            "object_lines_used": obj_lines_cap,  # diagnostics only
            "object_lines_total": len(obj_lines_all),
            "topk_object_pairs_for_diag": topk_obj_pairs,  # (img_idx,obj_idx)
            "caps": {
                "MAX_OBJECT_TEXT": MAX_OBJECT_TEXT,
                "MAX_IMAGES_TOPK": MAX_IMAGES_TOPK,
                "BBOX_DECIMALS": BBOX_DECIMALS,
            }
        },

        "meta": {
            "proxy": "SAP",
            "GroundingMaskWithLabel_URL": GroundingMaskWithLabel_URL,
            "QWEN_URL": QWEN_URL,
            "QWEN_CHAT_PATH": QWEN_CHAT_PATH,
            "QWEN_MODEL": QWEN_MODEL,
            "GROUNDING_IMAGE_MODE": GROUNDING_IMAGE_MODE,

            "POINTS_PER_BATCH": POINTS_PER_BATCH,
            "MAX_MASKS": MAX_MASKS,

            "MU_PLUS_LAMBDA": MU_PLUS_LAMBDA,
            "MU": MU,
            "LAMBDA": LAMBDA,
            "EVO_GENS": EVO_GENS,
            "TAU": tau,

            "num_images": len(imgs),
            "objects_total": total_objs,

            "fitness_weights_internal": {"CONS_W": CONS_W, "DIV_W": DIV_W, "UNC_W": UNC_W, "EVID_W": EVID_W},

            "calls_per_gen_expected": len(x_pop),
            "calls_total_expected_approx": (
                1
                + (EVO_GENS * len(x_pop))
                + max(0, EVO_GENS - 1)
                + (1 if USE_FINAL_AGGREGATOR > 0 else 0)
            ),

            "last_fit_stats": last_fit_stats,
            "last_fit_top": sorted(
                [{"principle": p, "fitness": int(last_fit.get(p, 0))} for p in x_pop],
                key=lambda x: x["fitness"],
                reverse=True
            )[: min(10, len(x_pop))],

            "USE_FINAL_AGGREGATOR": USE_FINAL_AGGREGATOR,
            **final_meta,
        },
    }
    if return_traces:
        resp["traces"] = traces

    return JSONResponse(resp)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "proxy": "SAP",
        "discrete_levels": ["low", "medium", "high"],
        "grounding_service": GroundingMaskWithLabel_URL,
        "qwen_model": QWEN_MODEL,
        "grounding_image_mode": GROUNDING_IMAGE_MODE,
        "population": {"MU_PLUS_LAMBDA": MU_PLUS_LAMBDA, "MU": MU, "LAMBDA": LAMBDA},
        "tau": TAU,
        "caps": {
            "MAX_OBJECT_TEXT": MAX_OBJECT_TEXT,
            "MAX_IMAGES_TOPK": MAX_IMAGES_TOPK,
            "BBOX_DECIMALS": BBOX_DECIMALS,
            "USE_FINAL_AGGREGATOR": USE_FINAL_AGGREGATOR,
        },
        "llm_input_policy": {
            "images_sent_to_llm": "original_images_only",
            "text_sent_to_llm": "grounding_summary + question + active_principle",
            "obj_lines_sent_to_llm": False,
            "obj_crops_sent_to_llm": False,
        },
    }
