# template.py
# SAP v3 prompt templates (user_text)
# Design goal:
#   - Images are the primary evidence.
#   - Grounding summary is OPTIONAL / NOISY / may be WRONG.
#   - Summary is used to suggest what to re-check, not what to believe.
#   - When conflict exists, trust the image; answer conservatively if still ambiguous.

import json
from typing import Any, Dict, List


# ------------------------------
# Shared instruction blocks
# ------------------------------
def _evidence_hierarchy_block() -> str:
    return (
        "Evidence hierarchy (MUST follow):\n"
        "1) The input image(s) are the ONLY reliable source.\n"
        "2) The grounding summary is OPTIONAL and may be incomplete, noisy, or wrong.\n"
        "Use the summary only as a hint for what to re-check in the image(s), not as a fact.\n"
        "If the summary conflicts with what you can see, trust the image(s).\n"
        "If the image(s) do not clearly support an answer, be conservative and state uncertainty.\n"
    )


def _grounding_reference_rule() -> str:
    return (
        "Grounding references:\n"
        "- If you mention grounded objects, cite them as 'img#<k>_obj#<j>' (no extra text inside the id).\n"
        "- Only cite an object if it truly supports your claim based on the image(s).\n"
        "- If you cannot verify a detail from the image(s), do NOT assert it as fact.\n"
    )


def _strict_json_rule(schema: str) -> str:
    return (
        "Output format:\n"
        f"- Return STRICT JSON only.\n"
        f"- Keys must match exactly this schema: {schema}\n"
        "- No extra keys. No markdown. No prose outside JSON.\n"
    )


# ------------------------------
# Template 1: init principles
# ------------------------------
def build_init_principles_prompt(
    *,
    p: str,
    n: int,
    grounding_summary: str,
    model: str,
) -> Dict[str, Any]:
    user_text = (
        "You will propose a diverse set of short reasoning principles (x) for grounded visual question answering.\n\n"
        + _evidence_hierarchy_block()
        + "\n"
        f"Grounding summary (optional/noisy):\n{grounding_summary}\n\n"
        f"Question (p):\n{p}\n\n"
        "Write principles that force attention to the image(s) and avoid hallucination.\n"
        "Principles should be short, actionable, and different from each other.\n"
        "Do NOT treat the summary as ground truth.\n\n"
        + _strict_json_rule("{principles: [str]}")
        + f"- Provide about {int(n)} items in 'principles'.\n"
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You generate reasoning principles x for grounded VQA."},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.7,
        "max_tokens": 384,
    }


# ------------------------------
# Template 2: evolve principles
# ------------------------------
def build_nextgen_principles_prompt(
    *,
    p: str,
    grounding_summary: str,
    known_principles_with_scores: List[Dict[str, Any]],
    selected: List[str],
    dropped: List[str],
    n_new: int,
    model: str,
) -> Dict[str, Any]:
    user_text = (
        "You will propose NEW reasoning principles (x) to improve grounded visual question answering.\n\n"
        + _evidence_hierarchy_block()
        + "\n"
        f"Grounding summary (optional/noisy):\n{grounding_summary}\n\n"
        f"Question (p):\n{p}\n\n"
        "Known principles with fitness (higher is better; may be noisy):\n"
        f"{json.dumps(known_principles_with_scores, ensure_ascii=False)}\n\n"
        "Selected elites (keep style, but you may refine):\n"
        f"{json.dumps(selected, ensure_ascii=False)}\n\n"
        "Dropped (avoid repeating similar ideas):\n"
        f"{json.dumps(dropped, ensure_ascii=False)}\n\n"
        "Requirements for NEW principles:\n"
        "- Exactly NEW (not copies/paraphrases of elites).\n"
        "- Emphasize checking the image(s) over relying on text summary.\n"
        "- Encourage conservative answers when visual evidence is weak.\n"
        "- Each principle should encourage a different angle (e.g., spatial check, text reading, counting, consistency check).\n\n"
        + _strict_json_rule("{new_principles: [str]}")
        + f"- Provide exactly {int(n_new)} items.\n"
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You propose improved principles x for grounded VQA."},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.8,
        "max_tokens": 768,
    }


# ------------------------------
# Template 3: multi-route reasoning (single call, tau paths)
# ------------------------------
def build_single_call_multipath_prompt_multi(
    *,
    p: str,
    grounding_summary: str,
    x_i: str,
    orig_image_data_urls: List[str],
    tau: int,
    model: str,
) -> Dict[str, Any]:
    tau = max(1, int(tau))
    user_text = (
        "You will answer a visual question using the image(s).\n\n"
        + _evidence_hierarchy_block()
        + "\n"
        f"Grounding summary (optional/noisy):\n{grounding_summary}\n\n"
        f"Question (p):\n{p}\n\n"
        "Active principle x (ONLY this one):\n"
        f"- {x_i}\n\n"
        + _grounding_reference_rule()
        + "\n"
        f"Task: Produce exactly {tau} distinct reasoning paths under the SAME principle x.\n"
        "Each path must:\n"
        "- Provide final_answer (string).\n"
        "- Provide 2–5 short reasons based on what is visible in the image(s).\n"
        "- Provide uncertainty in {'low','medium','high'}.\n"
        "- Provide evidence_used as 0–6 items, each either:\n"
        "  * a grounded object id 'img#<k>_obj#<j>' (preferred), or\n"
        "  * a short phrase if no object id applies.\n"
        "- Provide grounding_note: a short note about what you relied on in the image(s).\n\n"
        "Important:\n"
        "- Do NOT treat the summary as factual evidence.\n"
        "- If a path relies on something not clearly visible, increase uncertainty and say so.\n\n"
        "Then assess diversity among these paths and output diversity_level in {'low','medium','high'}.\n\n"
        + _strict_json_rule(
            "{paths:[{path_id:int, angle_tag:str, r:str, "
            "s:{final_answer:str, reasons:[str], uncertainty:'low|medium|high', "
            "evidence_used:[str], grounding_note:str}}], diversity_level:'low|medium|high'}"
        )
        + f"- paths length MUST equal {tau}.\n"
        "- path_id must be 0..TAU-1.\n"
    )

    content_blocks = [{"type": "text", "text": user_text}]
    for u in (orig_image_data_urls or []):
        content_blocks.append({"type": "image_url", "image_url": {"url": u}})

    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You answer visual questions using the given image(s). "
                    "Treat images as primary evidence; summary is optional/noisy. "
                    "Follow the active principle. Output strict JSON."
                ),
            },
            {"role": "user", "content": content_blocks},
        ],
        "temperature": 0.25,
        "max_tokens": 2000,
    }


# ------------------------------
# Template 4: final aggregator
# ------------------------------
def build_final_aggregator_prompt(
    *,
    p: str,
    grounding_summary: str,
    call_summaries: List[Dict[str, Any]],
    model: str,
) -> Dict[str, Any]:
    user_text = (
        "You will synthesize a final answer from multiple candidate answers.\n\n"
        + _evidence_hierarchy_block()
        + "\n"
        f"Grounding summary (optional/noisy):\n{grounding_summary}\n\n"
        f"Question (p):\n{p}\n\n"
        "Candidate summaries (may contain errors or be inconsistent):\n"
        f"{json.dumps(call_summaries, ensure_ascii=False)}\n\n"
        "Synthesis rules:\n"
        "- Prefer answers that are better supported by visual evidence.\n"
        "- If candidates conflict and visual evidence is unclear, choose the most conservative answer and raise uncertainty.\n"
        "- Keep reasons short and tied to observable cues.\n\n"
        + _strict_json_rule("{final_answer:str, reasons:[str], uncertainty:'low|medium|high'}")
        + "- Provide 2–5 reasons.\n"
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful answer synthesizer grounded in images."},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }
