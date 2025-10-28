"""
enrich_queries_agent.py (LLM-driven)

LangGraph node that *enriches* each planned SlotSpec in state.slots with a
context-aware, embedding-friendly query string (and optional soft_terms).
No hand-written rules; all planning is LLM-prompted with a strict JSON schema
and a repair loop for robustness.

Usage (example):
    export OPENAI_API_KEY=...
    python enrich_queries_agent.py  # demo with load_pet_state if available

Environment vars:
    ENRICH_QUERIES_MODEL  (optional; defaults to OPENAI_MODEL or gpt-4o-mini)
    OPENAI_MODEL          (fallback if above not set)

Depends on your domain models in nppState.py and optional loader in load_pet_state.py.
"""
from __future__ import annotations

import os
import json
from typing import Dict, List, Optional


# OpenAI (or compatible) client
from openai import OpenAI


# Domain models
from LG_Agents.states.sessionState import SessionStateModel
from LG_Agents.states.bundleState import SlotSpec
import LG_Agents.helperFunctions.agent_helper_function as agent_helper_function




# ----------------------------
# LLM plumbing
# ----------------------------

SYSTEM_PROMPT = """You are the *Enrich Queries Agent* for a new-pet-parent bundle builder.
Your goal: for each already-planned slot, write a short, embedding-friendly query that retrieves the most appropriate Chewy products for THIS pet.

QUALITY BAR (must meet ALL):
- Produce ONE query per input slot_id; do not invent or drop slots.
- Queries must be concise (≈ 6–18 tokens), natural, and specific.
- **MANDATORY CONTEXT INCLUSION:** Each query_text must include AT LEAST TWO of:
  1) species + life stage or explicit age (e.g., "puppy", "kitten", "adult", "senior", or "9 month puppy")
  2) size/weight signal (e.g., "large breed", "20–30 lb", "small cat")
  3) habit/condition (e.g., "teething", "aggressive chewer", "anxious", "heavy shedder", "indoor cat")
- Use the slot's product_type and reason_for_suggestion to steer specificity (function > brand).
- Regulated categories must be safety-forward (e.g., "flea & tick prevention for puppies", include weight if dosing matters).
- Prefer lowercase; minimal punctuation; no quotes; avoid filler stop-words.
- No brand names unless explicitly necessary.
- Optionally include `soft_terms` (3–8 related descriptors/synonyms) to help rerankers.

INPUT YOU’LL GET:
- pet_profile: species, breed, gender, age_months, weight_lb, life_stage, size_class, location_zip,
  habits[], recent_conditions[], geo_eventcondition[], recent_purchases[].
- slots: [{slot_id, name, role, product_type, reason_for_suggestion, must_have, allow_substitution}, …]

OUTPUT (STRICT JSON, no commentary):
{
  "queries": [
    {
      "slot_id": str,                 # exactly matches an input slot_id
      "query_text": str,              # the main embedding query phrase (<= 18 tokens)
      "soft_terms": [str, ...]        # optional related terms; [] if none
    },
    ...
  ]
}

STYLE PATTERN (guidance, not a template):
- <functional product> for <life-stage/age> <species> <size/weight> <habit/condition> <optional geo/season>
  e.g., "durable chew toy for puppy large breed aggressive chewer"
  e.g., "flea and tick prevention for 12 lb kitten indoor"
  e.g., "sensitive stomach puppy food large breed 9 month"

GOOD EXAMPLES:
- slot_id: chew_toys → "durable chew toy for puppy large breed teething aggressive chewer"
  soft_terms: ["indestructible", "rubber", "power chewer", "teether"]

- slot_id: puppy_food → "sensitive stomach puppy food for large breed 9 month"
  soft_terms: ["omega 3", "dha", "digestive care", "kibble"]

- slot_id: leash → "strong leash for adult dog 50–70 lb high energy"
  soft_terms: ["no pull", "padded handle", "traffic handle"]

- slot_id: flea_tick_prevention → "flea and tick prevention for puppy 20–25 lb"
  soft_terms: ["puppy safe", "weight based", "topical", "oral"]

- slot_id: grooming → "deshedding brush for adult cat heavy shedder indoor"
  soft_terms: ["long hair", "undercoat", "de-shed"]
"""



def _slots_brief(slots: List[SlotSpec]) -> List[dict]:
    """Trim slots to only fields the LLM needs."""
    brief = []
    for s in slots:
        brief.append({
            "slot_id": s.slot_id,
            "name": s.name,
            "role": s.role,
            "product_type": s.product_type,
            "reason_for_suggestion": s.reason_for_suggestion,
            "must_have": s.must_have,
            "allow_substitution": s.allow_substitution,
        })
    return brief


def _call_llm(pet_ctx: dict, slots_ctx: List[dict], *, temperature: float = 0.2, max_tokens: int = 1400) -> str:
    client = OpenAI()
    model = os.getenv("ENRICH_QUERIES_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Generate embedding queries for the provided slots.\n\n"
                + json.dumps({"pet_profile": pet_ctx, "slots": slots_ctx}, ensure_ascii=False)
            ),
        },
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or "{}"


def _coerce_queries_from_json(raw_json: str) -> Dict[str, dict]:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\nFirst 300: {raw_json[:300]}")

    if not isinstance(data, dict) or "queries" not in data or not isinstance(data["queries"], list):
        raise ValueError("JSON must contain top-level 'queries' as an array")

    out: Dict[str, dict] = {}
    for q in data["queries"]:
        if not isinstance(q, dict):
            continue
        sid = q.get("slot_id")
        qt = q.get("query_text")
        if not sid or not qt:
            continue
        out[sid] = {
            "query_text": str(qt).strip(),
            "soft_terms": [str(t).strip() for t in (q.get("soft_terms") or []) if str(t).strip()],
        }
    return out


def _repair_json(pet_ctx: dict, slots_ctx: List[dict], error_msg: str, previous_raw: str) -> str:
    client = OpenAI()
    model = os.getenv("ENRICH_QUERIES_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Your previous JSON did not validate. Return only corrected JSON.\n\n"
            f"Validation error: {error_msg}\n\n"
            f"Previous JSON:\n{previous_raw}\n\n"
            f"Context again:\n{json.dumps({'pet_profile': pet_ctx, 'slots': slots_ctx}, ensure_ascii=False)}"
        )},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or "{}"


# ----------------------------
# Enricher node
# ----------------------------

def enrich_queries_agent(state: SessionStateModel) -> SessionStateModel:
    """
    For each state.slots[i], populate query_text (and optionally soft_terms)
    using an LLM that considers both the slot metadata and the pet context.
    """
    if not state.bundle_state.slots:
        # Nothing to do—return unchanged state.
        return state

    pet_ctx = agent_helper_function.pet_context_helper(state.pet_profile)
    slots_ctx = _slots_brief(state.bundle_state.slots)
    raw = _call_llm(pet_ctx, slots_ctx)

    tries = 0
    while True:
        tries += 1
        try:
            qmap = _coerce_queries_from_json(raw)  # slot_id -> {"query_text", "soft_terms"}
            break
        except Exception as e:
            if tries >= 2:
                raise
            raw = _repair_json(pet_ctx, slots_ctx, str(e), raw)

    # Write results back into the state (preserve order and other fields)
    updated_slots: List[SlotSpec] = []
    for s in state.bundle_state.slots:
        if s.slot_id in qmap:
            s.query_text = qmap[s.slot_id]["query_text"]
            # Optional: store soft_terms if your downstream retriever/reranker uses them
            if hasattr(s, "soft_terms"):
                s.soft_terms = qmap[s.slot_id].get("soft_terms") or []
        updated_slots.append(s)

    state.bundle_state.slots = updated_slots
    return state

