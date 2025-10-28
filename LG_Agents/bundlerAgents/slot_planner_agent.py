"""
slot_planner_agent.py (LLM-driven)

LangGraph node that uses an LLM to *plan* bundle slots for a single pet,
producing structured `SlotSpec` objects (from nppState.py). No hand-written
if/else trees; planning logic lives in the LLM prompt with a strict JSON schema
and a repair loop for robustness.

Run example:
    python slot_planner_agent.py --pet-id 1

Dependencies:
    pip install langgraph openai pydantic

This module expects domain models defined in nppState.py and an optional
helper loader in load_pet_state.py.
"""
from __future__ import annotations

import os
import json
import argparse
import logging
from typing import List, Optional

# LangGraph
from langgraph.graph import StateGraph, END

# OpenAI (or compatible) client
from openai import OpenAI

# Domain models
from LG_Agents.states.sessionState import SessionStateModel
from LG_Agents.states.bundleState import SlotSpec
import LG_Agents.helperFunctions.agent_helper_function as agent_helper_function





# ----------------------------
# LLM plumbing
# ----------------------------

SYSTEM_PROMPT = (
    """You are the Slot Planner Agent for a new-pet-parent bundle builder.
    Given a PetProfile and context, infer *which* shopping slots are appropriate given the context.
    You *do not* pick products; you only plan slots.
    Return **strict JSON** matching the schema below. Do not include commentary.
    Principles:
    - Be context-aware: breed, species, gender, age (life stage), size/weight, habits, recent_conditions, and geo_eventcondition.
    - Think about outdoor vs indoor tendencies, behavior signals (hyperactive/lethargic/teething/chewer), hygiene (odor, ticks/fleas), and near-term geo events (heatwave, winter storm, holidays).
    - Keep it safety-first: if a slot implies medication or regulated items, frame it as a category (e.g., 'flea & tick prevention') and let downstream checks enforce policies.
    - Cap the plan at 12 slots. Prioritize 'core' essentials first, then 'optional' and 'premium_addon'.
    - Use short, shopper-friendly names and reasons (1–2 sentences).
    - Use snake_case for slot_id and avoid duplicates.
    Examples of how to map signals (not exhaustive, use judgment):
    • hyperactive → calming aid (spray/diffuser/chews).
    • lethargic → general vitamins/energy support.
    • outdoor → leash/harness; grooming shampoo; paw cleaner for muddy play.
    • ticks/fleas → monthly prevention + tick removal tool.
    • odor → deodorizing spray/shampoo.
    • heatwave → cooling mat, summer booties; winter storm → warm blanket, coat.
    • teething → teething chew toy; heavy chewer → durable chew, strong harness.
    • festive/holiday soon → festive apparel/bandana.
    JSON schema (informal):
    {
      'slots': [
        {
          'slot_id': str,
          'role': 'core'|'optional'|'premium_addon'|'single_purchase',
          'name': str,
          'must_have': bool,
          'allow_substitution': bool,
          'product_type': str,
          'reason_for_suggestion': str,
          'query_text': str | null,
        }
      ]
    }"""
)




def _call_llm(context: dict, *, temperature: float = 0.2, max_tokens: int = 1200) -> str:
    """Call the chat model and return raw JSON text."""
    client = OpenAI()
    model = os.getenv("SLOT_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Plan bundle slots for this pet." + 
                json.dumps({"pet_profile": context}, ensure_ascii=False)
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



# ----------------------------
# Planner node
# ----------------------------

def plan_slots_agent(state: SessionStateModel) -> SessionStateModel:
    """LLM-planned slots written into state.slots."""
    ctx = agent_helper_function.pet_context_helper(state.pet_profile)
    raw = _call_llm(ctx)

    tries = 0
    while True:
        tries += 1
        try:
            planned = _coerce_slots_from_json(raw)
            break
        except Exception as e:
            if tries >= 2:
                raise
            raw = _repair_json(ctx, str(e), raw)
    
    state.bundle_state.slots = planned
    return state

 
# ----------------------------
# Parse & coerce into domain models
# ----------------------------

def _coerce_slots_from_json(raw_json: str) -> List[SlotSpec]:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        # Get context around the error location
        lines = raw_json.split('\n')
        error_context = ""
        if hasattr(e, 'lineno') and e.lineno and e.lineno <= len(lines):
            start_line = max(0, e.lineno - 3)
            end_line = min(len(lines), e.lineno + 2)
            error_context = "\n".join([
                f"Line {i+1}: {lines[i]}" 
                for i in range(start_line, end_line)
            ])
        
        raise ValueError(
            f"LLM did not return valid JSON: {e}\n"
            f"Error context:\n{error_context}\n"
            f"First 300 chars: {raw_json[:300]}\n"
            f"Last 300 chars: {raw_json[-300:]}"
        )

    if not isinstance(data, dict) or "slots" not in data or not isinstance(data["slots"], list):
        raise ValueError("JSON must contain a top-level 'slots' array")

    slots: List[SessionStateModel.bundle_state.slots.SlotSpec] = []
    for i, s in enumerate(data["slots"]):
        if not isinstance(s, dict):
            continue

        slots.append(
            SlotSpec(
                slot_id=s.get("slot_id") or f"slot_{i+1}",
                role=s.get("role") or "optional",
                name=s.get("name"),
                must_have=bool(s.get("must_have", True)),
                allow_substitution=bool(s.get("allow_substitution", True)),
                product_type=s.get("product_type"),
                reason_for_suggestion=s.get("reason_for_suggestion"),
                query_text=s.get("query_text"),
                soft_terms=s.get("soft_terms") or [],
            )
        )
    # De-duplicate by slot_id, keep first
    seen = set()
    deduped: List[SlotSpec] = []
    for s in slots:
        if s.slot_id in seen:
            continue
        seen.add(s.slot_id)
        deduped.append(s)
    return deduped


def _repair_json(context: dict, error_msg: str, previous_raw: str) -> str:
    """Ask the LLM to return corrected JSON if parsing failed."""
    client = OpenAI()
    model = os.getenv("SLOT_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Your previous JSON did not validate. Fix it and return only corrected JSON.\n\n"
            f"Validation error: {error_msg}\n\n"
            f"Previous JSON:\n{previous_raw}\n\n"
            f"Pet context again for reference:\n{json.dumps({'pet_profile': context}, ensure_ascii=False, indent=2)}"
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
