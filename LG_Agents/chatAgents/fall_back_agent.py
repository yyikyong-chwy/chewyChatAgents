# fall_back_agent.py
from __future__ import annotations
import os
import re
import json
from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage

from openai import OpenAI  # optional polish; safe to run without API key
from LG_Agents.states.sessionState import SessionStateModel
from LG_Agents.states.ChewyJourneyChatState import ChewyJourneyChatState

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
FALLBACK_MODEL = os.getenv("CJ_FALLBACK_MODEL", DEFAULT_MODEL)
#USE_LLM_POLISH = os.getenv("CJ_FALLBACK_USE_LLM", "0") in {"1", "true", "True"}
USE_LLM_POLISH = True

# ---------------------------
# Heuristic detection
# ---------------------------
URGENT_MED = re.compile(
    r"\b(blood|bloody stool|seizure|collapsed|not breathing|poison(ed|ing)|antifreeze|xylitol|"
    r"hit by (a )?car|chocolate (bar|ingested)|parvo|distemper|bloat|unable to urinate)\b",
    re.I,
)
BEHAVIOR = re.compile(
    r"\b(chew(ed|ing)?|destroy(ed|ing)?|couch|sofa|shoes?|bark(ing)?|dig(ging)?|jump(ing)?|"
    r"house\s*(soil|soiling)|potty|crate training|anxious|separation anxiety)\b",
    re.I,
)
OFFTOPIC = re.compile(
    r"\b(election|politic(s|al)|president|congress|stock(s| market)?|bitcoin|nba|nfl|mlb|soccer|"
    r"weather (today|tomorrow)|celebrity|movie|gaming|video game|tech news)\b",
    re.I,
)

def _scenario(text: str) -> str:
    if not text or not text.strip():
        return "ambiguous"
    if URGENT_MED.search(text):
        return "urgent"
    if BEHAVIOR.search(text):
        return "behavior"
    if OFFTOPIC.search(text):
        return "offtopic"
    # If the router placed us here, it was neither clinic/pharmacy/product with confidence.
    # Treat as gentle steer.
    return "ambiguous"

# ---------------------------
# Response builders
# ---------------------------
def _species_label(species: Optional[str]) -> str:
    return "dog" if (species or "").lower() not in {"cat"} else "cat"

def _behavior_response(cjs: ChewyJourneyChatState, state: SessionStateModel) -> str:
    species = getattr(state.pet_profile, "species", None)
    pet_name = getattr(state.pet_profile, "pet_name", None)
    age_mo = getattr(state.pet_profile, "age_months", None)

    young = (age_mo is not None and age_mo <= 14)  # rough “teething/teen” window
    who = pet_name or f"your {_species_label(species)}"

    tips: List[str] = []
    if young:
        tips.append("offer a couple of durable teething chews and rotate them to keep interest")
    else:
        tips.append("provide tough chew toys and a daily chew session to channel the urge")
    tips.extend([
        "add a boredom-buster (treat puzzle or lick mat) to burn mental energy",
        "use a bitter no-chew furniture spray where the damage happens",
        "get more daytime exercise + short training reps (sit/place) before downtime",
    ])
    # Soft product nudges (no medical claims)
    nudges = [
        "durable chew toys (e.g., rubber or nylon)",
        "teething rings / puppy chews" if young else "long-lasting chews",
        "bitter deterrent spray for furniture",
        "calming treats (OTC) for stressful moments",
        "crate or playpen setup to manage when you can’t supervise",
    ]

    lines = [
        f"Oof—that’s frustrating. When {who} chews up furniture, it’s usually teething, stress, or extra energy.",
        "Here are a few quick wins:",
        *[f"• {t}" for t in tips],
        "Want me to pull options like "
        + ", ".join(nudges[:3])
        + (", or calming treats?" if "calming" in " ".join(nudges) else "?"),
        "If you’re worried this is more than behavior, I can loop in our clinic team for guidance."
    ]
    return " ".join(lines)

def _offtopic_response(cjs: ChewyJourneyChatState, state: SessionStateModel) -> str:
    examples = [
        "“Help me pick grain-free food for a sensitive stomach (25 lb dog).”",
        "“Refill my Simparica Trio prescription.”",
        "“Recommend a leak-proof litter box for a small apartment.”",
        "“My cat’s been sneezing—should I talk to a vet?”",
    ]
    lines = [
        "I can’t help with that topic, but I *can* help with your pet journey at Chewy—products, prescriptions, and quick vet guidance.",
        "Here are a few things I can do right now:",
        "• Product recommendations (food, toys, litter, beds, crates, training aids)",
        "• Pharmacy help (Rx refills, vet authorization status, dosing questions)",
        "• Clinic triage for non-emergencies (what’s normal vs. when to see a vet)",
        "Try one of these or tell me what you need:",
        *[f"• {ex}" for ex in examples],
    ]
    return " ".join(lines)

def _ambiguous_response(cjs: ChewyJourneyChatState, state: SessionStateModel) -> str:
    lines = [
        "I’m here to help with your pet—products, pharmacy refills, or clinic questions.",
        "Tell me what you’re trying to do (shop for something, check an Rx, or ask a health question), and I’ll route you to the right place.",
    ]
    return " ".join(lines)

def _urgent_response(cjs: ChewyJourneyChatState, state: SessionStateModel) -> str:
    lines = [
        "That could be urgent. I recommend contacting a vet or emergency clinic right away.",
        "If it’s not an immediate emergency, I can connect you to our clinic team for quick guidance. What symptoms are you seeing and for how long?",
    ]
    return " ".join(lines)

# ---------------------------
# Optional LLM polish (keeps wording friendly & concise)
# ---------------------------
_SYSTEM = """You are the Chewy fall-back agent.
Goals:
1) Be empathetic and concise (3–6 sentences).
2) If behavior-related, give 3–5 practical steps and 1–2 shopping nudges (no medical claims).
3) If off-topic, explain what you *can* help with and give 3–4 example prompts.
4) If urgent, advise immediate vet care and offer to route to clinic (no diagnosis).
Tone: warm, professional, no emojis, no guarantees. End with a simple next-step question.
Output: plain text only.
"""

def _polish_with_llm(model: str, system_prompt: str, text: str) -> str:
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.4,
        )
        return (resp.choices[0].message.content or "").strip() or text
    except Exception:
        return text

# ---------------------------
# MAIN NODE
# ---------------------------
def fall_back_agent(state: SessionStateModel) -> SessionStateModel:
    """
    Handles off-topic or unclear user input.
    - Sympathetic guidance for pet behavior complaints (e.g., “dog chewed my couch”).
    - Gentle steering back to Chewy-relevant topics when totally unrelated.
    - Escalation hint for possible urgent medical keywords.
    Writes a friendly reply to cjs.agent_response and keeps route='fallback'/domain='smalltalk'.
    """
    cjs: ChewyJourneyChatState = state.chewy_journey_chat_state
    user_msg = (cjs.last_user_message or "").strip()

    tag = _scenario(user_msg)
    if tag == "urgent":
        reply = _urgent_response(cjs, state)
    elif tag == "behavior":
        reply = _behavior_response(cjs, state)
    elif tag == "offtopic":
        reply = _offtopic_response(cjs, state)
    else:
        reply = _ambiguous_response(cjs, state)

    # Optionally polish with LLM for tone/conciseness
    if USE_LLM_POLISH:
        reply = _polish_with_llm(FALLBACK_MODEL, _SYSTEM, reply)

    #return state
    return {
        "messages": [AIMessage(content=reply)],
            "chewy_journey_chat_state": {
                "agent_response": reply,                
                "route": "fallback",
                    "confidence": 1.0,
                    "debug": {
                        "scenario": tag,
                        "used_llm_polish": USE_LLM_POLISH,
                    },
            }
    }

# ---- Local smoke test
if __name__ == "__main__":
    from pathlib import Path
    from LG_Agents.states.sessionState import load_state

    json_path = Path(__file__).resolve().parent.parent / "gradio_interface" / "last_session-3.json"
    state = load_state(json_path)

    tests = [
        "my dog has chewed up all my couch the other day",
        "what do you think about the election results",
        "my cat collapsed and is barely breathing",
        "hello?",
        "how is the weather in San Francisco?",
    ]
    for t in tests:
        state.chewy_journey_chat_state = ChewyJourneyChatState(last_user_message=t)
        s2 = fall_back_agent(state)
        print("\nUSER:", t)
        print("REPLY:", s2.chewy_journey_chat_state.agent_response)
        print("DEBUG:", json.dumps(s2.chewy_journey_chat_state.debug.get("fallback", {}), indent=2))
