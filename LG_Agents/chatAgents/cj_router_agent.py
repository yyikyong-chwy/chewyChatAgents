# cj_router_agent.py
from __future__ import annotations
import os
import re
import json
from typing import Dict, Any, Optional
from langchain_core.messages import messages_to_dict, messages_from_dict

from openai import OpenAI
from langchain_core.messages import HumanMessage
from LG_Agents.states.sessionState import SessionStateModel  # state container
from LG_Agents.states.ChewyJourneyChatState import ChewyJourneyChatState  # chat substate

# --------------------
# Config
# --------------------
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ROUTER_MODEL = os.getenv("CJ_ROUTER_MODEL", DEFAULT_MODEL)

# --------------------
# LLM helper
# --------------------
def _chat_json(model: str, system_prompt: str, user_msg: str, previous_messages: list) -> dict:
    """
    Robust JSON-only call that satisfies OpenAI's 'json in messages' requirement.
    Falls back to {} on any error; caller should handle gracefully.
    """
    try:
        client = OpenAI()
        messages = [
            {
                "role": "system",
                "content": f"{system_prompt}\n\nYou MUST reply with a single, valid JSON object. Output only JSON."
            },
            # Satisfy response_format 'json_object' requirement:
            {"role": "user", "content": "Please respond in json."},
            {"role": "user", "content": json.dumps(previous_messages, ensure_ascii=False)},
            {"role": "user", "content": user_msg},
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        return {"error": str(e)}

# --------------------
# Lightweight heuristic (no-API fallback)
# --------------------
_PHARMACY = re.compile(
    r"\b(rx|prescription|refill|meds?|medication|dosage|apoquel|gabapentin|simparica|nexgard|bravecto|sentinel|trifexis|heartworm|flea|tick|pharmacy|script|vet\s*authorization)\b",
    re.I,
)
_CLINIC = re.compile(
    r"\b(vomit(ing)?|diarrhea|letharg(ic|y)|injur(y|ed)|wound|limp|cough(ing)?|sneeze(ing)?|itch(y|ing)|allerg(y|ies)|ear infection|hot\s*spot|rash|urinate|pee|poop|stool|bloody|not\s*eating|loss\s*of\s*appetite|emergenc(y|ies)|tele.?vet|appointment|clinic|vaccin(e|ation)|spay|neuter|surgery|fever|pain)\b",
    re.I,
)
_PRODUCT = re.compile(
    r"\b(food|kibble|wet\s*food|treats?|toy|chew|crate|litter|collar|leash|harness|bed|brand|size|price|in\s*stock|reviews?|recommend|best|compare|sku|autoship|subscribe|flavor|grain[-\s]?free|raw|freeze[-\s]?dried)\b",
    re.I,
)

def _heuristic_route(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"route": "fallback", "confidence": 0.2, "reason": "Empty or no message."}

    scores = {
        "pharmacy": len(_PHARMACY.findall(text)),
        "clinic": len(_CLINIC.findall(text)),
        "product": len(_PRODUCT.findall(text)),
    }
    # choose highest; break ties preferring clinic > pharmacy > product for safety
    order = sorted(scores.items(), key=lambda kv: (kv[1], {"clinic": 3, "pharmacy": 2, "product": 1}[kv[0]]), reverse=True)
    top, top_score = order[0]
    if top_score == 0:
        return {"route": "fallback", "confidence": 0.3, "reason": "No domain keywords matched."}

    # simple confidence mapping
    conf = min(1.0, 0.35 + 0.2 * top_score)
    return {"route": top, "confidence": conf, "scores": scores, "reason": "Keyword match heuristic."}

# --------------------
# Router prompt
# --------------------
_ROUTER_SYSTEM_PROMPT = """You are the CJ Router Agent for Chewy.
Classify the user's latest message into EXACTLY ONE of these routes:
- "clinic": medical/symptom/triage questions, tele-vet, vaccinations, appointments, procedures.
- "pharmacy": prescriptions/Rx, refills, medications, doses, vet authorization, flea/tick/heartworm meds.
- "product": shopping for items, comparisons, recommendations, prices, availability, fit/size, brands.
- "insurance": pet insurance coverage/eligibility/quotes; buying or getting insurance, reimbursements/claims; deductibles; waiting periods; exclusions; wellness add-ons; provider/insurer questions; policy changes/cancellation.
- "fallback": off-topic, chit-chat, or unclear; we will gently steer the user back in a different node.

Return a single JSON object:
{
  "route": "clinic" | "pharmacy" | "product" | "insurance" | "fallback",
  "confidence": 0.0-1.0,
  "intent": "<short phrase>",
  "reasons": ["..."],
  "entities": { "pet_species": "...", "brands": ["..."], "medications": ["..."] }
}

Rules:
- Prefer "clinic" over other categories if symptoms/health risks are present, or questions are relating nearest clinic for the pet.
- Prefer "pharmacy" when the user mentions Rx/refills/medication names/dosing.
- Prefer "insurance" when the user mentions pet insurance coverage/eligibility/quotes; reimbursements/claims; deductibles; waiting periods; exclusions; wellness add-ons; provider/insurer questions; policy changes/cancellation.
- if user express interests in pet insurance, route to "insurance"
- Use "fallback" only when the message is not about clinic/pharmacy/product/insurance.
- Be conservative and do not offer advice; only classify.
"""

# --------------------
# MAIN NODE
# --------------------
def cj_router_agent(state: SessionStateModel) -> SessionStateModel:
    """
    Decide a single downstream route for the conversation and store it on state.chewy_journey_chat_state.route.
    Also sets confidence and debug metadata. Leaves actual response generation to the routed agent.
    """
    cjs: ChewyJourneyChatState = state.chewy_journey_chat_state

    #if there is no last user message, route it to the greeter agent
    if not cjs.last_user_message:
        meta = {"heuristic": "greeter", "llm_error": "unavailable"}
        return {
            "chewy_journey_chat_state": {
                "route": "greeter",
                "confidence": 1.0,
                "debug": meta,
            }
        }

    #perhaps add context from previous messages??? KIV
    user_msg = (cjs.last_user_message or "").strip()

    # 1) Heuristic first (works offline)
    heuristic = _heuristic_route(user_msg)

    # 2) Try LLM for richer classification; fall back to heuristic if API unavailable
    llm = _chat_json(ROUTER_MODEL, _ROUTER_SYSTEM_PROMPT, user_msg, messages_to_dict(state.messages))

    # choose best available signal
    if "route" in llm and llm.get("confidence") is not None:
        chosen_route = str(llm.get("route") or "fallback").lower().strip()
        confidence = float(llm.get("confidence") or 0.0)
        meta = {"llm": llm, "heuristic": heuristic}
    else:
        chosen_route = heuristic["route"]
        confidence = heuristic["confidence"]
        meta = {"heuristic": heuristic, "llm_error": llm.get("error") if isinstance(llm, dict) else "unavailable"}

    # Normalize route to expected set
    if chosen_route not in {"clinic", "pharmacy", "product", "fallback", "greeter", "insurance"}:
        chosen_route = "fallback"


    return {
        "messages": [HumanMessage(content=cjs.last_user_message)],
        "chewy_journey_chat_state": {
            "route": chosen_route,            
            "confidence": confidence,            
            "debug": meta,
        }
    }


# Optional: quick local test
if __name__ == "__main__":
    
    from pathlib import Path
    from LG_Agents.states.sessionState import load_state

    # Minimal smoke test without real state loader; replace with your state loader as needed.
    json_path = Path(__file__).resolve().parent.parent / "gradio_interface" / "last_session-3.json"
    state = load_state(json_path)
    last_user_message="Can I refill my dog's Apoquel prescription?"
    last_user_message="I am curious about the political fallout of the recent elections?"
    last_user_message="My dog has been barking all night, keeping me awake."
    state.chewy_journey_chat_state=ChewyJourneyChatState(last_user_message=last_user_message)    

    state = cj_router_agent(state)
    #the accessor methods below are not working in local test. this is because it is not using reducer. we are not invoking a graph afterall
    #hence it is simply returning a dict, not a pydantic model.
    #even worse, it is simply returning part of a dict that makes up the cjstate, and not the session state
    #state = SessionStateModel.model_validate(state)


    # print("ROUTE:", state.chewy_journey_chat_state.route)
    # print("CONF:", state.chewy_journey_chat_state.confidence)
    # print("DEBUG:", json.dumps(state.chewy_journey_chat_state.debug, indent=2))
