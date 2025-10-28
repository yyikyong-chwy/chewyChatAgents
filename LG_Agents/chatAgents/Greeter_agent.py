
from __future__ import annotations
import os
import json

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from LG_Agents.states.sessionState import SessionStateModel, _weather_brief
from LG_Agents.states.ChewyJourneyChatState import ChewyJourneyChatState
from openai import OpenAI
from langgraph.graph import END

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GREETER_MODEL = os.getenv("GREETER_MODEL", DEFAULT_MODEL)

# =========================
# System Prompts
# =========================
# ---- System prompt (adds weather + personalization instructions) ----
GREETER_SYSTEM_PROMPT = """You are the Greeter Agent for Chewy.

Objectives
- Produce a warm, professional greeting in 1–2 sentences.
- If weather context is provided, naturally include a brief note; omit if not provided.
- If pet_context is provided, optionally include ONE brief, natural reference using ONLY facts explicitly given (e.g., name, species, or provided notes). Do NOT invent behaviors, conditions, or preferences.
- Tone: friendly, concise, and safe—no emojis, no slang, no medical advice or guarantees.
- Output: return ONLY the greeting text (no extra fields or formatting).

Examples (adapt wording to the inputs; do not copy verbatim)
1) Weather + pet with notes
   Given: weather={city:"Austin", state:"TX", temp_f:74, conditions:"Sunny"}
          pet_context={name:"Brutus", species:"dog", notes:"teething"}
   Output: "Welcome to Chewy! Hope you’re enjoying the sunny 74°F in Austin today—how’s Brutus doing; are his teething days easing up?"

2) Weather only
   Given: weather={city:"Seattle", state:"WA", temp_f:61, conditions:"Light rain"}
   Output: "Welcome to Chewy—hope you’re staying dry in Seattle with that light rain today."

3) Pet name only (no notes)
   Given: pet_context={name:"Luna", species:"cat"}
   Output: "Welcome to Chewy! How’s Luna today?"

4) Pet with a gentle note
   Given: pet_context={name:"Milo", species:"dog", notes:"recovering from surgery"}
   Output: "Welcome to Chewy—how’s Milo doing today? Wishing him a smooth recovery."

5) No context
   Given: (no weather, no pet_context)
   Output: "Welcome to Chewy—great to have you here."

6) Avoid inventing details (DO NOT do this)
   Given: pet_context={name:"Brutus"}
   ❌ "Is Brutus still chewing up the couch?"
   ✅ "How’s Brutus today?"
"""

# ---- LLM helpers ----
def _chat_json(model: str, system_prompt: str, payload: dict) -> dict:
    client = OpenAI()
    messages = [
        {"role": "system",
         "content": f"{system_prompt}\n\nYou MUST reply with a single, valid JSON object. Output only JSON."},
        # 👇 This line satisfies the API requirement:
        {"role": "user", "content": "Please respond in json."},
        {"role": "user", "content": json.dumps(payload)},
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def _chat_text(model: str, system_prompt: str, user_payload: dict) -> str:
    client = OpenAI()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,          # keep whatever you prefer
        # 🚫 remove response_format entirely
    )
    return resp.choices[0].message.content or ""


# ---- Greeter node (mutates SessionStateModel.chewy_journey_chat_state) ----
def greeter_agent(state: SessionStateModel) -> SessionStateModel:
    cjs: ChewyJourneyChatState = state.chewy_journey_chat_state
    user_msg = cjs.last_user_message or ""

    payload = {
        "message": user_msg,
        "customer": {
            "first_name": state.customer.first_name,
            "zip_code": state.customer.zip_code,
            "city": getattr(state.customer.geo, "city", None) if state.customer.geo else None,
        },
        "pet": {
            "name": getattr(state.pet_profile, "pet_name", None),
            "species": getattr(state.pet_profile, "species", None),
            "breed": getattr(state.pet_profile, "breed", None),
            "gender": getattr(state.pet_profile, "gender", None),
            "age_months": getattr(state.pet_profile, "age_months", None),
            "weight_lb": getattr(state.pet_profile, "weight_lb", None),
            "habits": getattr(state.pet_profile, "habits", []),
            "recent_conditions": getattr(state.pet_profile, "recent_conditions", []),
            "geo_eventcondition": getattr(state.pet_profile, "geo_eventcondition", []),
        },
        "weather_brief": _weather_brief(state),
    }

    text = _chat_text(GREETER_MODEL, GREETER_SYSTEM_PROMPT, payload)

    return {
        "messages": [AIMessage(content=text)],
        "chewy_journey_chat_state": {
            "agent_response": text,
            "route": "greeter",
            "confidence": 1.0,
            "debug": {},
        }
    }

def main():

    from pathlib import Path
    from LG_Agents.states.sessionState import load_state

    json_path = Path(__file__).resolve().parent.parent / "gradio_interface" / "last_session-3.json"
    state = load_state(json_path)


    # Run the greeter
    state = greeter_agent(state)

    #printing this way since we are not using reducer.
    print("AGENT RESPONSE:", state["chewy_journey_chat_state"]["agent_response"])


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Note: set OPENAI_API_KEY for live model calls.")
    main()