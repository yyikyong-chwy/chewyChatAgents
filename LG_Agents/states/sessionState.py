from __future__ import annotations
from typing import List, Optional, Literal, Dict
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


from pathlib import Path
import json
import re
from langchain_core.messages import messages_from_dict, messages_to_dict
from langgraph.graph.message import add_messages
from typing import Annotated

#custom imports
from LG_Agents.states.customerState import Customer
from LG_Agents.states.petState import PetProfile
from LG_Agents.states.bundleState import bundleState
from LG_Agents.states.ChewyJourneyChatState import ChewyJourneyChatState
from LG_Agents.states.reducers import merge_chewy_chat_state

# ---------------------------
# Top-level Session State
# ---------------------------
class SessionStateModel(BaseModel):
    customer: Customer
    pet_profile: PetProfile
    bundle_state: bundleState
    chewy_journey_chat_state: Annotated[ChewyJourneyChatState, merge_chewy_chat_state] #specifies that the chewy_journey_chat_state should be merged with the merge_chewy_chat_state reducer

    #reducers only work at the top most level of the state. this is taken out from the chewy_journey_chat_state
    messages: Annotated[list, add_messages]  = Field(default_factory=list)

    schema_version: int = 1

#helper functions. its undecided if it should reside here or in agent_helper_function.py
def save_state(state: SessionStateModel, path: str | Path):
    
    
    #when passed here, state is no longer a Pydantic model, but a plain dict
    state = SessionStateModel.model_validate(state)

    base_path = Path(path)
    customer_id = state.customer.id

    if customer_id:
        customer_slug = slugify_name(customer_id)
        final_path = base_path.with_name(f"{base_path.stem}-{customer_slug}{base_path.suffix}")
    else:
        final_path = base_path  # fallback if customer_id is missing

    state.chewy_journey_chat_state.messages = messages_to_dict(state.chewy_journey_chat_state.messages)


    # Use Pydantic v2 if available, else fallback
    json_text = state.model_dump_json(
        indent=2,
    )
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    tmp.write_text(json_text, encoding="utf-8")
    tmp.replace(final_path)  # atomic on most OSes
    
    print(f"✅ Saved state to {final_path}")


def load_state(path: str | Path) -> SessionStateModel:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) #data would be a plain dict, need convert to pydantic model below    
    return SessionStateModel.model_validate(data)

def slugify_name(name: str, max_len: int = 32) -> str:
    """Make a filesystem-safe slug from the pet name."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return (slug or "pet")[:max_len]


def _weather_brief(state: SessionStateModel) -> Optional[str]:
    w = getattr(state.customer, "weather_now", None)
    if w is None:
        return None
    if isinstance(w, str):
        return w
    if isinstance(w, dict):
        # try to assemble "Sunny 74°F" style
        summary = w.get("summary") or w.get("condition") or w.get("desc")
        temp_f = w.get("temp_f") or w.get("temperature_f") or w.get("tempF")
        parts = [p for p in [summary, f"{temp_f}°F" if temp_f is not None else None] if p]
        return " ".join(parts) if parts else json.dumps(w, default=str)
    # Pydantic model?
    if hasattr(w, "model_dump"):
        d = w.model_dump()
        return _weather_brief(type("S", (), {"customer": type("C", (), {"weather_now": d})})())  # recurse on dict
    return str(w)  # fallback

#Fake example
if __name__ == "__main__":
    state = SessionStateModel(
        customer=Customer(id="123", name="John Doe", email="john.doe@example.com"),
        pet_profile=PetProfile(id="123", name="Luna", species="dog", age_months=12, weight_lb=20),
        bundle_state=bundleState(),
    )
    print(state)
