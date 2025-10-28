#from nppState import PetProfile, SlotSpec, nppState
from typing import List
import json
import os
import re
from openai import OpenAI
from typing import Optional
from pathlib import Path
from pydantic import BaseModel

#custom imports
from LG_Agents.states.petState import PetProfile
import LG_Agents.states.sessionState as sessionState

# ----------------------------
# Minimal deterministic helpers (NOT business rules)
# ----------------------------

def _life_stage(species: str, age_months: float) -> str:
    s = (species or "").lower()
    if s == "dog":
        if age_months < 12: return "puppy"
        if age_months >= 96: return "senior"
        return "adult"
    # cat
    if age_months < 12: return "kitten"
    if age_months >= 120: return "senior"
    return "adult"


def _size_class(species: str, weight_lb: float) -> Optional[str]:
    s = (species or "").lower()
    w = weight_lb or 0.0
    if s == "dog":
        if w < 10: return "xs"
        if w < 20: return "s"
        if w < 50: return "m"
        if w < 90: return "l"
        return "xl"
    # cats: coarse bucket
    if w < 7: return "s"
    if w < 15: return "m"
    return "l"


def pet_context_helper(p: PetProfile) -> dict:
    """Slim, explicit context to ground the LLM."""
    return {
        "species": p.species,
        "breed": p.breed,
        "gender": p.gender,
        "age_months": p.age_months,
        "weight_lb": p.weight_lb,
        # "life_stage": _life_stage(p.species, p.age_months),
        # "size_class": _size_class(p.species, p.weight_lb),
        # "location_zip": p.location_zip,
        "habits": getattr(p, "habits", []),
        "recent_conditions": getattr(p, "recent_conditions", []),
        # Support either field name if your profile varies
        "geo_eventcondition": getattr(p, "geo_eventcondition", getattr(p, "geo_condition", [])),
    }



   

def save_state(state: sessionState, path: str | Path):
    
    
    #when passed here, state is no longer a Pydantic model, but a plain dict
    state = sessionState.model_validate(state)

    base_path = Path(path)
    pet_name = getattr(getattr(state, "pet", None), "pet_name", None)
    if pet_name:
        pet_slug = slugify_name(pet_name)
        final_path = base_path.with_name(f"{base_path.stem}-{pet_slug}{base_path.suffix}")
    else:
        final_path = base_path  # fallback if pet_name is missing

    # Use Pydantic v2 if available, else fallback
    json_text = state.model_dump_json(
        indent=2,
    )
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    tmp.write_text(json_text, encoding="utf-8")
    tmp.replace(final_path)  # atomic on most OSes
    
    print(f"✅ Saved state to {final_path}")

def load_state(path: str | Path) -> sessionState:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) #data would be a plain dict, need convert to pydantic model below    
    return sessionState.model_validate(data)

def slugify_name(name: str, max_len: int = 32) -> str:
    """Make a filesystem-safe slug from the pet name."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return (slug or "pet")[:max_len]

