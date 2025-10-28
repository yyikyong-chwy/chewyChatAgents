from __future__ import annotations
from typing import List, Optional, Literal, Dict
from datetime import datetime, timezone
from pydantic import BaseModel, Field

# ---------- Core domain ----------

class PetProfile(BaseModel):
    species: Literal["dog", "cat"] = "dog"
    breed: str
    gender: Literal["male", "female"] = "male"
    pet_name: str
    age_months: float = Field(ge=0)
    weight_lb: float = Field(gt=0)
    habits: List[str] = []
    recent_conditions: List[str] = []
    recent_purchases: List[str] = []
    geo_eventcondition: List[str] = [] #uses tools to get weather and calendar conditions

