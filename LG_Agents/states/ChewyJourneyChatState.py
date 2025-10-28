
from __future__ import annotations
import os
import json
from typing import Optional, Dict, Any, Literal
from langgraph.graph.message import add_messages
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


# =========================
# Chat State
# =========================
class ChewyJourneyChatState(BaseModel):
    """
    state for a single routed care interaction.
    contains all interactions relating to the customer and pet concerns, as well as agent responses, and routing information
    """
    
    last_user_message: str = Field("", description="Raw latest user utterance")

    confidence: float = Field(0.0, description="0..1 from classifier")
    route: Optional[str] = Field("unknown", description="product|pharmacy|clinic|smalltalk|insurance|unknown")
    agent_response: Optional[str] = Field("", description="Agent response to the user's message")    
    grounded_score: Optional[float] = Field(0.0, description="0..1 from groundedness detector")
    escalate: Optional[bool] = Field(False, description="Whether to escalate to a human")
    debug: Dict[str, Any] = Field(default_factory=dict)

    


