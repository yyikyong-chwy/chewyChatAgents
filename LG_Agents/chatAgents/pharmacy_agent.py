
from __future__ import annotations
import os, json
from typing import Dict, Any, Optional

from pathlib import Path
import sys
from openai import OpenAI
from langgraph.graph import StateGraph, END

#custom imports
from LG_Agents.states.sessionState import SessionStateModel




def pharmacy_agent(state: SessionStateModel) -> SessionStateModel:
    pre = (state.chewy_journey_chat_state.agent_response or "").strip()
    state.chewy_journey_chat_state.agent_response = f"{pre}\n\nI can explain how Chewy Pharmacy works and next steps for prescriptions. What’s your question?"
    return state
    return state