
from __future__ import annotations
import os, json
from typing import Dict, Any, Optional

from pathlib import Path
import sys
from openai import OpenAI
from langgraph.graph import StateGraph, END

#custom imports
from LG_Agents.states.sessionState import SessionStateModel




def order_enquiry_agent(state: SessionStateModel) -> SessionStateModel:
    pre = (state.chewy_journey_chat_state.agent_response or "").strip()
    state.chewy_journey_chat_state.agent_response = f"{pre}\n\nI can help with orders. What’s your order number?"
    return state