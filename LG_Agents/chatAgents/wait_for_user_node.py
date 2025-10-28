from __future__ import annotations
from langgraph.graph import END
from langgraph.types import interrupt
from LG_Agents.states.sessionState import SessionStateModel

#this is a simple node that will wait for a user reply and return the state
def wait_for_user_node(state: SessionStateModel) -> SessionStateModel:
    interrupt({"awaiting": "user_reply"})
    return {} #return an empty state update