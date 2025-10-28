# cj_agent_orchestrator.py
from __future__ import annotations
from typing import Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage
from uuid import uuid4

#custom imports
from LG_Agents.states.sessionState import SessionStateModel
from LG_Agents.states.ChewyJourneyChatState import ChewyJourneyChatState

from LG_Agents.chatAgents.Greeter_agent import greeter_agent
from LG_Agents.chatAgents.cj_router_agent import cj_router_agent
from LG_Agents.chatAgents.clinic_agent import clinic_agent
from LG_Agents.chatAgents.pharmacy_agent import pharmacy_agent
from LG_Agents.chatAgents.fall_back_agent import fall_back_agent
from LG_Agents.chatAgents.product_enquiry_agent import product_enquiry_agent
from LG_Agents.chatAgents.pet_insurance_agent import pet_insurance_agent
from LG_Agents.chatAgents.quality_gate_agent import quality_gate_agent



# -------- Helpers --------
def _route_selector(state: SessionStateModel) -> str:
    route = (state.chewy_journey_chat_state.route or "").strip().lower()
    if route == "greeter":
        return "greeter_agent"
    if route == "clinic":
        return "clinic_agent"
    if route == "pharmacy":
        return "pharmacy_agent"
    if route == "product":
        return "product_enquiry_agent"
    if route == "insurance":
        return "pet_insurance_agent"
    return "fall_back_agent"

def _should_end(state: SessionStateModel) -> bool:
    msg = (state.chewy_journey_chat_state.last_user_message or "").strip().lower()
    return msg in {"bye", "goodbye", "quit", "exit", "stop", "end"}


# -------- Graph builder --------
def build_cj_graph() -> StateGraph:
    """
    simple loop:
      router -> handler -> router -> handler -> ...
    """
    g = StateGraph(SessionStateModel)

    # Nodes
    g.add_node("cj_router_agent", cj_router_agent)
    g.add_node("greeter_agent", greeter_agent)
    g.add_node("clinic_agent", clinic_agent)
    g.add_node("pharmacy_agent", pharmacy_agent)
    g.add_node("fall_back_agent", fall_back_agent)
    g.add_node("product_enquiry_agent", product_enquiry_agent)
    g.add_node("pet_insurance_agent", pet_insurance_agent)
    g.add_node("quality_gate_agent", quality_gate_agent)

    # Entry
    g.set_entry_point("cj_router_agent")

    # ROUTER → exactly one HANDLER
    g.add_conditional_edges(
        "cj_router_agent",
        _route_selector,
        {
            "greeter_agent": "greeter_agent",
            "clinic_agent": "clinic_agent",
            "pharmacy_agent": "pharmacy_agent",
            "product_enquiry_agent": "product_enquiry_agent",
            "pet_insurance_agent": "pet_insurance_agent",
            "fall_back_agent": "fall_back_agent",
        },
    )

    g.add_edge("clinic_agent", "quality_gate_agent")
    g.add_edge("pharmacy_agent", "quality_gate_agent")
    g.add_edge("product_enquiry_agent", "quality_gate_agent")
    g.add_edge("pet_insurance_agent", "quality_gate_agent")
    g.add_edge("fall_back_agent", "quality_gate_agent")
    #g.add_edge("greeter_agent", "quality_gate_agent")

    g.add_edge("quality_gate_agent", END)
    g.add_edge("greeter_agent", END)
    return g


# -------- Minimal driver API --------

def compile_cj_app():
    graph = build_cj_graph()
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Example: quick manual loop (optional)
if __name__ == "__main__":
    app = compile_cj_app()
    
    from pathlib import Path
    from langgraph.errors import GraphInterrupt
    from LG_Agents.states.sessionState import load_state
    from langchain_core.runnables.graph import MermaidDrawMethod

    # Adjust to your path for a quick run
    #json_path = Path(__file__).resolve().parent.parent / "gradio_interface" / "last_session-3.json"
    json_path = Path(__file__).resolve().parent.parent / "gradio_interface" / "dd1-3.json"
    state = load_state(json_path)

    # Kick off: greeter runs -> WAIT interrupt (handled by your UI normally)
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": "test_session"}}
    app = compile_cj_app()
    state = app.invoke(state, config=config)
    state = SessionStateModel.model_validate(state)
    print("ROUTE:", state.chewy_journey_chat_state.route)
    print("CONF:", state.chewy_journey_chat_state.confidence)
    # print("DEBUG:", json.dumps(state.chewy_journey_chat_state.debug, indent=2))

    
    png_bytes = app.get_graph().draw_mermaid_png(  # xray=True works here too
    draw_method=MermaidDrawMethod.API  # or MermaidDrawMethod.PYPPETEER for offline
    )
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    print("Saved to graph.png")
