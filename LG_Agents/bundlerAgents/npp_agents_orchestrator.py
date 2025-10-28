from langgraph.graph import StateGraph, END

import json

#custom imports
from LG_Agents.bundlerAgents.enrich_queries_agent import enrich_queries_agent
from LG_Agents.bundlerAgents.slot_candidates_agent import slot_candidates_agent
from LG_Agents.bundlerAgents.candidate_eval_agent import candidate_eval_agent
from LG_Agents.bundlerAgents.bundle_creator_agent import bundle_creator_agent
from LG_Agents.bundlerAgents.slot_planner_agent import plan_slots_agent
from LG_Agents.bundlerAgents.enrich_queries_agent import enrich_queries_agent
from LG_Agents.states.load_customer_Session import load_langgraph_state_session
from LG_Agents.states.sessionState import save_state, SessionStateModel
from pydantic import BaseModel

from pathlib import Path
import sys
def build_graph():
    graph = StateGraph(SessionStateModel)
    graph.add_node("plan_slots_agent", plan_slots_agent)
    graph.add_node("enrich_queries_agent", enrich_queries_agent)
    graph.add_node("slot_candidates_agent", slot_candidates_agent)
    graph.add_node("candidate_eval_agent", candidate_eval_agent)
    graph.add_node("bundle_creator_agent", bundle_creator_agent)

    graph.set_entry_point("plan_slots_agent")
    graph.add_edge("plan_slots_agent", "enrich_queries_agent")
    graph.add_edge("enrich_queries_agent", "slot_candidates_agent")
    graph.add_edge("slot_candidates_agent", "candidate_eval_agent")
    graph.add_edge("candidate_eval_agent", "bundle_creator_agent")
    graph.add_edge("bundle_creator_agent", END)
    return graph.compile()

if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parents[1]  # ...\LG_Agents
    sys.path.insert(0, str(ROOT))

    # Load example state and run the planner

    state = load_langgraph_state_session(2)
    app = build_graph()
    out: SessionStateModel = app.invoke(state)  # type: ignore[assignment]
    #langgraph is returning a plain dict, not a pydantic model



    save_state(out, "../sessions/last_session.json")

