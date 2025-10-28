# pip install langgraph gradio langchain-core

from typing import TypedDict, Annotated, List
from uuid import uuid4

import gradio as gr
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

from pathlib import Path
from LG_Agents.states.sessionState import load_state


from LG_Agents.chatAgents.cj_agents_orchestrator import compile_cj_app
from LG_Agents.states.sessionState import SessionStateModel




app = compile_cj_app()

# ── Gradio UI glue ────────────────────────────────────────────────────────────
# Use the modern Chatbot API with type='messages' (OpenAI-style dicts)

def init_session():
    """Start a fresh thread and emit the greeting from the graph."""
    thread_id = str(uuid4())

    # Load the initial state
    json_path = Path(__file__).resolve().parent.parent / "gradio_interface" / "last_session-3.json"
    state = load_state(json_path)
    state = SessionStateModel.model_validate(state)

    # Kick off: greeter runs -> WAIT interrupt (handled by your UI normally)
    config = {"configurable": {"thread_id": "test_session"}}
    
    #out = app.invoke({"messages": []}, config={"thread_id": thread_id})
    state = app.invoke(state, config=config)
    state = SessionStateModel.model_validate(state)
    # out["messages"] is a list[BaseMessage]; find the AIMessage
    first_ai = next((m.content for m in state.messages if isinstance(m, AIMessage)), "")
    # Chatbot(type='messages') expects a list of {role, content} dicts
    return ([{"role": "assistant", "content": first_ai}], state, thread_id)


def handle_user(chat_history, user_message, thread_id, state):
    """Send the user's message to the graph and append the reply to the chat.
    `chat_history` is a list of {role, content} dicts (Gradio messages format).
    """
    if state is None:
        return chat_history, "", state
    
    # Set the user message in the state
    state.chewy_journey_chat_state.last_user_message = user_message
    
    # Push only the new human message; reducer + MemorySaver hold prior state
    config = {"configurable": {"thread_id": "test_session"}}
    state = app.invoke(state, config=config)
    state = SessionStateModel.model_validate(state)
    #ai_reply = next((m.content for m in out["messages"] if isinstance(m, AIMessage)), "")
    ai_reply = next((m.content for m in reversed(state.messages) if isinstance(m, AIMessage)), "")

    chat_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ai_reply},
    ]
    return chat_history, "", state  # clear textbox


with gr.Blocks(title="Minimal LangGraph × Gradio Chat (Reducer)") as demo:
    gr.Markdown("## Typing and chatting here")

    chat = gr.Chatbot(type="messages", height=420)
    txt = gr.Textbox(placeholder="Type your message and press Enter…", show_label=False)
    thread_state = gr.State("")  # holds thread_id per browser session
    app_state = gr.State(None)  # holds SessionStateModel

    # First load → show greeting
    demo.load(fn=init_session, inputs=[], outputs=[chat, app_state, thread_state])

    # User sends a message
    txt.submit(fn=handle_user, inputs=[chat, txt, thread_state, app_state], outputs=[chat, txt, app_state])

if __name__ == "__main__":
    demo.launch()
