# pip install langgraph gradio langchain-core

from typing import TypedDict, Annotated, List
from uuid import uuid4

import gradio as gr
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage


# ── LangGraph state with a reducer ────────────────────────────────────────────
# We store LangChain Message objects (BaseMessage & subclasses) and use the
# `add_messages` reducer so new messages append to conversation history.
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def chat_node(state: ChatState) -> ChatState:
    """Single-node logic with reducer-backed history.
    - On first call (no HumanMessage last), greet.
    - Otherwise, respond to the latest HumanMessage.
    Returning a list of AIMessage objects appends to state via the reducer.
    """
    msgs = state.get("messages", [])

    # If there's no user message yet, greet.
    if not msgs or not isinstance(msgs[-1], HumanMessage):
        return {"messages": [AIMessage(content="Hi there! 👋 Welcome—how can I help today?")]}

    # Otherwise, reply to the latest user message
    user_text = msgs[-1].content
    reply = (
        f"Thanks for your message! You said: “{user_text}”. "
        "What else can I help you with?"
    )
    return {"messages": [AIMessage(content=reply)]}


# ── Build & compile the graph ─────────────────────────────────────────────────
graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


# ── Gradio UI glue ────────────────────────────────────────────────────────────
# Use the modern Chatbot API with type='messages' (OpenAI-style dicts)

def init_session():
    """Start a fresh thread and emit the greeting from the graph."""
    thread_id = str(uuid4())
    out = app.invoke({"messages": []}, config={"thread_id": thread_id})
    # out["messages"] is a list[BaseMessage]; find the AIMessage
    first_ai = next((m.content for m in out["messages"] if isinstance(m, AIMessage)), "")
    # Chatbot(type='messages') expects a list of {role, content} dicts
    return ([{"role": "assistant", "content": first_ai}], thread_id)


def handle_user(chat_history, user_message, thread_id):
    """Send the user's message to the graph and append the reply to the chat.
    `chat_history` is a list of {role, content} dicts (Gradio messages format).
    """
    # Push only the new human message; reducer + MemorySaver hold prior state
    out = app.invoke({"messages": [HumanMessage(content=user_message)]}, config={"thread_id": thread_id})
    #ai_reply = next((m.content for m in out["messages"] if isinstance(m, AIMessage)), "")
    ai_reply = next((m.content for m in reversed(out["messages"]) if isinstance(m, AIMessage)), "")

    chat_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ai_reply},
    ]
    return chat_history, ""  # clear textbox


with gr.Blocks(title="Minimal LangGraph × Gradio Chat (Reducer)") as demo:
    gr.Markdown("## Minimal LangGraph × Gradio Chat (Reducer) A super-simple greeter → wait → respond bot with reducer-based state.")

    chat = gr.Chatbot(type="messages", height=420)
    txt = gr.Textbox(placeholder="Type your message and press Enter…", show_label=False)
    thread_state = gr.State("")  # holds thread_id per browser session

    # First load → show greeting
    demo.load(fn=init_session, outputs=[chat, thread_state])

    # User sends a message
    txt.submit(fn=handle_user, inputs=[chat, txt, thread_state], outputs=[chat, txt])

if __name__ == "__main__":
    demo.launch()
