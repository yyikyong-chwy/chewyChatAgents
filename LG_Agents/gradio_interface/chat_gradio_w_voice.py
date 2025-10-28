# pip install langgraph gradio langchain-core openai
from typing import TypedDict, Annotated, List
from uuid import uuid4
from pathlib import Path
import os, io, wave, tempfile

import gradio as gr
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

from LG_Agents.states.sessionState import load_state
from LG_Agents.chatAgents.cj_agents_orchestrator import compile_cj_app
from LG_Agents.states.sessionState import SessionStateModel

# --- NEW: OpenAI client for TTS ---
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
oai_client = OpenAI(api_key=OPENAI_API_KEY)

app = compile_cj_app()

# --------- TTS helpers ----------
# Pick from OpenAI's built-in voices (adjust as you like)
OPENAI_TTS_VOICES = [
    "alloy","ash","ballad","coral","echo","fable","nova","onyx","sage","shimmer","verse"
]

def tts_wav_to_tempfile(text: str, voice: str) -> str:
    """
    Generate WAV bytes from OpenAI TTS and write to a temp file.
    Returns the temp file path that Gradio's Audio can play.
    """
    if not text or not voice:
        return None
    resp = oai_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        response_format="wav",
    )
    audio_bytes = resp.read()
    # Write to a temp .wav file for the Gradio player
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(audio_bytes)
    return tmp_path

def _fmt_if(val, suffix=""):
    return f"{val}{suffix}" if val not in (None, "", [], {}) else "—"

def _markdown_card(title: str, rows: list[tuple[str, str]]) -> str:
    lines = [f"### {title}"]
    for k, v in rows:
        lines.append(f"- **{k}:** {v}")
    return "\n".join(lines)

def customer_markdown(state: SessionStateModel) -> str:
    c = getattr(state, "customer", None)
    if c is None:
        return _markdown_card("Customer", [("Status", "No customer loaded")])

    # Try common fields; show only if present
    first_name = getattr(c, "first_name", None)
    last_name  = getattr(c, "last_name", None)
    name = " ".join([x for x in [first_name, last_name] if x]).strip() or getattr(c, "name", None)

    rows = [
        ("ID", _fmt_if(getattr(c, "id", None))),
        ("Name", _fmt_if(name)),
        ("Email", _fmt_if(getattr(c, "email", None))),
        ("Phone", _fmt_if(getattr(c, "phone", None))),
        # Optional/nice-to-have fields if your model has them:
        ("Loyalty Tier", _fmt_if(getattr(c, "loyalty_tier", None))),
        ("Address (City)", _fmt_if(getattr(c, "city", None))),
        ("State", _fmt_if(getattr(c, "state", None))),
        ("Zip", _fmt_if(getattr(c, "zip", None))),
    ]
    # Filter out rows that are all "—" except ID/Name
    keep = []
    for k, v in rows:
        if k in {"ID", "Name"} or v != "—":
            keep.append((k, v))
    return _markdown_card("Customer", keep)

def pet_markdown(state: SessionStateModel) -> str:
    p = getattr(state, "pet_profile", None)
    if p is None:
        return _markdown_card("Pet", [("Status", "No pet profile loaded")])

    species = getattr(p, "species", None)
    breed = getattr(p, "breed", None)
    gender = getattr(p, "gender", None)
    pet_name = getattr(p, "pet_name", None)
    age_mo = getattr(p, "age_months", None)
    weight_lb = getattr(p, "weight_lb", None)

    # Friendly age/weight render
    if isinstance(age_mo, (int, float)):
        years = int(age_mo // 12)
        months = int(round(age_mo % 12))
        age_txt = f"{years} yr {months} mo" if years else f"{months} mo"
    else:
        age_txt = "—"

    if isinstance(weight_lb, (int, float)):
        kg = round(weight_lb * 0.453592, 1)
        wt_txt = f"{weight_lb:.1f} lb ({kg} kg)"
    else:
        wt_txt = "—"

    # Lists → comma-joined if present
    def join_list(attr):
        val = getattr(p, attr, None)
        return ", ".join(val) if isinstance(val, list) and val else "—"

    rows = [
        ("Name", _fmt_if(pet_name)),
        ("Species", _fmt_if(species)),
        ("Breed", _fmt_if(breed)),
        ("Gender", _fmt_if(gender)),
        ("Age", age_txt),
        ("Weight", wt_txt),
        ("Habits", join_list("habits")),
        ("Recent Conditions", join_list("recent_conditions")),
        ("Recent Purchases", join_list("recent_purchases")),
        ("Geo/Events", join_list("geo_eventcondition")),
    ]
    return _markdown_card("Pet", rows)


# ── LangGraph glue ────────────────────────────────────────────────────────────
def init_session():
    """Start a fresh thread and emit the greeting from the graph."""
    thread_id = str(uuid4())

    # Load the initial state
    json_path = Path(__file__).resolve().parent.parent / "gradio_interface" / "last_session-3.json"
    state = load_state(json_path)
    state = SessionStateModel.model_validate(state)

    # Kick off your app
    config = {"configurable": {"thread_id": "test_session"}}
    state = app.invoke(state, config=config)
    state = SessionStateModel.model_validate(state)

    first_ai = next((m.content for m in state.messages if isinstance(m, AIMessage)), "")

    # ❗️DO NOT mutate components here. Return VALUES (strings) for the Markdown outputs.
    cust_md = customer_markdown(state)
    pet_md  = pet_markdown(state)

    # Return values in the same order as outputs=[chat, app_state, thread_state, customer_card, pet_card]
    return ([{"role": "assistant", "content": first_ai}], state, thread_id, cust_md, pet_md)


def handle_user(chat_history, user_message, thread_id, state, voice, auto_speak):
    """
    Send the user's message to the graph and append the reply to the chat.
    Also optionally TTS the AI reply using selected voice.
    """
    if state is None:
        # Return placeholders for all outputs (player, customer_card, pet_card)
        return chat_history, "", state, gr.update(value=None), customer_markdown(SessionStateModel.model_validate({})), pet_markdown(SessionStateModel.model_validate({}))

    # Push message into your state
    state.chewy_journey_chat_state.last_user_message = user_message

    # Run graph
    config = {"configurable": {"thread_id": "test_session"}}
    state = app.invoke(state, config=config)
    state = SessionStateModel.model_validate(state)

    ai_reply = next((m.content for m in reversed(state.messages) if isinstance(m, AIMessage)), "")

    # Update chat
    chat_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ai_reply},
    ]

    # Optional: auto-speak the reply
    audio_path = None
    if auto_speak and ai_reply:
        try:
            audio_path = tts_wav_to_tempfile(ai_reply, voice or "alloy")
        except Exception:
            audio_path = None

    # ❗️Return VALUES for the Markdown components (or gr.update(value=...))
    cust_md = customer_markdown(state)
    pet_md  = pet_markdown(state)

    # Order must match: outputs=[chat, txt, app_state, player, customer_card, pet_card]
    return chat_history, "", state, gr.update(value=audio_path), cust_md, pet_md


def audition_voice(voice, sample_text):
    """
    Generate a short sample for the selected voice so you can audition it.
    """
    sample = sample_text.strip() or "Hi there! This is a quick voice preview from your Chewy assistant."
    try:
        path = tts_wav_to_tempfile(sample, voice or "alloy")
        return gr.update(value=path)
    except Exception:
        return gr.update(value=None)
# ── Gradio UI ────────────────────────────────────────────────────────────
with gr.Blocks(title="Minimal LangGraph × Gradio Chat (Reducer) + OpenAI TTS") as demo:
    gr.Markdown("## Chat with TTS (OpenAI) — pick a voice, preview, and auto-speak replies")

    with gr.Row():
        # Left: chat
        chat = gr.Chatbot(type="messages", height=420)

        # Right: tabs (MUST be inside Blocks)
        with gr.Column():
            with gr.Tabs():
                # --- Tab 1: Customer & Pet ---
                with gr.TabItem("Customer & Pet"):
                    with gr.Accordion("Customer & Pet", open=True):
                        customer_card = gr.Markdown(value="Loading customer…")
                        pet_card = gr.Markdown(value="Loading pet…")

                # --- Tab 2: Voice & Audio ---
                with gr.TabItem("Voice & Audio"):
                    voice_dd = gr.Dropdown(
                        choices=OPENAI_TTS_VOICES,
                        value="alloy",
                        label="Voice",
                        interactive=True,
                    )
                    auto_ck = gr.Checkbox(label="Auto-speak AI replies", value=True)
                    player_voice = gr.Audio(label="Voice Player", autoplay=True)

                    with gr.Accordion("Audition / Preview voice", open=False):
                        sample_tb = gr.Textbox(
                            value="Hello! I can help with your pet’s needs. How can I assist today?",
                            lines=2,
                            label="Sample text"
                        )
                        audition_btn = gr.Button("Preview selected voice")
                        # IMPORTANT: Event registration stays INSIDE Blocks
                        audition_btn.click(
                            fn=audition_voice,
                            inputs=[voice_dd, sample_tb],
                            outputs=[player_voice],
                        )

    # Input & state (also INSIDE Blocks)
    txt = gr.Textbox(placeholder="Type your message and press Enter…", show_label=False)
    thread_state = gr.State("")   # holds thread_id per browser session
    app_state = gr.State(None)    # holds SessionStateModel

    # First load → show greeting + populate cards
    demo.load(
        fn=init_session,
        inputs=[],
        outputs=[chat, app_state, thread_state, customer_card, pet_card]
    )

    # User sends a message
    # NOTE: audio output targets player_voice in the Voice tab
    txt.submit(
        fn=handle_user,
        inputs=[chat, txt, thread_state, app_state, voice_dd, auto_ck],
        outputs=[chat, txt, app_state, player_voice, customer_card, pet_card]
    )

if __name__ == "__main__":
    # Make sure OPENAI_API_KEY is set in your environment
    demo.launch()
