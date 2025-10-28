import gradio as gr
import os
import sys
import asyncio
import tempfile
import re
import subprocess
from pydub import AudioSegment

# --- NEW: Local STT (faster-whisper) ---
from faster_whisper import WhisperModel

# --- NEW: Local TTS (Piper Python API) + auto-download (Option 3) ---
from piper import PiperVoice

# =============== CONFIG ===============
# Piper voice you want; change if desired (see `python -m piper.download_voices`)
PIPER_VOICE_NAME = "en_US-amy-medium"

# Where to keep downloaded voices
PIPER_VOICE_DIR = os.path.join(os.getcwd(), "piper_voices")
PIPER_MODEL_PATH = os.path.join(PIPER_VOICE_DIR, f"{PIPER_VOICE_NAME}.onnx")
PIPER_CONFIG_PATH = PIPER_MODEL_PATH + ".json"

# Faster-Whisper model (choose tiny/base/small/medium/large-v3 or language-specific)
WHISPER_MODEL_NAME = "base.en"  # good default for English
WHISPER_COMPUTE_TYPE = "int8"   # "int8" or "float16" or "int8_float16"
# =====================================

# ---------------- UI Spinner ----------------
SPINNER_HTML = """
<div style="display:flex;align-items:center;gap:8px;">
  <div style="
    width:14px;height:14px;border:3px solid #e5e7eb;
    border-top-color:#6b7280;border-radius:50%;
    animation:spin 1s linear infinite;"></div>
  <span>⏳ {msg}</span>
</div>
<style>
@keyframes spin { 0% { transform: rotate(0deg);} 100% { transform: rotate(360deg);} }
</style>
"""

def thinking(msg="Thinking…"):
    return SPINNER_HTML.format(msg=msg)

# ------------- Helpers: Piper voice -------------
def ensure_piper_voice():
    """Option 3: auto-download Piper voice if missing."""
    model_exists = os.path.exists(PIPER_MODEL_PATH)
    cfg_exists = os.path.exists(PIPER_CONFIG_PATH)
    if model_exists and cfg_exists:
        return

    os.makedirs(PIPER_VOICE_DIR, exist_ok=True)
    # Use the current Python to run the downloader module
    # `--output-dir` ensures both .onnx and .onnx.json land in PIPER_VOICE_DIR
    cmd = [
        sys.executable, "-m", "piper.download_voices",
        PIPER_VOICE_NAME,
        "--output-dir", PIPER_VOICE_DIR
    ]
    subprocess.run(cmd, check=True)
    # sanity check
    if not (os.path.exists(PIPER_MODEL_PATH) and os.path.exists(PIPER_CONFIG_PATH)):
        raise FileNotFoundError(
            f"Downloaded voice files not found: {PIPER_MODEL_PATH} / {PIPER_CONFIG_PATH}"
        )

# Load models once at startup
ensure_piper_voice()
PIPER_VOICE = PiperVoice.load(PIPER_MODEL_PATH)  # finds .onnx.json next to it

WHISPER = WhisperModel(WHISPER_MODEL_NAME, compute_type=WHISPER_COMPUTE_TYPE)

# ------------- Local STT using faster-whisper -------------
async def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using faster-whisper (local)."""
    def _do_transcribe():
        segments, info = WHISPER.transcribe(audio_path, beam_size=1)
        return " ".join(seg.text for seg in segments)
    return await asyncio.to_thread(_do_transcribe)

# ------------- Local TTS using Piper (Python API) -------------
async def text_to_speech(text: str) -> str:
    """Synthesize speech to WAV using Piper, return path to .wav file."""
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    def _synth():
        import wave
        with wave.open(out_path, "wb") as wav:
            PIPER_VOICE.synthesize_wav(text, wav)
    await asyncio.to_thread(_synth)
    return out_path

# If you prefer MP3 output (Gradio works fine with WAV though), you can wrap a converter:
def wav_to_mp3(wav_path: str) -> str:
    mp3_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")
    try:
        os.remove(wav_path)
    except Exception:
        pass
    return mp3_path

# ------------- Sentence boundary -------------
def is_sentence_end(text: str) -> bool:
    return bool(re.search(r'[.!?]\s*$', text.strip()))

# ============== LLM (placeholder streaming) ==============
# Keep your existing OpenAI chat streaming if you want hybrid setup.
# If you’re using a LangGraph agent for streaming, plug it here.
# Below is a simple placeholder generator that yields tokens;
# replace with your get_llm_response_streaming(messages).
async def get_llm_response_streaming(messages):
    """
    REPLACE WITH YOUR REAL LLM STREAM.
    For demonstration, this yields a fixed string token-by-token.
    """
    reply = "This is a local TTS + local STT demo response. It streams sentences nicely."
    for ch in reply:
        await asyncio.sleep(0.01)
        yield ch

# ============== Streaming Pipelines ==============
async def process_voice_input_parallel(audio, history):
    """Parallel TTS generation: starts TTS on each completed sentence while LLM still streams."""
    if audio is None:
        yield history, None, ""
        return

    # Start: transcribing
    yield history, None, thinking("Transcribing…")

    # STT
    user_text = await transcribe_audio(audio)

    history = history + [[user_text, ""]]
    yield history, None, thinking("Generating response…")

    # Build chat history into messages (keep structure from your original if needed)
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for user_msg, assistant_msg in history[:-1]:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": user_text})

    # Stream LLM + sentence-chunk TTS
    full_response = ""
    current_sentence = ""
    tts_tasks = []

    async for token in get_llm_response_streaming(messages):
        full_response += token
        current_sentence += token

        # Update UI text
        history[-1][1] = full_response
        yield history, None, thinking("Generating response…")

        if is_sentence_end(current_sentence):
            sentence = current_sentence.strip()
            if sentence:
                tts_tasks.append(asyncio.create_task(text_to_speech(sentence)))
                current_sentence = ""
                yield history, None, thinking("Generating speech…")

    # Handle tail without punctuation
    if current_sentence.strip():
        tts_tasks.append(asyncio.create_task(text_to_speech(current_sentence.strip())))
        yield history, None, thinking("Generating speech…")

    audio_files = []
    if tts_tasks:
        audio_files = await asyncio.gather(*tts_tasks)

    # Merge WAVs
    final_audio = None
    if audio_files:
        if len(audio_files) == 1:
            final_audio = audio_files[0]
        else:
            combined = AudioSegment.from_wav(audio_files[0])
            for f in audio_files[1:]:
                combined += AudioSegment.from_wav(f)
            out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            combined.export(out_wav, format="wav")
            # cleanup parts
            for f in audio_files:
                try: os.remove(f)
                except: pass
            final_audio = out_wav

    yield history, final_audio, ""

async def process_voice_input_first_sentence(audio, history):
    """First sentence ASAP: speak the first sentence immediately, keep streaming text."""
    if audio is None:
        yield history, None, ""
        return

    yield history, None, thinking("Transcribing…")
    user_text = await transcribe_audio(audio)

    history = history + [[user_text, ""]]
    yield history, None, thinking("Generating first sentence…")

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for user_msg, assistant_msg in history[:-1]:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": user_text})

    full_response = ""
    first_sentence = ""
    first_audio_generated = False
    audio_path = None

    async for token in get_llm_response_streaming(messages):
        full_response += token

        if not first_audio_generated:
            first_sentence += token
            if is_sentence_end(first_sentence):
                yield history, None, thinking("Generating speech…")
                audio_path = await text_to_speech(first_sentence.strip())
                first_audio_generated = True
                history[-1][1] = full_response
                yield history, audio_path, thinking("Continuing response…")

        history[-1][1] = full_response
        if first_audio_generated:
            yield history, audio_path, thinking("Continuing response…")
        else:
            yield history, None, thinking("Generating first sentence…")

    if not first_audio_generated:
        yield history, None, thinking("Generating speech…")
        audio_path = await text_to_speech(full_response)

    yield history, audio_path, ""

# ============== Gradio UI ==============
with gr.Blocks(title="Voice Chat - Local STT/TTS + Spinner") as demo:
    gr.Markdown("# ⚡ Voice Chat (Local STT/TTS) with Streaming + Spinner")
    gr.Markdown("Using **faster-whisper** (local STT) and **Piper** (local TTS).")

    chatbot = gr.Chatbot(label="Conversation", height=400)

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="🎤 Speak your message"
        )
        audio_output = gr.Audio(
            label="🔊 AI Voice Response (WAV)",
            autoplay=True
        )

    status = gr.HTML("", label="Status")

    with gr.Row():
        mode_radio = gr.Radio(
            choices=["Parallel (All Sentences)", "First Sentence Fast"],
            value="First Sentence Fast",
            label="Processing Mode"
        )

    with gr.Row():
        clear_btn = gr.Button("Clear Conversation")

    gr.Markdown("""
    ### Notes
    - **STT**: faster-whisper (`{model}` / `{ctype}`)
    - **TTS**: Piper (`{voice}`) — will **auto-download** if not present (Option 3)
    - Outputs **WAV** by default (best compatibility). You can convert to MP3 with `wav_to_mp3`.
    """.format(model=WHISPER_MODEL_NAME, ctype=WHISPER_COMPUTE_TYPE, voice=PIPER_VOICE_NAME))

    async def process_with_mode(audio, history, mode):
        if mode == "Parallel (All Sentences)":
            async for chat, audio_out, status_html in process_voice_input_parallel(audio, history):
                yield chat, audio_out, status_html
        else:
            async for chat, audio_out, status_html in process_voice_input_first_sentence(audio, history):
                yield chat, audio_out, status_html

    audio_input.stop_recording(
        fn=process_with_mode,
        inputs=[audio_input, chatbot, mode_radio],
        outputs=[chatbot, audio_output, status]
    )

    clear_btn.click(
        fn=lambda: ([], None, ""),
        outputs=[chatbot, audio_output, status]
    )

if __name__ == "__main__":
    demo.launch()
