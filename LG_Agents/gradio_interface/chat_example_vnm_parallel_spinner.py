import gradio as gr
import os
import asyncio
import tempfile
import re
from faster_whisper import WhisperModel
from openai import AsyncOpenAI
from pydub import AudioSegment

# =============== CONFIG ===============
WHISPER_MODEL_NAME = "base.en"
WHISPER_COMPUTE_TYPE = "int8"

# OpenAI TTS settings
OPENAI_TTS_MODEL = "tts-1"  # tts-1 is faster than tts-1-hd
OPENAI_TTS_VOICE = "nova"   # Options: alloy, echo, fable, onyx, nova, shimmer
# =====================================

# Initialize clients
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
WHISPER = WhisperModel(WHISPER_MODEL_NAME, compute_type=WHISPER_COMPUTE_TYPE)

# ------------- STT -------------
async def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using faster-whisper."""
    def _transcribe():
        segments, _ = WHISPER.transcribe(audio_path, beam_size=1)
        return " ".join(seg.text for seg in segments)
    return await asyncio.to_thread(_transcribe)

# ------------- TTS (OpenAI) -------------
async def text_to_speech(text: str) -> str:
    """Convert text to speech using OpenAI TTS."""
    response = await client.audio.speech.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=text
    )
    
    # Save to temporary file
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    response.stream_to_file(out_path)
    return out_path

# ------------- Helper: Sentence Detection -------------
def is_sentence_end(text: str) -> bool:
    """Check if text ends with sentence punctuation."""
    return bool(re.search(r'[.!?]\s*$', text.strip()))

# ------------- Streaming LLM Response -------------
async def get_llm_response_streaming(messages):
    """Stream response from OpenAI."""
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# ------------- Main Voice Chat Handler (PLAY FIRST AUDIO ASAP) -------------
async def handle_voice_chat(audio, history):
    """Process voice input and play first audio segment immediately."""
    if audio is None:
        yield history, None
        return
    
    # Step 1: Transcribe user's speech
    user_text = await transcribe_audio(audio)
    
    # Add user message to history
    history = history + [[user_text, None]]
    yield history, None
    
    # Step 2: Build messages for LLM
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for user_msg, assistant_msg in history[:-1]:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": user_text})
    
    # Step 3: Stream LLM and generate TTS for first sentence ASAP
    full_response = ""
    current_sentence = ""
    audio_files = []
    first_audio_played = False
    
    async for token in get_llm_response_streaming(messages):
        full_response += token
        current_sentence += token
        
        # Update chat display
        history[-1][1] = full_response
        
        # When we complete a sentence
        if is_sentence_end(current_sentence):
            sentence = current_sentence.strip()
            if sentence:
                # Generate TTS for this sentence
                audio_path = await text_to_speech(sentence)
                audio_files.append(audio_path)
                
                # Play FIRST audio immediately!
                if not first_audio_played:
                    yield history, audio_path
                    first_audio_played = True
                else:
                    # For subsequent sentences, just update text (no audio update)
                    yield history, None
                
                current_sentence = ""
        else:
            # Just update text while streaming
            yield history, None
    
    # Handle any remaining text
    if current_sentence.strip():
        audio_path = await text_to_speech(current_sentence.strip())
        audio_files.append(audio_path)
        if not first_audio_played:
            yield history, audio_path
            first_audio_played = True
    
    # Combine all audio files for final complete version
    if len(audio_files) > 1:
        combined = AudioSegment.from_mp3(audio_files[0])
        for audio_file in audio_files[1:]:
            combined += AudioSegment.from_mp3(audio_file)
        
        # Export combined audio
        final_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        combined.export(final_path, format="mp3")
        
        # Clean up individual files
        for audio_file in audio_files:
            try:
                os.remove(audio_file)
            except:
                pass
        
        # Yield final complete audio
        yield history, final_path
    elif audio_files:
        # Only one sentence, already played
        yield history, audio_files[0]

# ============== Gradio UI ==============
with gr.Blocks(title="Voice Chat") as demo:
    gr.Markdown("# 🎙️ Voice Chat Assistant (Instant Audio)")
    gr.Markdown("First audio segment plays immediately while rest generates!")
    
    chatbot = gr.Chatbot(label="Conversation", height=400)
    
    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="🎤 Speak your message"
        )
        audio_output = gr.Audio(
            label="🔊 AI Response",
            autoplay=True
        )
    
    clear_btn = gr.Button("Clear Conversation")
    
    gr.Markdown(f"""
    ### Setup
    - **Speech-to-Text**: faster-whisper ({WHISPER_MODEL_NAME})
    - **Text-to-Speech**: OpenAI TTS ({OPENAI_TTS_VOICE})
    - **LLM**: OpenAI GPT-4o-mini (streaming)
    
    **How it works:**
    1. First sentence TTS generated → **plays immediately** ⚡
    2. Remaining sentences generated in background
    3. Final combined audio provided when complete
    4. Minimal perceived lag!
    """)
    
    # Handle voice input
    audio_input.stop_recording(
        fn=handle_voice_chat,
        inputs=[audio_input, chatbot],
        outputs=[chatbot, audio_output]
    )
    
    # Clear button
    clear_btn.click(
        fn=lambda: ([], None),
        outputs=[chatbot, audio_output]
    )

if __name__ == "__main__":
    demo.launch()