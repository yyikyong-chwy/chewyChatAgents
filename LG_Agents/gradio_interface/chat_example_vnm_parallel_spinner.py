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
OPENAI_TTS_MODEL = "tts-1"  # or "tts-1-hd" for better quality
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

# ------------- OpenAI TTS -------------
async def text_to_speech_openai(text: str) -> str:
    """Convert text to speech using OpenAI TTS."""
    async with client.audio.speech.with_streaming_response.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=text
    ) as response:
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        await response.stream_to_file(out_path)
    
    return out_path

# ------------- Helper: Get Audio Duration -------------
def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    audio = AudioSegment.from_file(audio_path)
    duration = len(audio) / 1000.0  # Convert milliseconds to seconds
    return duration

# ------------- Helper: Sentence Detection -------------
def is_sentence_end(text: str) -> bool:
    """Check if text ends with sentence punctuation."""
    return bool(re.search(r'[.!?]\s*$', text.strip()))

def extract_sentences(text: str, num_sentences: int = 3):
    """Extract first N sentences and return (first_sentences, remaining_text)."""
    sentences = []
    current = ""
    
    for char in text:
        current += char
        if re.search(r'[.!?]\s*$', current):
            sentences.append(current.strip())
            current = ""
            if len(sentences) >= num_sentences:
                break
    
    # Handle remaining text
    remaining = current + text[len("".join(sentences)):]
    
    return " ".join(sentences), remaining.strip()

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

# ------------- OPTIMIZED Voice Chat Handler -------------
async def handle_voice_chat(audio, history):
    """Process voice input - play first 3 sentences, wait for it to finish, then play rest."""
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
    
    # Step 3: Stream LLM and play first 5 sentences ASAP
    full_response = ""
    current_sentence = ""
    sentence_count = 0
    first_sentences = ""
    first_audio_played = False
    first_audio_path = None
    
    async for token in get_llm_response_streaming(messages):
        full_response += token
        current_sentence += token
        
        # Update chat display
        history[-1][1] = full_response
        
        # When a sentence completes
        if is_sentence_end(current_sentence):
            sentence_count += 1
            
            # When we have 5 sentences, generate and play audio immediately
            if not first_audio_played and sentence_count >= 5:
                # Extract first 3 sentences
                first_sentences, _ = extract_sentences(full_response, 5)
                
                # Generate TTS for FIRST 3 sentences using OpenAI
                first_audio_path = await text_to_speech_openai(first_sentences)
                yield history, first_audio_path
                first_audio_played = True
            
            current_sentence = ""
        else:
            yield history, None
    
    # Step 4: After streaming completes
    if not first_audio_played:
        # Less than 3 sentences total - play full response
        audio_path = await text_to_speech_openai(full_response)
        yield history, audio_path
    else:
        # Wait for first audio to finish playing
        if first_audio_path:
            duration = get_audio_duration(first_audio_path)
            await asyncio.sleep(duration)
        
        # Generate audio for REMAINING text (skip first 3 sentences)
        remaining_text = full_response[len(first_sentences):].strip()
        
        if remaining_text:
            # Play the remaining text (without first 3 sentences)
            audio_path = await text_to_speech_openai(remaining_text)
            yield history, audio_path
        else:
            # No remaining text
            yield history, None

# ============== Gradio UI ==============
with gr.Blocks(title="Voice Chat") as demo:
    gr.Markdown("# 🎙️ Voice Chat Assistant (OpenAI TTS)")
    gr.Markdown("Using **OpenAI TTS** for high-quality voice synthesis!")
    
    chatbot = gr.Chatbot(label="Conversation", height=400, type='tuples')
    
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
    - **Speech-to-Text**: faster-whisper ({WHISPER_MODEL_NAME}) - LOCAL
    - **Text-to-Speech**: OpenAI TTS ({OPENAI_TTS_VOICE}) - CLOUD ☁️
    - **LLM**: OpenAI GPT-4o-mini (streaming)
    
    **How it works:**
    1. First 3 sentences complete → TTS generated → Plays immediately ⚡
    2. **Waits for first audio to finish** ⏱️
    3. Rest of response completes → TTS generated for remaining text → Plays
    4. **No interruption** - second audio waits for first to complete!
    
    **Voice Options:**
    You can change the voice by modifying `OPENAI_TTS_VOICE`:
    - **alloy** - Neutral, balanced
    - **echo** - Male, clear
    - **fable** - British accent
    - **onyx** - Deep male voice
    - **nova** - Female, friendly (current)
    - **shimmer** - Soft female voice
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
# ```

# **Key changes:**

# 1. ✅ **Track sentence count**: `sentence_count` increments each time a sentence completes
# 2. ✅ **Wait for 3 sentences**: Only triggers audio when `sentence_count >= 3`
# 3. ✅ **Extract first 3 sentences**: Uses `extract_sentences()` helper function
# 4. ✅ **Play remaining text**: Skips the first 3 sentences in the second audio

# **User experience:**
# ```
# User speaks → "Tell me about dogs"

# LLM streams:
# - Sentence 1: "Dogs are loyal companions."
# - Sentence 2: "They come in many breeds."
# - Sentence 3: "Dogs require daily exercise."
#   → 🔊 Plays sentences 1-3 together

# - Sentence 4: "They are known for their intelligence."
# - Sentence 5: "Many families choose dogs as pets."
#   → Waits for first audio to finish
#   → 🔊 Plays sentences 4-5 together