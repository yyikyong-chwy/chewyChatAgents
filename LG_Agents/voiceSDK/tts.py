import io, wave, os
import simpleaudio as sa
from openai import OpenAI

client = OpenAI()

r = client.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="nova",
    input="Welcome to Chewy! How is Luna doing? I see she's chewing on the couch again.",
    response_format="wav",
)
b = r.read()
with wave.open(io.BytesIO(b), "rb") as wf:
    sa.WaveObject.from_wave_read(wf).play().wait_done()
