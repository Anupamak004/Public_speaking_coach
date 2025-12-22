import whisper

# Load once (important for performance)
model = whisper.load_model("base")

def extract_text(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]
