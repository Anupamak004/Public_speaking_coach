import whisper

# Load once
model = whisper.load_model("base")

def extract_text(audio_path):
    result = model.transcribe(audio_path)
    return result["text"], result["segments"]
