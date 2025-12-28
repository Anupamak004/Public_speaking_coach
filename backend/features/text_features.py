import torch
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel

# -------- Load BERT once --------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# ===============================
# 1️⃣ Semantic Embedding (BERT)
# ===============================
def extract_text_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze().numpy()   # (768,)

# ===============================
# 2️⃣ Fluency Features
# ===============================
def extract_fluency_features(text, segments):
    filler_patterns = [
        r"\bum+\b",
        r"\buh+\b",
        r"\ber+\b",
        r"\bah+\b",
        r"\blike\b",
        r"\byou know\b",
        r"\bi mean\b",
        r"\bso\b",
        r"\bokay\b"
    ]

    text_lower = text.lower()

    # -------- Filler count (regex-based) --------
    filler_count = 0
    for pattern in filler_patterns:
        filler_count += len(re.findall(pattern, text_lower))

    # -------- Word count --------
    words = re.findall(r"\b\w+\b", text_lower)
    total_words = len(words)

    filler_ratio = filler_count / max(total_words, 1)

    # -------- Pause & speech rate using segments --------
    pauses = []
    long_pauses = 0

    for i in range(1, len(segments)):
        pause = segments[i]["start"] - segments[i-1]["end"]
        if pause > 0:
            pauses.append(pause)
        if pause > 1.5:
            long_pauses += 1

    avg_pause = np.mean(pauses) if pauses else 0

    duration = segments[-1]["end"] if segments else 60
    speech_rate = total_words / duration        # words/sec
    filler_rate = (filler_count / duration) * 60  # per minute

    return np.array([
        filler_count,
        filler_ratio,
        filler_rate,
        speech_rate,
        avg_pause,
        long_pauses
    ])
