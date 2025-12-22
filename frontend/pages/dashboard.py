import streamlit as st
import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.ingestion.audio_handler import extract_audio
from backend.features.audio_features import extract_audio_features
from backend.ingestion.video_handler import extract_frames
from backend.features.video_features import extract_video_features
from backend.ingestion.text_handler import extract_text
from backend.features.text_features import extract_text_features


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Public Speaking Dashboard", layout="wide")

st.markdown("## 🎥 AI-Based Public Speaking Analysis")
st.write("Upload your speech video for multimodal analysis")

# ---------------- VIDEO UPLOAD ----------------
video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

if video_file:
    os.makedirs("temp", exist_ok=True)
    video_path = "temp/input_video.mp4"

    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.video(video_path)

    if st.button("Analyze Speech"):
        progress = st.progress(0)

        # ---------------- AUDIO ----------------
        st.info("🔊 Extracting audio...")
        audio_path = extract_audio(video_path)
        progress.progress(20)

        features = extract_audio_features(audio_path)
        mfcc = features[:13]
        pitch = features[13]
        rms = features[14]

        st.markdown("### 🔊 Audio Analysis")
        c1, c2 = st.columns(2)
        c1.metric("Average Pitch (Hz)", f"{pitch:.2f}")
        c2.metric("Loudness (RMS)", f"{rms:.4f}")

        mfcc_df = pd.DataFrame({
            "MFCC": [f"MFCC {i+1}" for i in range(13)],
            "Value": mfcc
        })
        st.dataframe(mfcc_df, use_container_width=True)
        progress.progress(50)

        # ---------------- VIDEO ----------------
        st.info("🎭 Extracting facial expressions...")
        frame_dir = extract_frames(video_path)
        progress.progress(70)

        emotions = extract_video_features(frame_dir)
        progress.progress(90)

        if emotions:
            emo_df = pd.DataFrame(emotions.items(), columns=["Emotion", "Score"])
            emo_df = emo_df.sort_values("Score", ascending=False)

            st.markdown("### 😀 Facial Emotion Analysis")
            st.dataframe(emo_df, use_container_width=True)
            st.bar_chart(emo_df.set_index("Emotion"))

            dominant = emo_df.iloc[0]["Emotion"]
            st.metric("Dominant Emotion", dominant.capitalize())

        progress.progress(100)
        st.success("✅ Analysis Completed")

        st.markdown("### 🤖 AI Feedback (Prototype)")
        if pitch > 180:
            st.success("Confident vocal tone detected")
        elif pitch > 120:
            st.info("Moderate vocal confidence")
        else:
            st.warning("Low vocal confidence detected")

        if rms < 0.02:
            st.warning("Low voice projection")
        else:
            st.success("Good voice projection")
        # ---------------- TEXT (ASR) ----------------
        st.info("📝 Transcribing speech (Whisper)...")

        text = extract_text(audio_path)

        st.markdown("### 🗣️ Speech Transcription")
        st.text_area("Recognized Text", text, height=180)

# ---------------- TEXT FEATURES ----------------
        st.info("🧠 Extracting text embeddings (BERT)...")

        text_embedding = extract_text_features(text)

        st.markdown("### 🧠 Text Feature Summary")

        st.write("Embedding Shape:", text_embedding.shape)
        st.write("Sample Embedding Values:", text_embedding[:10])
    