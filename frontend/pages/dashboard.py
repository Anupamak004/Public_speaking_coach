import streamlit as st
import os
import pandas as pd
import sys
from pathlib import Path

# ==================================================
# 🎨 LOAD CSS
# ==================================================
def load_css():
    css_path = Path(__file__).parent.parent / "styles" / "dashboard.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ==================================================
# 🔧 PATH SETUP
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from backend.ingestion.audio_handler import extract_audio
from backend.features.audio_features import extract_audio_features
from backend.ingestion.video_handler import extract_frames
from backend.features.video_features import extract_video_features
from backend.ingestion.text_handler import extract_text
from backend.features.text_features import (
    extract_text_embedding,
    extract_fluency_features
)

# ==================================================
# ⚙️ PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="AI Public Speaking Coach",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# 👋 WELCOME BAR WITH RIGHT-ALIGNED HAMBURGER
# ==================================================
user_name = "Anupama"

left, right = st.columns([8, 1])

with left:
    st.markdown(
        f"""
        <div class="welcome-box">
            <h2>Welcome, {user_name}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with right:
    st.markdown("<div class='menu-wrapper'>", unsafe_allow_html=True)
    with st.popover("☰"):
        st.markdown("### Menu")
        option = st.radio(
            "",
            ["Dashboard", "History", "Reports", "Settings"],
            label_visibility="collapsed"
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ==================================================
# 📦 CENTERED CONTENT WRAPPER
# ==================================================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ==================================================
# 🧠 TITLE
# ==================================================
st.markdown("## AI-Based Public Speaking Coach")
st.write("Upload your speech video for **audio, facial expression, and fluency analysis**.")

# ==================================================
# 📤 UPLOAD + 🎬 PREVIEW
# ==================================================
upload_col, preview_col = st.columns([1, 1.5], gap="large")

with upload_col:
    st.markdown("### Upload Video")
    video_file = st.file_uploader(
        "",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )

with preview_col:
    st.markdown("### Video Preview")
    if video_file:
        os.makedirs("temp", exist_ok=True)
        video_path = "temp/input_video.mp4"

        with open(video_path, "wb") as f:
            f.write(video_file.read())

        st.video(video_path)

# ==================================================
# 🚀 ANALYSIS PIPELINE
# ==================================================
if video_file and st.button("🚀 Analyze Speech"):
    progress = st.progress(0)

    # ---------------- AUDIO ----------------
    st.info("🔊 Extracting audio...")
    audio_path = extract_audio(video_path)
    progress.progress(15)

    features = extract_audio_features(audio_path)

    mfcc = features[:13]
    avg_pitch, pitch_var, rms = features[13:16]
    speech_rate, silence_ratio = features[16:18]
    jitter, shimmer = features[18:20]

    st.markdown("### 🔊 Audio Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Pitch (Hz)", f"{avg_pitch:.2f}")
    c2.metric("Pitch Variance", f"{pitch_var:.2f}")
    c3.metric("Loudness (RMS)", f"{rms:.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Speaking Rate", f"{speech_rate:.2f}")
    c5.metric("Silence Ratio", f"{silence_ratio:.2f}")
    c6.metric("Voice Stability (Jitter)", f"{jitter:.4f}")

    st.metric("Amplitude Variation (Shimmer)", f"{shimmer:.4f}")

    mfcc_df = pd.DataFrame({
        "MFCC Coefficient": [f"MFCC {i+1}" for i in range(13)],
        "Mean Value": mfcc
    })

    st.markdown("### 🎼 MFCC Summary")
    st.dataframe(mfcc_df, use_container_width=True)

    progress.progress(35)

    # ---------------- VIDEO ----------------
    st.info("🎭 Extracting facial features...")
    frame_dir = extract_frames(video_path)
    progress.progress(50)

    video_features = extract_video_features(frame_dir)
    progress.progress(70)

    if video_features:
        emo_df = pd.DataFrame(
            video_features["emotion_probs"].items(),
            columns=["Emotion", "Score"]
        ).sort_values("Score", ascending=False)

        st.markdown("### 😀 Facial Emotion Analysis")
        st.dataframe(emo_df, use_container_width=True)
        st.bar_chart(emo_df.set_index("Emotion"))

        c1, c2, c3 = st.columns(3)
        c1.metric("Dominant Emotion", video_features["dominant_emotion"].capitalize())
        c2.metric("Eye Contact Ratio", f"{video_features['eye_contact_ratio']:.2f}")
        c3.metric("Head Motion Score", f"{video_features['head_motion_score']:.2f}")
    else:
        st.warning("No consistent face detected.")

    progress.progress(80)

    # ---------------- TEXT ----------------
    st.info("📝 Transcribing speech...")
    transcript, segments = extract_text(audio_path)

    st.markdown("### 🗣️ Transcription")
    st.text_area("", transcript, height=200)

    extract_text_embedding(transcript)
    fluency = extract_fluency_features(transcript, segments)

    fluency_df = pd.DataFrame({
        "Metric": [
            "Filler Count", "Filler Ratio", "Filler Rate",
            "Speech Rate", "Avg Pause", "Long Pauses"
        ],
        "Value": fluency
    })

    st.markdown("### 📊 Fluency Metrics")
    st.dataframe(fluency_df, use_container_width=True)

    progress.progress(100)
    st.success("✅ Analysis Completed Successfully")

# ==================================================
# CLOSE CENTER CONTAINER
# ==================================================
st.markdown('</div>', unsafe_allow_html=True)
