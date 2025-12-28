import cv2
import os
import numpy as np
from deepface import DeepFace

def extract_video_features(frame_dir, skip_frames=2):
    """
    MediaPipe-FREE implementation
    Works with your current environment
    """

    emotions_list = []
    total_frames = 0

    frame_files = sorted(os.listdir(frame_dir))

    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        total_frames += 1

        # ---------- Emotion Detection ----------
        if idx % skip_frames == 0:
            try:
                emo = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False
                )
                emotions_list.append(emo[0]["emotion"])
            except:
                pass

    if total_frames == 0:
        return {}

    # ---------- Emotion Aggregation ----------
    emotion_probs = {}
    if emotions_list:
        emotion_probs = {
            e: np.mean([f[e] for f in emotions_list])
            for e in emotions_list[0]
        }

    dominant_emotion = (
        max(emotion_probs, key=emotion_probs.get)
        if emotion_probs else "neutral"
    )

    # ---------- SAFE DEFAULTS ----------
    return {
        "emotion_probs": emotion_probs,
        "dominant_emotion": dominant_emotion,
        "eye_contact_ratio": 0.0,      # disabled safely
        "head_motion_score": 0.0       # disabled safely
    }
