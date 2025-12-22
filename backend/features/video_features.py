import cv2
import os
import numpy as np
from deepface import DeepFace

def extract_video_features(frame_dir):
    emotions = []

    for frame_file in sorted(os.listdir(frame_dir)):
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )
            emotions.append(result[0]["emotion"])
        except:
            continue

    if not emotions:
        return {}

    avg_emotion = {
        emotion: np.mean([e[emotion] for e in emotions])
        for emotion in emotions[0]
    }

    return avg_emotion
