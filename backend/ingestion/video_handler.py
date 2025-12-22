import cv2
import os

def extract_frames(video_path, frame_dir="frames", fps=0.2, max_frames=20):
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = int(video_fps / fps)

    count, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            cv2.imwrite(f"{frame_dir}/frame_{saved}.jpg", frame)
            saved += 1
            if saved >= max_frames:
                break

        count += 1

    cap.release()
    return frame_dir
