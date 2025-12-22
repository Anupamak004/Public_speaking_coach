from ingestion.audio_handler import extract_audio
from ingestion.video_handler import extract_frames
from ingestion.text_handler import extract_text
from features.audio_features import extract_audio_features
from features.video_features import extract_video_features
from features.text_features import extract_text_features

def process_video(video_path):
    audio_path = extract_audio(video_path)
    frame_dir = extract_frames(video_path)

    audio_feat = extract_audio_features(audio_path)
    video_feat = extract_video_features(frame_dir)
    text = extract_text(audio_path)
    text_feat = extract_text_features(text)

    return audio_feat, video_feat, text_feat
