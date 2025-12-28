import librosa
import numpy as np

# ======================================================
# MAIN FEATURE EXTRACTION FUNCTION
# ======================================================

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    # ---------------- BASIC FEATURES ----------------
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Pitch (F0)
    pitch = librosa.yin(y, fmin=60, fmax=300)
    pitch = pitch[~np.isnan(pitch)]
    avg_pitch = np.mean(pitch) if len(pitch) > 0 else 0
    pitch_variance = np.var(pitch) if len(pitch) > 0 else 0

    # Loudness
    rms = librosa.feature.rms(y=y)
    rms_energy = np.mean(rms)

    # ---------------- FLUENCY FEATURES ----------------
    # Speaking rate (onsets/sec)
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    speech_rate = len(onsets) / (len(y) / sr)

    # Silence ratio
    silence_threshold = 0.02
    silence_frames = np.sum(np.abs(y) < silence_threshold)
    silence_ratio = silence_frames / len(y)

    # ---------------- VOICE STABILITY ----------------
    jitter = pitch_variance / avg_pitch if avg_pitch > 0 else 0
    shimmer = np.std(rms)

    # ---------------- FILLER DETECTION (ADVANCED) ----------------
    filler_count, filler_rate = detect_audio_fillers_advanced(y, sr)

    # ---------------- FINAL FEATURE VECTOR ----------------
    features = np.concatenate([
        mfcc_mean,              # 0–12
        [avg_pitch],            # 13
        [pitch_variance],       # 14
        [rms_energy],           # 15
        [speech_rate],          # 16
        [silence_ratio],        # 17
        [jitter],               # 18
        [shimmer],              # 19
        [filler_count],         # 20
        [filler_rate]           # 21
    ])

    return features


# ======================================================
# ADVANCED AUDIO-ONLY FILLER DETECTION
# ======================================================

def detect_audio_fillers_advanced(y, sr):
    """
    Detects filler sounds like 'uh', 'um' using:
    - voiced segment detection
    - duration constraint
    - pitch stability
    - MFCC stability
    """

    # Split into voiced segments
    intervals = librosa.effects.split(y, top_db=25)

    filler_count = 0

    for start, end in intervals:
        segment = y[start:end]
        duration = (end - start) / sr

        # ---- Duration filter (fillers are short) ----
        if duration < 0.2 or duration > 1.0:
            continue

        # ---- Pitch analysis ----
        pitch = librosa.yin(segment, fmin=60, fmax=300)
        pitch = pitch[~np.isnan(pitch)]

        if len(pitch) < 5:
            continue

        pitch_mean = np.mean(pitch)
        pitch_var = np.var(pitch)

        # ---- MFCC stability ----
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        mfcc_var = np.mean(np.var(mfcc, axis=1))

        # ---- Heuristic rules ----
        if (
            80 < pitch_mean < 250 and     # human filler pitch
            pitch_var < 50 and            # stable pitch
            mfcc_var < 20                 # vowel-like sound
        ):
            filler_count += 1

    duration_sec = len(y) / sr
    filler_rate = (filler_count / duration_sec) * 60 if duration_sec > 0 else 0

    return filler_count, filler_rate
