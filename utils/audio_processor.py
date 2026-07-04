"""
Audio Ingestion and Preprocessing Module - Chapter 4.1
Uses pydub, librosa, ffmpeg-python to convert and normalize audio.
"""

import os
import librosa
import soundfile as sf
from pydub import AudioSegment

ffmpeg_path = r"C:\Users\GAYATHRIDHAYALAN\Downloads\ffmpeg-2026-06-29-git-de6bcf5c05-full_build\ffmpeg\bin"
if ffmpeg_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.path.pathsep + ffmpeg_path
    
def preprocess_audio(input_path: str) -> str:
    """
    Convert any supported audio/video file to 16kHz mono WAV.
    Required format for both Whisper and PyAnnote.

    Steps:
    1. Extract audio from video (if mp4)
    2. Convert to WAV using pydub
    3. Resample to 16kHz mono using librosa
    4. Trim silence using librosa

    Returns path to processed WAV file.
    """
    base = os.path.splitext(input_path)[0]
    ext = os.path.splitext(input_path)[1].lower()

    # Step 1: Extract audio from video if needed
    if ext == ".mp4":
        print("[AudioProcessor] Extracting audio from video...")
        audio = AudioSegment.from_file(input_path, format="mp4")
        mp3_path = base + "_extracted.mp3"
        audio.export(mp3_path, format="mp3")
        input_path = mp3_path

    # Step 2: Convert to WAV using pydub
    print("[AudioProcessor] Converting to WAV...")
    audio = AudioSegment.from_file(input_path)
    raw_wav_path = base + "_raw.wav"
    audio.export(raw_wav_path, format="wav")

    # Step 3: Resample to 16kHz mono using librosa
    print("[AudioProcessor] Resampling to 16kHz mono...")
    y, sr = librosa.load(raw_wav_path, sr=16000, mono=True)

    # Step 4: Trim leading and trailing silence
    print("[AudioProcessor] Trimming silence...")
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    # Save final processed WAV
    processed_path = base + "_processed.wav"
    sf.write(processed_path, y_trimmed, 16000)

    print(f"[AudioProcessor] Done. Output: {processed_path}")
    return processed_path
