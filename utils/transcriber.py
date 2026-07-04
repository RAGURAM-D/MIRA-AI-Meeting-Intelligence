"""
Whisper ASR Transcription Module - Chapter 4.1
Uses OpenAI Whisper (local, open-source) for speech-to-text.
Model: medium (default) — balances accuracy and speed.
"""

import whisper


def transcribe_audio(wav_path: str, model_size: str = "medium") -> list[dict]:
    """
    Transcribe audio file using OpenAI Whisper (local inference).
    No external API — runs entirely on local machine.

    Args:
        wav_path: Path to preprocessed 16kHz mono WAV file.
        model_size: Whisper model — "base", "medium", or "large".
                    base   = faster, lower accuracy
                    medium = default, best balance
                    large  = highest accuracy, slower

    Returns:
        List of segment dicts:
        [
            {"start": 0.0, "end": 4.5, "text": "Good morning everyone..."},
            ...
        ]
    """
    print(f"[Whisper] Loading {model_size} model...")
    model = whisper.load_model(model_size)

    print(f"[Whisper] Transcribing: {wav_path}")
    result = model.transcribe(wav_path, verbose=False)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })

    print(f"[Whisper] Transcription complete. {len(segments)} segments.")
    return segments
