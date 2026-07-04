"""
Speaker Diarization and Segmentation Module - Chapter 4.2
Uses PyAnnote.audio to identify speakers and align with Whisper segments.
"""

import os
import traceback
import torchaudio
from dotenv import load_dotenv


# Compatibility fix for pyannote + torchaudio 2.11
torchaudio.set_audio_backend = lambda x: None
torchaudio.get_audio_backend = lambda: "soundfile"
from pyannote.audio import Pipeline

# Load .env file
load_dotenv()


def diarize_audio(wav_path: str) -> list[dict]:
    """
    Run PyAnnote speaker diarization on audio file.
    Identifies speaker boundaries and assigns speaker labels.

    Requires HuggingFace token in .env:
        HUGGINGFACE_TOKEN=hf_...

    Returns:
        List of speaker interval dicts:
        [
            {"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00"},
            ...
        ]
    """

    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN not set in .env file.")

    print("[PyAnnote] Loading speaker diarization pipeline...")
    print("HF Token loaded:", hf_token[:10] + "...")

    try:
        pipeline = Pipeline.from_pretrained(
             "pyannote/speaker-diarization-3.1",
           use_auth_token=hf_token
        )

        print("✅ Pipeline loaded successfully!")

    except Exception:
        print("\n❌ FULL ERROR:")
        traceback.print_exc()
        raise

    print(f"[PyAnnote] Running diarization on: {wav_path}")

    try:
        diarization = pipeline(wav_path)

    except Exception:
        print("\n❌ DIARIZATION ERROR:")
        traceback.print_exc()
        raise

    intervals = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        intervals.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker,
        })

    print(f"[PyAnnote] Diarization complete. {len(intervals)} speaker segments.")

    return intervals


def align_segments(
    whisper_segments: list[dict],
    speaker_intervals: list[dict]
) -> list[dict]:
    """
    Align Whisper time-stamped text segments with PyAnnote speaker labels.
    """

    unique_speakers = sorted(
        set(s["speaker"] for s in speaker_intervals)
    )

    speaker_map = {
        spk: f"Speaker {i + 1}"
        for i, spk in enumerate(unique_speakers)
    }

    attributed = []

    for seg in whisper_segments:

        seg_start = seg["start"]
        seg_end = seg["end"]

        best_speaker = "Unknown Speaker"
        best_overlap = 0.0

        for interval in speaker_intervals:

            overlap_start = max(seg_start, interval["start"])
            overlap_end = min(seg_end, interval["end"])

            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker_map.get(
                    interval["speaker"],
                    "Unknown Speaker"
                )

        attributed.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg["text"],
            "speaker": best_speaker,
        })

    print(
        f"[Alignment] {len(attributed)} segments attributed to "
        f"{len(unique_speakers)} speakers."
    )

    return attributed