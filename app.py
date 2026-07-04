"""
Autonomous Meeting Summarization System
Flask REST API - Chapter 4.4
"""

import os
import uuid
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from utils.audio_processor import preprocess_audio
from utils.transcriber import transcribe_audio
from utils.diarizer import diarize_audio, align_segments
from agents.pipeline import run_agent_pipeline
from utils.report_generator import generate_pdf_report

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)
print(f"Loaded .env configuration from {env_path}")

REQUIRED_ENV_VARS = ["OPENROUTER_API_KEY", "HUGGINGFACE_TOKEN"]
missing_env_vars = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
if missing_env_vars:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing_env_vars)}. "
        f"Check {env_path} and fill in your keys."
    )

print(f"BASE_DIR={BASE_DIR}")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
TEMPLATES_FOLDER = os.path.join(BASE_DIR, "templates")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

# In-memory job store
jobs = {}


def process_meeting(job_id, audio_path):
    """Background thread: full pipeline from audio to PDF report."""
    try:
        # Stage 1: Audio Preprocessing
        jobs[job_id]["stage"] = "preprocessing"
        jobs[job_id]["progress"] = 10
        print(f"[{job_id}] Starting preprocessing...")
        wav_path = preprocess_audio(audio_path)
        print(f"[{job_id}] Preprocessing complete: {wav_path}")

        # Stage 2: Whisper Transcription
        jobs[job_id]["stage"] = "transcribing"
        jobs[job_id]["progress"] = 25
        print(f"[{job_id}] Starting transcription...")
        segments = transcribe_audio(wav_path)
        print(f"[{job_id}] Transcription complete: {len(segments)} segments")

        # Stage 3: Speaker Diarization
        jobs[job_id]["stage"] = "diarizing"
        jobs[job_id]["progress"] = 45
        print(f"[{job_id}] Starting diarization...")
        speaker_intervals = diarize_audio(wav_path)
        attributed_segments = align_segments(segments, speaker_intervals)
        print(f"[{job_id}] Diarization complete")

               # Save transcript as TXT file
        transcript_path = os.path.join(
            OUTPUT_FOLDER,
            f"{job_id}_transcript.txt"
        )

        with open(
            transcript_path,
            "w",
            encoding="utf-8"
        ) as f:

            for seg in attributed_segments:

                f.write(
                    f"[{seg['speaker']}] {seg['text']}\n"
                )

        # Stage 4: Multi-Agent LangChain Pipeline
        jobs[job_id]["stage"] = "agents"
        jobs[job_id]["progress"] = 65
        print(f"[{job_id}] Starting agent pipeline...")
        report_data = run_agent_pipeline(attributed_segments)
        print(f"[{job_id}] Agent pipeline complete")

        # Stage 5: PDF Report Generation
        jobs[job_id]["stage"] = "generating_report"
        jobs[job_id]["progress"] = 88
        print(f"[{job_id}] Generating PDF report...")
        pdf_path = generate_pdf_report(report_data, job_id, OUTPUT_FOLDER)
        print(f"[{job_id}] PDF generated: {pdf_path}")

        jobs[job_id]["stage"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["report_data"] = report_data
        jobs[job_id]["pdf_path"] = pdf_path
        print(f"[{job_id}] Processing complete!")

    except Exception as e:
        print(f"[{job_id}] ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        jobs[job_id]["stage"] = "error"
        jobs[job_id]["error"] = str(e)


@app.route("/upload", methods=["POST"])
def upload():
    """POST /upload — Accept meeting audio/video file, start processing."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    allowed = {".mp3", ".mp4", ".wav", ".m4a"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format. Use: {allowed}"}), 400

    job_id = str(uuid.uuid4())
    audio_path = os.path.join(UPLOAD_FOLDER, f"{job_id}{ext}")
    file.save(audio_path)

    jobs[job_id] = {
        "stage": "queued",
        "progress": 0,
        "audio_path": audio_path,
    }

    thread = threading.Thread(target=process_meeting, args=(job_id, audio_path))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "status": "processing started"}), 202


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    """GET /status/<job_id> — Return current processing stage and progress."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]
    return jsonify({
        "job_id": job_id,
        "stage": job["stage"],
        "progress": job["progress"],
        "error": job.get("error"),
    })


@app.route("/report/<job_id>", methods=["GET"])
def report(job_id):
    """GET /report/<job_id> — Return structured JSON summary and PDF download link."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]
    if job["stage"] != "completed":
        return jsonify({"error": "Report not ready", "stage": job["stage"]}), 202

    return jsonify({
        "job_id": job_id,
        "report": job["report_data"],
        "pdf_url": f"/download/{job_id}",
         "transcript_url": f"/transcript/{job_id}",
        "audio_url": f"/audio/{job_id}",
    })


@app.route("/download/<job_id>", methods=["GET"])
def download(job_id):
    """GET /download/<job_id> — Serve the generated PDF report."""
    if job_id not in jobs or jobs[job_id]["stage"] != "completed":
        return jsonify({"error": "Report not available"}), 404

    return send_from_directory(OUTPUT_FOLDER, f"{job_id}_report.pdf", as_attachment=True)


@app.route("/audio/<job_id>", methods=["GET"])
def get_audio(job_id):

    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    audio_path = jobs[job_id]["audio_path"]

    folder = os.path.dirname(audio_path)
    filename = os.path.basename(audio_path)

    return send_from_directory(
        folder,
        filename,
        as_attachment=False
    )

@app.route("/transcript/<job_id>", methods=["GET"])
def transcript(job_id):

    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    transcript_path = os.path.join(
        OUTPUT_FOLDER,
        f"{job_id}_transcript.txt"
    )

    if not os.path.exists(transcript_path):
        return jsonify({"error": "Transcript not found"}), 404

    return send_from_directory(
        OUTPUT_FOLDER,
        f"{job_id}_transcript.txt",
        as_attachment=True
    )

@app.route("/")
def index():
    return send_from_directory(TEMPLATES_FOLDER, "index.html")


if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:000")
    app.run(host="0.0.0.0",debug=False, port=5000, threaded=True)
