import subprocess
from langchain_core.tools import tool

@tool
def run_ffmpeg(command: str) -> str:
    """Runs FFmpeg commands to extract audio from meeting videos."""
    if not command.startswith("ffmpeg"):
        return "Error: Command must start with ffmpeg"
    
    try:
        print(f"\n[Running FFmpeg]: {command}\n")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return "Success! Audio extracted properly."
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"System Error: {str(e)}"