"""ffmpeg helpers: chunked parallel frame encoding, concat, mux, probing."""
import json, subprocess, shutil, logging
from pathlib import Path
from .config import W, H, FPS

log = logging.getLogger("gghuman.encode")


def ffmpeg_ok():
    return shutil.which("ffmpeg") and shutil.which("ffprobe")


def open_writer(path, crf=18, preset="veryfast"):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-profile:v", "high", "-g", str(FPS * 2),
           "-keyint_min", str(FPS), "-sc_threshold", "0", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def concat(chunks, out):
    lst = Path(out).with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(c).resolve().as_posix()}'\n" for c in chunks), encoding="utf-8")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)], check=True)
    lst.unlink()


def mux(video, wav, out, offset=0.0, abr="192k"):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video)]
    if offset:
        cmd += ["-itsoffset", f"{offset:.4f}"]
    cmd += ["-i", str(wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", abr, "-ar", "48000", "-ac", "2",
            "-shortest", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def decode_audio(path, sr=48000):
    import numpy as np
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32)
