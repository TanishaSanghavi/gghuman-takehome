"""Speech-recognition QA: does the FINISHED video actually say what the script says?

Timing checks (qa.py) prove that sound and picture line up, but not that the sound is *speech*.  A video
built with the babble fallback voice, a mangled line, or a mix that buries the voice would pass every
timing check.  This module closes that gap with an offline Whisper model (tiny.en, run through
sherpa-onnx): every scripted line is cut out of the final mp4's audio at the time the timeline says it is
spoken, transcribed, and compared with the script text.

Lines are transcribed one by one (not the whole video) because tiny Whisper models tend to stop after
the first sentence on clips with long gaps.
"""
import difflib, logging, re
from pathlib import Path
import numpy as np
from .config import MODELS

log = logging.getLogger("gghuman.asr")
MODEL_DIR = MODELS / "sherpa-onnx-whisper-tiny.en"
ASR_SR = 16000
_rec = None

# spelled-out forms Whisper may use for numbers / common variants; the script uses words, Whisper may use digits
_NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"}


def available():
    return all((MODEL_DIR / f).exists() for f in ("tiny.en-encoder.int8.onnx", "tiny.en-decoder.int8.onnx", "tiny.en-tokens.txt"))


def _recognizer():
    global _rec
    if _rec is None:
        import sherpa_onnx
        _rec = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(MODEL_DIR / "tiny.en-encoder.int8.onnx"), decoder=str(MODEL_DIR / "tiny.en-decoder.int8.onnx"),
            tokens=str(MODEL_DIR / "tiny.en-tokens.txt"), num_threads=1, language="en", task="transcribe", decoding_method="greedy_search")
    return _rec


def norm(text):
    t = re.sub(r"[^a-z0-9' ]", " ", text.lower().replace("-", " "))
    out = []
    for w in t.split():
        if w.isdigit() and len(w) > 1 and w not in _NUM:        # Whisper writes "nine six" as "96": spell the digits out
            out += [_NUM[d] for d in w]
        else:
            out.append(_NUM.get(w, w))
    return out


def similarity(expected, heard):
    """0..1 similarity, tolerant of ASR near-misses ('badly' -> 'bad lady', 'win' -> 'when'): the better of the word-level
    and the letter-level match.  Real garbage (babble voice, wrong or missing line) scores far below any threshold used."""
    a, b = norm(expected), norm(heard)
    if not a:
        return 1.0
    if not b:
        return 0.0
    word = difflib.SequenceMatcher(None, a, b).ratio()
    letters = difflib.SequenceMatcher(None, "".join(a), "".join(b)).ratio()
    return max(word, letters)


def _resample16(x, sr):
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, ASR_SR)
    return resample_poly(x, ASR_SR // g, sr // g).astype(np.float32)


def transcribe(x16):
    r = _recognizer()
    s = r.create_stream()
    s.accept_waveform(ASR_SR, x16)
    r.decode_stream(s)
    return s.result.text.strip()


def check_lines(x, sr, tl, pad_before=0.12, pad_after=0.30):
    """x: mono float32 decoded from the final mp4.  Returns [(line_id, expected, heard, similarity)]."""
    x16 = _resample16(x, sr)
    out = []
    for t0, t1, who, lid in sorted(tl["speech"]):
        a = max(0, int((t0 - pad_before) * ASR_SR))
        b = min(len(x16), int((t1 + pad_after) * ASR_SR))
        seg = x16[a:b]
        text = next(l["text"] for s in tl["shots"] for l in s["lines"] if l["id"] == lid)
        # whisper wants some lead-in silence; pad with 0.3 s each side
        seg = np.concatenate([np.zeros(int(0.3 * ASR_SR), np.float32), seg, np.zeros(int(0.3 * ASR_SR), np.float32)])
        heard = transcribe(seg)
        out.append((lid, text, heard, similarity(text, heard)))
    return out
