"""Text-to-speech with a fallback chain, on-disk cache, word alignment and a
lip-sync envelope.  Backends (first available wins, can be forced via env
GGH_TTS=kokoro|espeak|blip):

  kokoro  – Kokoro v0.19 neural TTS through sherpa-onnx (offline, 11 voices)
  espeak  – espeak-ng / espeak binary if installed
  blip    – built-in 'animalese' babble synth (always available: pipeline never dies)
"""
import hashlib, os, re, shutil, subprocess, tempfile, wave, logging
from pathlib import Path
import numpy as np
from scipy import signal
from .config import MODELS, CACHE, SR

log = logging.getLogger("gghuman.tts")
TTS_SR = 24000

# default voice per cast member (kokoro speaker ids) + fx
VOICES = {
    "narrator": dict(sid=1, speed=1.05, fx=None, pitch=1.0),
    "wren":   dict(sid=4,  speed=1.05, fx=None,    pitch=1.0),
    "red":    dict(sid=5,  speed=0.95, fx=None,    pitch=0.93),
    "gpt":    dict(sid=6,  speed=1.05, fx="robot", pitch=1.0),
    "claude": dict(sid=7,  speed=1.05, fx="robot", pitch=1.04),
    "gemini": dict(sid=10, speed=0.95, fx="robot", pitch=0.97),
}
_EXTRA_SIDS = [1, 2, 3, 8, 9, 0]   # used for custom cast members


def voice_for(who, spec_voices=None):
    if spec_voices and who in spec_voices:
        return {**VOICES.get(who, VOICES["wren"]), **spec_voices[who]}
    if who in VOICES:
        return VOICES[who]
    h = int(hashlib.md5(who.encode()).hexdigest(), 16)
    return dict(sid=_EXTRA_SIDS[h % len(_EXTRA_SIDS)], speed=1.0, fx="robot", pitch=1.0)


# ------------------------------------------------------------------ backends
class KokoroBackend:
    name = "kokoro"

    def __init__(self):
        import sherpa_onnx
        m = MODELS / "kokoro-en-v0_19"
        for f in ("model.onnx", "voices.bin", "tokens.txt", "espeak-ng-data"):
            if not (m / f).exists():
                raise FileNotFoundError(f"missing {m / f} (run: python setup_assets.py)")
        cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(),
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(m / "model.onnx"), voices=str(m / "voices.bin"),
                tokens=str(m / "tokens.txt"), data_dir=str(m / "espeak-ng-data")),
            num_threads=max(1, os.cpu_count() or 1), provider="cpu"))
        self.tts = sherpa_onnx.OfflineTts(cfg)
        self.sr = self.tts.sample_rate

    def synth(self, text, voice):
        a = self.tts.generate(text, sid=int(voice["sid"]), speed=float(voice.get("speed", 1.0)))
        x = np.asarray(a.samples, np.float32)
        if len(x) < 100:
            raise RuntimeError("kokoro returned empty audio")
        return x, a.sample_rate


class EspeakBackend:
    name = "espeak"

    def __init__(self):
        self.bin = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.bin:
            raise FileNotFoundError("espeak not installed")
        self.sr = 22050

    def synth(self, text, voice):
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "o.wav"
            sp = int(150 * float(voice.get("speed", 1.0)))
            subprocess.run([self.bin, "-w", str(wav), "-s", str(sp), "-p", str(int(50 * voice.get("pitch", 1.0))),
                            "-v", "en-us+m3" if voice["sid"] % 2 else "en-us+f3", text], check=True, capture_output=True, timeout=60)
            with wave.open(str(wav)) as w:
                x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
                return x, w.getframerate()


class BlipBackend:
    """Animalese-style babble: one pitched blip per syllable.  Not pretty speech
    but perfectly timed and deterministic – the 'never fail' last resort."""
    name = "blip"
    sr = TTS_SR

    def synth(self, text, voice):
        base = 150 + 22 * (voice["sid"] % 8)
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        words = re.findall(r"[A-Za-z']+|[,.!?…]", text)
        out = []
        for w in words:
            if w in ",.!?…":
                out.append(np.zeros(int(self.sr * 0.18)))
                continue
            for syl in re.findall(r"[^aeiouy]*[aeiouy]+(?:[^aeiouy]+$)?|[^aeiouy]+$", w.lower()) or [w]:
                n = int(self.sr * (0.085 / float(voice.get("speed", 1.0))))
                f = base * (1 + 0.35 * rng.uniform(-1, 1))
                t = np.arange(n) / self.sr
                env = np.sin(np.pi * t / t[-1]) ** 0.6
                x = (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t) + 0.2 * np.sin(6 * np.pi * f * t)) * env
                out.append(x * 0.5)
            out.append(np.zeros(int(self.sr * 0.03)))
        x = np.concatenate(out) if out else np.zeros(int(self.sr * 0.3))
        return x.astype(np.float32), self.sr


_BACKENDS = {"kokoro": KokoroBackend, "espeak": EspeakBackend, "blip": BlipBackend}
_active = None


def get_backend():
    global _active
    if _active is not None:
        return _active
    order = [os.environ["GGH_TTS"]] if os.environ.get("GGH_TTS") else ["kokoro", "espeak", "blip"]
    for name in order + ["blip"]:
        try:
            _active = _BACKENDS[name]()
            log.info("TTS backend: %s", _active.name)
            return _active
        except Exception as e:  # noqa
            log.warning("TTS backend %s unavailable: %s", name, e)
    raise RuntimeError("no TTS backend")


# ------------------------------------------------------------------ DSP helpers
def _resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    g = np.gcd(int(sr_in), int(sr_out))
    return signal.resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def _trim(x, sr, thr_db=-42, keep=0.012):
    if len(x) == 0:
        return x
    a = np.abs(x)
    th = max(a.max() * 10 ** (thr_db / 20), 1e-4)
    idx = np.where(a > th)[0]
    if len(idx) == 0:
        return x
    s = max(0, idx[0] - int(keep * sr))
    e = min(len(x), idx[-1] + int(keep * sr * 2))
    return x[s:e]


def _fx_robot(x, sr):
    b, a = signal.butter(2, [220, 7200], btype="band", fs=sr)
    y = signal.lfilter(b, a, x)
    ring = y * np.sin(2 * np.pi * 70 * np.arange(len(y)) / sr)
    return (0.9 * y + 0.12 * ring).astype(np.float32)


def _fx_whisper(x, sr):
    f, t, Z = signal.stft(x, sr, nperseg=512)
    rng = np.random.default_rng(1)
    Z2 = np.abs(Z) * np.exp(1j * rng.uniform(0, 2 * np.pi, Z.shape))
    _, y = signal.istft(Z2, sr, nperseg=512)
    b, a = signal.butter(2, 1200, btype="high", fs=sr)
    y = signal.lfilter(b, a, y)
    y = y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    return (y * 1.8).astype(np.float32)


def _rms_norm(x, target_db=-21.0):
    r = np.sqrt(np.mean(x ** 2) + 1e-9)
    g = 10 ** (target_db / 20) / r
    return np.clip(x * g, -1, 1).astype(np.float32)


def _envelope(x, sr, hop=0.01):
    n = int(sr * hop)
    m = len(x) // n
    rms = np.sqrt(np.mean(x[:m * n].reshape(m, n) ** 2, axis=1))
    rms = np.convolve(rms, np.ones(3) / 3, "same")
    return (rms / (rms.max() + 1e-9)).astype(np.float32), hop


def _syll(w):
    return max(1, len(re.findall(r"[aeiouy]+", w.lower())) - (1 if w.lower().endswith("e") and len(w) > 3 else 0))


def align_words(text, x, sr):
    """Word timings inside a trimmed clip: syllable-weighted proportional split,
    boundaries snapped to the nearest energy valley."""
    toks = text.split()
    if not toks:
        return []
    dur = len(x) / sr
    weights = []
    for t in toks:
        core = re.sub(r"[^A-Za-z0-9']", "", t) or "a"
        w = _syll(core) * 3.0 + len(core) * 0.35
        pause = 0.0
        if t.endswith(("…", "...")):
            pause = 4.5
        elif t.endswith((",", ";", ":", "—", "-")):
            pause = 2.2
        elif t.endswith((".", "!", "?")):
            pause = 1.0
        weights.append((w, pause))
    tot = sum(w + p for w, p in weights)
    env, hop = _envelope(x, sr, 0.005)
    out, cur = [], 0.0
    for i, (t, (w, p)) in enumerate(zip(toks, weights)):
        w0 = cur
        w1 = cur + dur * w / tot
        cur = w1 + dur * p / tot
        out.append([t, w0, w1])
    # snap boundaries (between word i end & i+1 start) to energy minima
    for i in range(len(out) - 1):
        mid = (out[i][2] + out[i + 1][1]) / 2
        span = min(0.09, 0.3 * (out[i][2] - out[i][1]), 0.3 * (out[i + 1][2] - out[i + 1][1]))
        a, b = int(max(0, (mid - span) / hop)), int(min(len(env) - 1, (mid + span) / hop))
        if b > a:
            m = (a + int(np.argmin(env[a:b + 1]))) * hop
            gap = out[i + 1][1] - out[i][2]
            out[i][2] = min(out[i][2], m) if gap > 0 else m
            out[i + 1][1] = max(out[i + 1][1], m) if gap > 0 else m
    out[0][1] = 0.0
    out[-1][2] = dur
    return [(w, float(a), float(b)) for w, a, b in out]


# ------------------------------------------------------------------ public API
def audible_onset(x, sr=SR, frac=0.03):
    """Seconds until the clip first exceeds `frac` of its peak = the moment a viewer actually HEARS the line."""
    th = frac * float(np.abs(x).max() + 1e-9)
    idx = np.where(np.abs(x) > th)[0]
    return float(idx[0] / sr) if len(idx) else 0.0


class Clip:
    def __init__(self, wav, dur, words, env, backend, text):
        self.wav, self.dur, self.words, self.env, self.backend, self.text = wav, dur, words, env, backend, text
        self.onset = audible_onset(wav)
        if words:   # first word starts exactly when speech becomes audible
            w0 = list(words[0]); w0[1] = self.onset
            self.words = [tuple(w0)] + list(words[1:])


def render_line(text, who, style=None, speed=None, spec_voices=None, cache=True):
    """-> Clip (float32 @ SR, trimmed, level-normalised, with word timings + envelope)."""
    v = dict(voice_for(who, spec_voices))
    if speed:
        v["speed"] *= speed
    be = get_backend()
    key = hashlib.sha1(f"{be.name}|{who}|{text}|{style}|{sorted(v.items(), key=str)}".encode()).hexdigest()[:20]
    cp = CACHE / "tts" / f"{key}.npz"
    if cache and cp.exists():
        z = np.load(cp, allow_pickle=True)
        return Clip(z["x"], float(z["dur"]), [tuple(w) for w in z["words"].tolist()], z["env"], be.name, text)
    try:
        x, sr = be.synth(text, v)
    except Exception as e:  # runtime failure -> drop to blip for THIS line
        log.warning("TTS %s failed on %r (%s); using blip fallback", be.name, text, e)
        be = BlipBackend()
        x, sr = be.synth(text, v)
    x = _resample(x, sr, SR)
    if v.get("pitch", 1.0) != 1.0:  # cheap pitch/formant shift by resampling (changes speed slightly)
        x = _resample(x, SR, int(SR / v["pitch"]))
        x = x.astype(np.float32)
        # compensate duration roughly
    x = _trim(x, SR)
    if v.get("fx") == "robot":
        x = _fx_robot(x, SR)
    if style == "whisper":
        x = _fx_whisper(x, SR) * 0.8
    x = _rms_norm(x, -24.0 if style == "whisper" else -21.0)
    x = (x * min(1.0, 0.92 / (np.abs(x).max() + 1e-9))).astype(np.float32)
    # short fades to avoid clicks
    f = int(0.004 * SR)
    x[:f] *= np.linspace(0, 1, f)
    x[-f:] *= np.linspace(1, 0, f)
    words = align_words(text, x, SR)
    env, hop = _envelope(x, SR, 0.01)
    clip = Clip(x, len(x) / SR, words, env, be.name, text)
    if cache:
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cp, x=x, dur=clip.dur, words=np.array(words, dtype=object), env=env)
    return clip
