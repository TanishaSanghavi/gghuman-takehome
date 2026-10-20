"""Mix voice + SFX + music from the timeline: sample-accurate placement, music
ducking under speech, enforced silence windows, LUFS normalisation + limiter."""
import wave
import numpy as np
import pyloudnorm as pyln
from scipy import signal
from .config import SR, FPS
from . import sfx as S
from .music import render_music

TARGET_LUFS = -14.0
DEFAULT_CEILING = 0.62   # ≈ -4 dBFS: leaves AAC overshoot headroom
CEILING = DEFAULT_CEILING


def _place(buf, x, t, gain=1.0, pan=0.0):
    a = int(round(t * SR))
    s = max(0, -a)
    e = min(len(x), buf.shape[1] - a)
    if e <= s:
        return
    l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
    buf[0, a + s:a + e] += x[s:e] * gain * l
    buf[1, a + s:a + e] += x[s:e] * gain * r


def mix(tl_obj, out_wav, stems_dir=None):
    tl = tl_obj.data
    total = tl["meta"]["duration"]
    n = int(round(total * SR))
    pad = int(0.5 * SR)
    voice = np.zeros((2, n + pad), np.float32)
    fx = np.zeros_like(voice)
    # voices ---------------------------------------------------------------
    for s in tl["shots"]:
        for ln in s["lines"]:
            clip = tl_obj.clips[ln["id"]]
            pan = {"wren": -0.1, "red": -0.25, "gpt": 0.1, "claude": 0.25, "gemini": 0.0}.get(ln["who"], 0.0) * 0.5
            _place(voice, clip.wav, ln["t0"] - ln.get("onset", clip.onset), 1.0, pan)
    # sfx ------------------------------------------------------------------
    for e in tl["sfx"]:
        x, pk = S.get(e["name"])
        _place(fx, x, e["t"] - pk if e.get("peak", True) else e["t"], e.get("gain", 1.0) * 0.8, e.get("pan", 0))
    # room tone for the secret camera + blackout shots
    for s in tl["shots"]:
        if s["type"] in ("secret", "secret_split"):
            h = S.hum(s["t1"] - s["t0"], s["i"])
            _place(fx, h, s["t0"], 1.0, 0)
    fx = np.tanh(fx * 1.6) / 1.6          # tame transient peaks (boom/stamp) so overall loudness can rise
    # silence windows apply to SFX too (a few explicit sounds like riser are in music)
    for a, b in tl["mute"]:
        ia, ib = int(a * SR), int(b * SR)
        fx[:, ia:ib] = 0
    # music with sidechain ducking -----------------------------------------
    mcfg = tl["meta"]["music"]
    if mcfg.get("level", 1.0) <= 0:
        m = np.zeros(n + pad, np.float32)          # sparse GGhuman style: voice + silence + sfx, no music bed
    else:
        m = render_music(tl, bpm=mcfg.get("bpm", 96), root=mcfg.get("root", "A"), seed=mcfg.get("seed", 1)) * float(mcfg.get("level", 1.0))
    m = np.pad(m, (0, max(0, n + pad - len(m))))[:n + pad]
    env = np.abs(voice).max(axis=0)
    env = signal.sosfilt(signal.butter(1, 6, "low", fs=SR, output="sos"), env)  # slow follower
    env = np.clip(env / 0.18, 0, 1)
    # fast attack, slow release
    duck = 1 - 0.55 * env
    music = np.stack([m, np.concatenate([np.zeros(int(0.011 * SR), np.float32), m[:-int(0.011 * SR)]])]) * duck * 0.55
    if stems_dir:
        stems_dir.mkdir(parents=True, exist_ok=True)
        for nm, st in (("voice", voice), ("sfx", fx), ("music", music)):
            _write(stems_dir / f"{nm}.wav", st[:, :n])
    mix_ = voice * 1.0 + fx * 0.9 + music
    mix_ = mix_[:, :n]
    # end tail fade (reveal hit rings out, then fades before the loop point)
    f = int(0.25 * SR)
    mix_[:, -f:] *= np.linspace(1, 0.15, f)
    # loudness normalise (BS.1770) through a look-ahead brickwall limiter (2 passes to land on target)
    meter = pyln.Meter(SR)
    lufs0 = meter.integrated_loudness(mix_.T)
    g = 10 ** ((TARGET_LUFS - lufs0) / 20) if np.isfinite(lufs0) else 1.0
    out = mix_
    for _ in range(3):
        out = limit(mix_ * g, CEILING)
        l1 = meter.integrated_loudness(out.T)
        if abs(l1 - TARGET_LUFS) < 0.35:
            break
        g *= 10 ** ((TARGET_LUFS - l1) / 20)
    mix_ = out
    lufs = lufs0
    _write(out_wav, mix_)
    return dict(lufs_before=float(lufs0), lufs_after=float(meter.integrated_loudness(mix_.T)), peak=float(np.abs(mix_).max()))


def limit(x, ceiling=0.7, look=0.006, release=0.09):
    """Look-ahead brickwall limiter on a (2,n) array."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    peak = np.abs(x).max(axis=0) + 1e-9
    gain = np.minimum(1.0, ceiling / peak)
    w = int(look * SR) * 2 + 1
    gain = minimum_filter1d(gain, w, mode="nearest")
    gain = uniform_filter1d(gain, w, mode="nearest")          # smooth attack so min is still honoured
    rel = np.exp(-1.0 / (release * SR))
    # release smoothing (only lets gain recover slowly)
    out = np.empty_like(gain)
    g = 1.0
    # vectorised via blocks of 64 samples for speed
    B = 64
    gb = gain[: len(gain) // B * B].reshape(-1, B).min(axis=1)
    sm = np.empty_like(gb)
    for i, v in enumerate(gb):
        g = v if v < g else g * rel ** B + v * (1 - rel ** B)
        sm[i] = g
    out = np.repeat(sm, B)
    out = np.pad(out, (0, len(gain) - len(out)), mode="edge")
    out = np.minimum(out, gain + 1e-9) if False else np.minimum(out, 1.0)
    y = x * out
    over = np.abs(y).max(axis=0) > ceiling * 1.001
    if over.any():   # safety net: hard scale any residual overs
        y[:, over] *= (ceiling / np.abs(y[:, over]).max(axis=0))
    return y


def _write(path, x):
    pcm = (np.clip(x.T, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
