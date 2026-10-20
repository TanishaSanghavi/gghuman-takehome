"""Procedural score: tense lo-fi pluck/bass/pad whose energy follows the shots.
The bar grid restarts at every silence window so music re-enters ON the
reveal downbeat."""
import numpy as np
from scipy import signal
from .config import SR
from . import sfx as S

NOTE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
MIN_PROG = [(0, [0, 3, 7]), (-4, [0, 4, 7]), (3, [0, 4, 7]), (-5, [0, 4, 7, 10])]  # i - VI - III - V7


def _mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def _energy_curve(tl, n, smooth=0.35):
    e = np.zeros(n, np.float32)
    for s in tl["shots"]:
        a, b = int(s["t0"] * SR), min(n, int(s["t1"] * SR))
        e[a:b] = s["energy"]
    k = int(smooth * SR)
    kernel = np.ones(k) / k
    return np.convolve(np.pad(e, (k // 2, k // 2), mode="edge"), kernel, "valid")[:n].astype(np.float32)


def _add(buf, x, t0, g=1.0):
    a = int(round(t0 * SR))
    if a >= len(buf) or a + len(x) <= 0:
        return
    s = max(0, -a)
    e = min(len(x), len(buf) - a)
    buf[a + s:a + e] += x[s:e] * g


def _pluck(f, d=0.5):
    t = np.arange(int(d * SR)) / SR
    x = np.zeros_like(t)
    for h, g, dc in ((1, 1, 6), (2, .5, 9), (3, .3, 12), (4, .15, 16), (5, .08, 20)):
        x += g * np.sin(2 * np.pi * f * h * t) * np.exp(-t * dc)
    return (x * np.minimum(1, t / 0.004)).astype(np.float32)


def _bass(f, d=0.4):
    t = np.arange(int(d * SR)) / SR
    x = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * (2 * ((f * t) % 1) - 1)
    return (x * np.exp(-t * 4.5) * np.minimum(1, t / 0.006)).astype(np.float32)


def _kick():
    t = np.arange(int(0.25 * SR)) / SR
    return (S._sweep(110, 42, 0.25) * np.exp(-t * 16)).astype(np.float32)


def _hat(seed):
    n = int(0.05 * SR)
    return (S._hp(S._noise(n, seed), 6000) * np.exp(-np.arange(n) / SR * 90)).astype(np.float32)


def _wood():
    n = int(0.08 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 820 * t) * np.exp(-t * 55) * 0.8 + S._bp(S._noise(n, 3), 1500, 4000) * np.exp(-t * 120) * 0.3).astype(np.float32)


def _pad(chord_midis, d, g=1.0):
    t = np.arange(int(d * SR)) / SR
    x = np.zeros_like(t)
    for m in chord_midis:
        for det in (-0.07, 0.07):
            x += np.sin(2 * np.pi * _mtof(m) * (1 + det * 0.01) * t)
    env = np.minimum(1, t / 0.25) * np.minimum(1, (d - t) / 0.3)
    return (x * env * 0.12 * g).astype(np.float32)


def render_music(tl, bpm=96, root="A", seed=0):
    n = int(tl["meta"]["duration"] * SR) + int(0.6 * SR)
    buf = np.zeros(n, np.float32)
    energy = _energy_curve(tl, n)
    beat = 60.0 / bpm
    bar = beat * 4
    r = NOTE.get(root, 9)
    mutes = tl["mute"]
    total = tl["meta"]["duration"]
    # segments between silence windows
    cuts = [0.0]
    ends = []
    for a, b in mutes:
        ends.append(a)
        cuts.append(b)
    ends.append(total)
    rng = np.random.default_rng(seed)
    for s0, s1 in zip(cuts, ends):
        k = 0
        t0 = s0
        while t0 < s1 - 1e-3:
            root_off, chord = MIN_PROG[k % 4]
            base = 45 + r - 9 + root_off  # bass register
            e_at = lambda tt: float(energy[min(n - 1, int(tt * SR))])
            e0 = e_at(t0)
            # pad
            _add(buf, _pad([base + 24 + c for c in chord], bar + 0.2), t0, e0 * 0.9)
            # bass: beat 1 and 3, 8ths when busy
            for bb in range(4):
                tt = t0 + bb * beat
                e = e_at(tt)
                if e > 0.15 and bb in (0, 2):
                    _add(buf, _bass(_mtof(base)), tt, 0.5 + 0.5 * e)
                if e > 0.55 and bb in (1, 3):
                    _add(buf, _bass(_mtof(base + chord[-1] % 12 if False else base + 7), 0.25), tt + beat * 0.5, 0.45 * e)
            # arpeggio plucks (8ths)
            for st in range(8):
                tt = t0 + st * beat / 2
                e = e_at(tt)
                if e > 0.33:
                    seq = [0, 1, 2, 1, 2, 3 % len(chord), 2, 1]
                    m = base + 24 + chord[seq[st] % len(chord)] + (12 if st % 4 == 3 else 0)
                    _add(buf, _pluck(_mtof(m)), tt, 0.28 * min(1, e * 1.3))
            # percussion
            for bb in range(4):
                tt = t0 + bb * beat
                e = e_at(tt)
                if e > 0.5 and bb in (0, 2):
                    _add(buf, _kick(), tt, 0.7 * e)
                if e <= 0.45 and e > 0.12:
                    _add(buf, _wood(), tt, 0.28)   # suspense 'clock'
                if e > 0.6:
                    _add(buf, _hat(bb), tt + beat * 0.5, 0.25 * e)
            t0 += bar
            k += 1
    # secret-cam drone under low-energy shots
    for s in tl["shots"]:
        if s["type"] in ("secret", "secret_split", "blackout"):
            d = s["t1"] - s["t0"]
            t = np.arange(int(d * SR)) / SR
            drone = (np.sin(2 * np.pi * 55 * t) + np.sin(2 * np.pi * 55.6 * t)) * 0.12 * np.minimum(1, t / 0.4) * np.minimum(1, (d - t) / 0.2)
            _add(buf, drone.astype(np.float32), s["t0"], 1.0)
    # risers
    for a, b in tl["risers"]:
        d = b - a
        if d <= 0.2:
            continue
        t = np.arange(int(d * SR)) / SR
        ramp = (t / d) ** 2
        x = S._sweep(160, 1900, d) * ramp * 0.30
        x += S._hp(S._noise(len(t), 33), 2200) * ramp * 0.22
        x += S._sweep(80, 700, d) * ramp * 0.18
        _add(buf, x.astype(np.float32), a, 1.0)
    # hard gates for the silence windows (50 ms fade-out into silence)
    for a, b in mutes:
        ia, ib = int(a * SR), min(n, int(b * SR))
        f = int(0.05 * SR)
        buf[max(0, ia - f):ia] *= np.linspace(1, 0, ia - max(0, ia - f))
        buf[ia:ib] = 0
    buf = signal.sosfilt(signal.butter(2, 9000, "low", fs=SR, output="sos"), buf).astype(np.float32)
    return buf
