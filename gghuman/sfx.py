"""Procedural sound effects.  Every generator returns (samples, peak_time):
`peak_time` is the moment inside the sample that must land on the visual event,
so a 'whoosh' that builds up is placed *before* the cut and peaks ON it."""
import numpy as np
from scipy import signal
from .config import SR


def _t(d):
    return np.arange(int(d * SR)) / SR


def _noise(n, seed=0):
    return np.random.default_rng(seed).normal(0, 1, n).astype(np.float32)


def _bp(x, lo, hi, order=2):
    return signal.sosfilt(signal.butter(order, [lo, hi], "band", fs=SR, output="sos"), x).astype(np.float32)


def _lp(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, fc, "low", fs=SR, output="sos"), x).astype(np.float32)


def _hp(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, fc, "high", fs=SR, output="sos"), x).astype(np.float32)


def _sweep(f0, f1, d, expo=True):
    t = _t(d)
    f = f0 * (f1 / f0) ** (t / d) if expo else f0 + (f1 - f0) * t / d
    return np.sin(2 * np.pi * np.cumsum(f) / SR).astype(np.float32)


def _fade(x, a=0.003, b=0.01):
    x = x.copy()
    na, nb = int(a * SR), int(b * SR)
    if na: x[:na] *= np.linspace(0, 1, na)
    if nb: x[-nb:] *= np.linspace(1, 0, nb)
    return x


def _pad(x, d):
    n = int(d * SR)
    return np.pad(x, (0, max(0, n - len(x))))[:n] if n else x


def knock():
    t = _t(0.3)
    th = _sweep(95, 50, 0.3) * np.exp(-t * 18)
    cl = _hp(_noise(len(t), 1), 600) * np.exp(-t * 140) * 0.8
    return _fade(0.9 * th + cl), 0.0


def scribble():
    t = _t(0.42)
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 26 * t)
    x = _bp(_noise(len(t), 2), 1800, 6500) * am * np.hanning(len(t)) * 0.5
    return x.astype(np.float32), 0.2


def switch():
    out = np.zeros(int(0.25 * SR), np.float32)
    for off, g in ((0.0, 1.0), (0.075, 0.7)):
        n = int(0.03 * SR)
        c = _bp(_noise(n, int(off * 100) + 3), 900, 4200) * np.exp(-np.arange(n) / SR * 150) * g
        o = int(off * SR)
        out[o:o + n] += c
    t = _t(0.25)
    out += _sweep(180, 90, 0.25) * np.exp(-t * 30) * 0.5
    return out, 0.0


def power_down():
    d = 1.0
    t = _t(d)
    x = _sweep(240, 28, d) * np.exp(-t * 3.0) * 0.55
    x += _lp(_noise(len(t), 4), 400) * np.exp(-t * 4) * 0.25
    return _fade(x, 0.002, 0.08), 0.0


def power_up():
    d = 0.5
    t = _t(d)
    x = _sweep(40, 320, d) * (t / d) ** 2 * 0.5
    x[-int(0.02 * SR):] = 0
    n = int(0.03 * SR)
    c = np.zeros_like(x)
    c[-n:] = _bp(_noise(n, 5), 900, 4200) * np.exp(-np.arange(n) / SR * 120)
    return _fade(x + c, 0.002, 0.0), d - 0.03


def pop():
    d = 0.11
    t = _t(d)
    return _fade(_sweep(520, 1000, d) * np.exp(-t * 28) * 0.7), 0.0


def pop_big():
    d = 0.3
    t = _t(d)
    x = _sweep(180, 420, d) * np.exp(-t * 12) * 0.8 + _hp(_noise(len(t), 6), 1500) * np.exp(-t * 60) * 0.35
    return _fade(x), 0.0


def rec_beep():
    d = 0.14
    t = _t(d)
    x = np.sign(np.sin(2 * np.pi * 880 * t)) * 0.25 + np.sin(2 * np.pi * 880 * t) * 0.4
    return _fade(x * np.hanning(len(t)) ** 0.3, 0.003, 0.02), 0.0


def slide():
    d = 0.45
    t = _t(d)
    x = _bp(_noise(len(t), 7), 500, 3000) * np.sin(np.pi * t / d) ** 2 * 0.5
    return x, d * 0.9


def stamp():
    d = 0.4
    t = _t(d)
    x = _sweep(120, 45, d) * np.exp(-t * 14) * 1.0
    x += _bp(_noise(len(t), 8), 300, 2500) * np.exp(-t * 90) * 0.9
    return _fade(x), 0.0


def whoosh():
    d = 0.32
    t = _t(d)
    n = _noise(len(t), 9)
    f = np.linspace(0, 1, len(t))
    x = _bp(n, 400, 6000) * (f ** 1.5) * np.exp(-((t - 0.24) ** 2) / 0.006) * 0.9
    x2 = _hp(n, 2500) * np.sin(np.pi * np.clip(t / d, 0, 1)) ** 3 * 0.25
    return _fade(x + x2), 0.24


def flash():
    out = np.zeros(int(0.3 * SR), np.float32)
    for off in (0.0, 0.05):
        n = int(0.04 * SR)
        out[int(off * SR):int(off * SR) + n] += _hp(_noise(n, 10), 1500) * np.exp(-np.arange(n) / SR * 110) * 0.8
    out += _sweep(1600, 700, 0.3) * np.exp(-_t(0.3) * 40) * 0.3
    return out, 0.0


def iris():
    d = 0.3
    t = _t(d)
    return _fade(_sweep(1400, 260, d) * np.exp(-t * 7) * 0.5), 0.0


def boom():
    d = 1.6
    t = _t(d)
    x = _sweep(75, 32, d) * np.exp(-t * 2.6) * 1.1
    x += _lp(_noise(len(t), 11), 900) * np.exp(-t * 7) * 0.8
    x += _hp(_noise(len(t), 12), 3000) * np.exp(-t * 40) * 0.4
    return _fade(x, 0.001, 0.2), 0.0


def sting():
    d = 1.3
    t = _t(d)
    x = np.zeros_like(t)
    for f, g in ((220, 1), (233.1, 0.9), (329.6, 0.6), (349.2, 0.55), (110, 0.8)):
        saw = 2 * ((f * t) % 1) - 1
        x += saw * g
    x = _lp(x, 3500) * np.exp(-t * 3.2) * 0.22
    return _fade(x, 0.001, 0.15), 0.0


def gasp():
    d = 0.4
    t = _t(d)
    x = _bp(_noise(len(t), 13), 700, 4200) * (t / d) ** 1.5 * np.exp(-((t - d) ** 2) / 0.02) * 0.7
    return _fade(x, 0.01, 0.02), d - 0.04


def ding():
    d = 1.4
    t = _t(d)
    x = sum(np.sin(2 * np.pi * f * t) * g * np.exp(-t * dc) for f, g, dc in ((880, 1, 3.5), (1320, .5, 5), (1760, .3, 7), (2640, .15, 9)))
    return _fade(x * 0.25, 0.001, 0.2), 0.0


def heartbeat():
    out = np.zeros(int(0.5 * SR), np.float32)
    for off, g in ((0.0, 1.0), (0.2, 0.75)):
        n = int(0.22 * SR)
        t = np.arange(n) / SR
        b = _sweep(75, 42, 0.22) * np.exp(-t * 22) * g
        o = int(off * SR)
        out[o:o + n] += b
    return out * 0.9, 0.0


def tick():
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    x = _bp(_noise(n, 14), 1800, 5500) * np.exp(-t * 110) + np.sin(2 * np.pi * 1300 * t) * np.exp(-t * 90) * 0.4
    return _fade(x * 0.7, 0.0005, 0.005), 0.0


def tick_pop():
    a, _ = tick()
    b, _ = pop()
    return _pad(a, 0.12) + _pad(b, 0.12) * 0.6, 0.0


def record_scratch():
    d = 0.5
    t = _t(d)
    f = 1800 * np.exp(-t * 7)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.3 + _bp(_noise(len(t), 15), 800, 5000) * 0.5
    x *= np.exp(-t * 4) * (1 + 0.5 * np.sin(2 * np.pi * 14 * t))
    return _fade(x, 0.002, 0.05), 0.0


def drip():
    d = 0.2
    t = _t(d)
    return _fade(_sweep(2200, 900, d) * np.exp(-t * 30) * 0.4), 0.0


def gulp():
    d = 0.3
    t = _t(d)
    return _fade(_sweep(300, 120, d) * np.exp(-t * 14) * 0.8 + _bp(_noise(len(t), 16), 200, 900) * np.exp(-t * 20) * 0.3), 0.0


def scream():
    d = 1.7
    t = _t(d)
    f = 700 + 150 * (1 - np.exp(-t * 14)) + 18 * np.sin(2 * np.pi * 6.2 * t) + 40 * np.exp(-t * 3)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = sum(np.sin(h * ph) / h ** 0.7 for h in range(1, 12))
    x = _bp(x.astype(np.float32), 500, 3800) * 1.3 + _hp(_noise(len(t), 41), 2500) * 0.10
    env = np.minimum(1, t / 0.02) * np.minimum(1, (d - t) / 0.5)
    return _fade(x * env * 0.5, 0.002, 0.05), 0.0


def click():
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    a = _bp(_noise(n, 42), 1800, 6000) * np.exp(-t * 260)
    b = np.sin(2 * np.pi * 170 * t) * np.exp(-t * 90) * 0.7
    out = np.zeros(n, np.float32)
    out += a * 0.8 + b
    k = int(0.07 * SR)
    out[k:k + int(0.03 * SR)] += _bp(_noise(int(0.03 * SR), 43), 2000, 6500) * np.exp(-np.arange(int(0.03 * SR)) / SR * 300) * 0.5
    return _fade(out, 0.0005, 0.01), 0.0


def flip():
    d = 0.18
    t = _t(d)
    x = _bp(_noise(len(t), 44), 1200, 7000) * np.sin(np.pi * np.clip(t / d, 0, 1)) ** 2 * 0.7
    k = int(0.14 * SR)
    x[k:k + int(0.02 * SR)] += np.sin(2 * np.pi * 900 * t[:int(0.02 * SR)]) * np.exp(-np.arange(int(0.02 * SR)) / SR * 200) * 0.5
    return _fade(x, 0.003, 0.02), 0.14


def chips():
    out = np.zeros(int(0.5 * SR), np.float32)
    rng = np.random.default_rng(45)
    for off in (0.0, 0.05, 0.11, 0.2, 0.27):
        n = int(0.06 * SR)
        t = np.arange(n) / SR
        f = rng.uniform(2800, 5200)
        c = (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 1.5 * t)) * np.exp(-t * 70) * 0.5
        o = int(off * SR)
        out[o:o + n] += c
    return out, 0.0


LIB = {k: v for k, v in dict(scream=scream, click=click, flip=flip, chips=chips, knock=knock, scribble=scribble, switch=switch, power_down=power_down, power_up=power_up, pop=pop,
                              pop_big=pop_big, rec_beep=rec_beep, slide=slide, stamp=stamp, whoosh=whoosh, flash=flash, iris=iris,
                              boom=boom, sting=sting, gasp=gasp, ding=ding, heartbeat=heartbeat, tick=tick, tick_pop=tick_pop,
                              record_scratch=record_scratch, drip=drip, gulp=gulp).items()}
_cache = {}


def get(name):
    if name not in LIB:
        raise KeyError(f"unknown sfx {name!r}; available: {sorted(LIB)}")
    if name not in _cache:
        x, pk = LIB[name]()
        m = float(np.abs(x).max())
        if m > 0.95:
            x = x * (0.95 / m)
        _cache[name] = (x.astype(np.float32), pk)
    return _cache[name]


def hum(duration, seed=0):
    """Low electrical hum + faint tape hiss: the 'secret camera' room tone."""
    t = _t(duration)
    x = np.sin(2 * np.pi * 50 * t) * 0.05 + np.sin(2 * np.pi * 100 * t) * 0.02
    x += _hp(_noise(len(t), seed + 20), 4000) * 0.012
    return x.astype(np.float32)
