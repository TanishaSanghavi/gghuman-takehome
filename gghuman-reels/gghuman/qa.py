"""Automated quality gate.  Everything is verified on the FINAL mp4 (decoded
audio + decoded frames) against the Timeline, not on intermediate files."""
import re, wave, logging
import numpy as np, cv2
import pyloudnorm as pyln
from scipy import signal
from .config import W, H, FPS, SR, CAPTION_Y
from . import encode

log = logging.getLogger("gghuman.qa")


def _chk(name, ok, value, expect, level="fail", note=""):
    return dict(name=name, status="pass" if ok else level, value=value, expect=expect, note=note)


def _db(x):
    return 20 * np.log10(max(float(x), 1e-9))


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12)) if len(x) else 0.0


def _band_env(x, sr=SR, hop=0.005):
    sos = signal.butter(2, [300, 3400], "band", fs=sr, output="sos")
    y = signal.sosfilt(sos, x)
    n = int(sr * hop)
    m = len(y) // n
    e = np.sqrt(np.mean(y[:m * n].reshape(m, n) ** 2, axis=1))
    return e, hop


def read_wav_mono(path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        return x.reshape(-1, w.getnchannels()).mean(axis=1)


# ---------------------------------------------------------------- checks
def check_container(path, tl):
    p = encode.probe(path)
    v = next(s for s in p["streams"] if s["codec_type"] == "video")
    a = next((s for s in p["streams"] if s["codec_type"] == "audio"), None)
    dur = float(p["format"]["duration"])
    fps = eval(v["r_frame_rate"])
    out = [
        _chk("video_format", (v["codec_name"], v["width"], v["height"], v["pix_fmt"]) == ("h264", W, H, "yuv420p"),
             f'{v["codec_name"]} {v["width"]}x{v["height"]} {v["pix_fmt"]}', f"h264 {W}x{H} yuv420p"),
        _chk("fps", abs(fps - FPS) < 0.01, fps, FPS),
        _chk("has_audio", a is not None and a["codec_name"] == "aac" and int(a["sample_rate"]) == 48000, a and f'{a["codec_name"]} {a["sample_rate"]}Hz', "aac 48000Hz"),
        _chk("duration_matches_timeline", abs(dur - tl["meta"]["duration"]) < 0.12, round(dur, 3), f'{tl["meta"]["duration"]:.2f}±0.12'),
        _chk("duration_reel_length", 12 <= dur <= 59, round(dur, 2), "12–59 s", level="warn"),
        _chk("file_size_mb", float(p["format"]["size"]) / 1e6 < 60, round(float(p["format"]["size"]) / 1e6, 1), "<60 MB"),
    ]
    return out


def check_audio(path, tl, voice_stem=None):
    x = encode.decode_audio(path, SR)
    out = []
    meter = pyln.Meter(SR)
    lufs = float(meter.integrated_loudness(x[:, None]))
    peak = float(np.abs(x).max())
    out.append(_chk("loudness_lufs", -16.5 <= lufs <= -11.5, round(lufs, 2), "-16.5…-11.5 (IG ≈ -14)"))
    out.append(_chk("peak", peak <= 0.98, round(peak, 3), "≤0.98"))
    out.append(_chk("clipping", float(np.mean(np.abs(x) > 0.995)) < 1e-4, float(np.mean(np.abs(x) > 0.995)), "<1e-4"))
    # silence before every reveal + a hit right after it
    for r in tl["reveals"]:
        t = r["t"]
        pre = x[int((t - 0.6) * SR):int((t - 0.08) * SR)]
        post = x[int(t * SR):int((t + 0.25) * SR)]
        out.append(_chk(f"silence_before_reveal@{t:.1f}s", _db(_rms(pre)) < -48, round(_db(_rms(pre)), 1), "< -48 dBFS (true pre-reveal silence)"))
        out.append(_chk(f"reveal_hit@{t:.1f}s", _db(_rms(post)) > _db(_rms(x)) + 3 or _db(_rms(post)) > -22, round(_db(_rms(post)), 1), "loud downbeat on reveal frame"))
    # dead-air: longest stretch with neither voice nor sfx
    ev = sorted([(l[0], l[1]) for l in tl["speech"]] + [(e["t"], e["t"] + 0.25) for e in tl["sfx"] if e["name"] not in ("scribble",)])
    gaps, cur = [], 0.0
    for a, b in ev:
        if a > cur:
            gaps.append(a - cur)
        cur = max(cur, b)
    gaps.append(max(0.0, tl["meta"]["duration"] - cur))
    mg = max(gaps)
    out.append(_chk("max_dead_air_s", mg <= 1.7, round(mg, 2), "≤1.7 s without voice/sfx", level="warn"))
    # voice-stem vs timeline onset, and final-mix vs stem lag
    if voice_stem is not None and voice_stem.exists():
        vs = read_wav_mono(voice_stem)
        errs = []
        prev_end = 0.0
        for s in sorted(tl["speech"]):
            t0 = s[0]
            a = max(prev_end, t0 - 0.12)
            seg = vs[int(a * SR):int((t0 + 0.4) * SR)]
            th = max(0.03 * np.abs(vs[int(t0 * SR):int(s[1] * SR)]).max(), 1e-4)
            idx = np.where(np.abs(seg) > th)[0]
            if len(idx):
                errs.append((idx[0] / SR + a) - t0)
            prev_end = s[1]
        errs = np.array(errs) if errs else np.zeros(1)
        out.append(_chk("voice_onset_vs_timeline_ms", float(np.abs(errs).max()) < 0.021, round(float(np.abs(errs).max()) * 1000, 1), "≤20 ms (audible speech starts on its caption/mouth frame)"))
        e1, hop = _band_env(vs)
        e2, _ = _band_env(x[:len(vs)])
        n = min(len(e1), len(e2))
        a1, a2 = e1[:n] - e1[:n].mean(), e2[:n] - e2[:n].mean()
        L = int(0.12 / hop)
        cc = np.correlate(a2, a1, "full")[n - 1 - L:n + L]
        lag = (np.argmax(cc) - L) * hop
        out.append(_chk("final_mix_lag_vs_voice_ms", abs(lag) <= 0.02, round(lag * 1000, 1), "≤20 ms", note="encoder/mux delay"))
    return out, lufs, x


def check_video(path, tl, spec_state=None):
    cap = cv2.VideoCapture(str(path))
    n_exp = int(round(tl["meta"]["duration"] * FPS))
    gl, luma, capreg, capregb, capregc, white = [], [], [], [], [], []
    prev = None
    first = None
    y0, y1 = CAPTION_Y - 75, CAPTION_Y + 75
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if first is None:
            first = fr.copy()
        g = cv2.cvtColor(cv2.resize(fr, (W // 4, H // 4), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        luma.append(float(g.mean()))
        cr = fr[y0:y1, 40:680]
        white.append(int(np.sum(np.all(cr >= 248, axis=2))))
        crg = cv2.cvtColor(cr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        capreg.append(crg)
        capregb.append(cv2.cvtColor(fr[475:625, 40:680], cv2.COLOR_BGR2GRAY).astype(np.float32))
        capregc.append(cv2.cvtColor(fr[1012:1074, 40:680], cv2.COLOR_BGR2GRAY).astype(np.float32))
        gl.append(g)
    cap.release()
    n = len(gl)
    out = [_chk("frame_count", abs(n - n_exp) <= 1, n, n_exp)]
    d = np.array([0.0] + [float(np.abs(gl[i] - gl[i - 1]).mean()) for i in range(1, n)])
    cd = np.array([0.0] + [float(np.abs(capreg[i] - capreg[i - 1]).mean()) for i in range(1, n)])
    cdb = np.array([0.0] + [float(np.abs(capregb[i] - capregb[i - 1]).mean()) for i in range(1, n)])
    cdc = np.array([0.0] + [float(np.abs(capregc[i] - capregc[i - 1]).mean()) for i in range(1, n)])
    types = {s["i"]: s["type"] for s in tl["shots"]}
    # cuts land on their frame
    hits, tot = 0, 0
    for s in tl["shots"][1:]:
        if s["trans"] not in ("cut", "flash"):
            continue
        f = int(round(s["t0"] * FPS))
        if f >= n:
            continue
        tot += 1
        win = d[max(1, f - 1):f + 2]
        if len(win) and d[f] >= win.max() - 1e-6 and d[f] > 3.0:
            hits += 1
    out.append(_chk("cuts_on_timeline_frame", tot == 0 or hits / tot >= 0.85, f"{hits}/{tot}", "≥85% of hard cuts land exactly on their scheduled frame"))
    # captions pop on their frame
    hit, tot = 0, 0
    for c in tl["captions"]:
        f = int(round(c["t0"] * FPS))
        if f < 2 or f + 2 >= n:
            continue
        tot += 1
        sig = cdb if types.get(c["shot"]) == "secret_split" else (cdc if types.get(c["shot"]) == "private" else cd)
        base = np.percentile(sig[max(1, f - 7):f - 1], 25) if f > 3 else 0     # quiet baseline (a pop 0.2 s earlier must not raise the bar)
        spike = sig[f - 1:f + 2].max()
        sh = tl["shots"][c["shot"]]
        in_trans = sh["trans"] != "cut" and (c["t0"] - sh["t0"]) < 0.35     # picture is changing anyway -> check presence instead
        present = sig is cd and white[min(n - 1, f + 3)] > 250
        if spike > max(5.0, base * 2.2) or (sig is cd and white[f] > 250 and white[f - 3] < white[f] * 0.5) or (in_trans and present):
            hit += 1
    out.append(_chk("captions_appear_on_spoken_frame", tot == 0 or hit / tot >= 0.9, f"{hit}/{tot}", "≥90% of caption chunks pop on their word's frame"))
    # frozen / black / hook
    run, mx = 0, 0
    for v in d[1:]:
        run = run + 1 if v < 0.12 else 0
        mx = max(mx, run)
    out.append(_chk("max_frozen_run_s", mx / FPS <= 0.5, round(mx / FPS, 2), "≤0.5 s of identical frames"))
    lm = np.array(luma)
    run, mxb = 0, 0
    for v in lm:
        run = run + 1 if v < 6 else 0
        mxb = max(mxb, run)
    out.append(_chk("black_frames_s", mxb / FPS <= 0.2, round(mxb / FPS, 2), "≤0.2 s near-black"))
    f0 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    dark_hook = int(np.sum(f0[60:300, 60:660] < 100))
    first_secret = tl["shots"][0]["type"] in ("secret", "secret_split")
    out.append(_chk("first_frame_hook_visible", first_secret or dark_hook > 700, dark_hook, ">700 hook-text pixels in frame 0 (thumbnail shows the POV)"))
    out.append(_chk("first_frame_not_flat", float(f0.std()) > 25, round(float(f0.std()), 1), ">25"))
    return out, first


def check_cta(path, tl):
    """Series rule: the video must END on the gghuman.ai call-to-action (yellow button visible in the last frames)."""
    out = []
    if tl["meta"].get("series") not in ("private_view", "rounds"):
        return out
    last = tl["shots"][-1]
    out.append(_chk("ends_on_cta_card", last["type"] == "endcard", last["type"], "last shot is the endcard"))
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 4))
    ok, fr = cap.read()
    cap.release()
    if ok:
        reg = fr[930:1100, 100:620].astype(int)
        yellow = int(np.sum((np.abs(reg[..., 2] - 255) < 25) & (np.abs(reg[..., 1] - 214) < 28) & (reg[..., 0] < 120)))
        out.append(_chk("cta_button_visible_last_frame", yellow > 6000, yellow, ">6000 yellow button pixels in the final frame"))
    d = tl["meta"]["duration"]
    out.append(_chk("series_length_s", 18 <= d <= 27, round(d, 1), "18–27 s (page reels are ~20 s)", level="warn"))
    return out


def check_script(tl):
    out = []
    norm = lambda s: re.sub(r"[^a-z0-9']", "", s.lower())
    cap_words = [norm(w) for c in tl["captions"] for w in c["text"].split()]
    miss = []
    for s in tl["shots"]:
        for l in s["lines"]:
            for w in l["text"].split():
                if norm(w) and norm(w) not in cap_words:
                    miss.append(w)
    out.append(_chk("every_spoken_word_captioned", not miss, miss[:5] or "ok", "captions ⊇ dialogue"))
    sp = sorted(tl["speech"])
    ov = any(sp[i][0] < sp[i - 1][1] - 1e-3 for i in range(1, len(sp)))
    out.append(_chk("no_overlapping_voices", not ov, ov, "False"))
    hook = tl["meta"]["hook"]["text"]
    out.append(_chk("hook_length", len(hook) <= 70, len(hook), "≤70 chars", level="warn"))
    first_voice = min([s[0] for s in tl["speech"]] or [0])
    out.append(_chk("first_line_within_1s", first_voice <= 1.0, round(first_voice, 2), "≤1.0 s (hook payoff starts immediately)", level="fail"))
    return out


def check_voice_backend(tl, allow_fallback=False):
    """A video voiced by the babble/espeak fallback must never be marked ok: it would pass every timing check."""
    used = tl["meta"].get("tts_backends", [])
    ok = used == ["kokoro"] or (allow_fallback and bool(used))
    return [_chk("neural_voice_used", ok, ",".join(used) or "none", "kokoro (neural TTS) for every line")]


def check_speech_asr(x, tl, min_mean=0.85, min_line=0.55, strict=False):
    """Transcribe every scripted line from the final audio and compare with the script (Whisper tiny.en, offline)."""
    from . import asr
    if not asr.available():
        return [_chk("speech_intelligible_asr", False, "asr model missing", "run: python setup_assets.py", level="fail" if strict else "warn",
                     note="speech intelligibility NOT verified")]
    res = asr.check_lines(x, SR, tl)
    sims = [r[3] for r in res]
    mean = float(np.mean(sims)) if sims else 1.0
    worst = min(res, key=lambda r: r[3]) if res else None
    bad = [f"{r[1]!r} heard {r[2]!r}" for r in res if r[3] < min_line]
    out = [_chk("speech_intelligible_asr", mean >= min_mean and not bad, round(mean, 3), f"mean line similarity ≥{min_mean} and every line ≥{min_line}",
                note="; ".join(bad[:3]) or (f"worst: {worst[1]!r} heard {worst[2]!r} ({worst[3]:.2f})" if worst else ""))]
    out[0]["lines"] = [dict(id=r[0], expected=r[1], heard=r[2], sim=round(r[3], 3)) for r in res]
    return out


def run_qa(final_mp4, tl_data, voice_stem=None, allow_fallback_voice=False, asr=True, strict_asr=False):
    checks = []
    checks += check_container(final_mp4, tl_data)
    checks += check_voice_backend(tl_data, allow_fallback_voice)
    a, lufs, xfinal = check_audio(final_mp4, tl_data, voice_stem)
    checks += a
    if asr:
        checks += check_speech_asr(xfinal, tl_data, strict=strict_asr)
    v, first = check_video(final_mp4, tl_data)
    checks += v
    checks += check_script(tl_data)
    checks += check_cta(final_mp4, tl_data)
    fails = [c for c in checks if c["status"] == "fail"]
    warns = [c for c in checks if c["status"] == "warn"]
    fixes = {}
    for c in fails:
        if c["name"] == "loudness_lufs":
            fixes["gain_db"] = -14.0 - float(c["value"])
        if c["name"] in ("peak", "clipping"):
            fixes["ceiling_scale"] = 0.85
        if c["name"] == "final_mix_lag_vs_voice_ms":
            fixes["mux_offset_s"] = -float(c["value"]) / 1000
    return dict(passed=not fails, n_pass=sum(c["status"] == "pass" for c in checks), n_warn=len(warns), n_fail=len(fails), checks=checks, fixes=fixes, lufs=lufs), first
