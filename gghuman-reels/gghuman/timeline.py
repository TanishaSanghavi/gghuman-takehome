"""Compile an episode spec into a Timeline: ONE shot-by-shot, frame-snapped
schedule that the video renderer, the audio mixer and the QA all read.

Every voice line, caption chunk, SFX, reaction word, pose change and music cue
gets an absolute time here.  Nothing downstream invents timing."""
import hashlib, json, logging
from .config import FPS, snap
from .spec import validate, parse_anchor
from .tts import render_line

log = logging.getLogger("gghuman.timeline")

ENERGY = {"table": 0.3, "insert": 0.3, "private": 0.2, "scream": 0.5, "endcard": 0.2, "wide": 0.45, "two": 0.5, "close": 0.5, "blackout": 0.12, "secret": 0.22,
          "secret_split": 0.25, "vote": 0.7, "reveal": 0.95, "card": 0.4}
DEFAULT_PAD = {"table": 0.3, "insert": 0.3, "private": 0.5, "scream": 0.0, "endcard": 0.0, "wide": 0.35, "two": 0.3, "close": 0.3, "blackout": 0.0, "secret": 0.5, "secret_split": 0.5,
               "vote": 0.5, "reveal": 0.0, "card": 0.3}
MIN_DUR = {"table": 0.8, "insert": 0.8, "private": 2.4, "scream": 1.4, "endcard": 3.4, "blackout": 1.4, "secret": 2.4, "secret_split": 3.2, "vote": 2.4, "reveal": 2.6, "card": 0.8}
TRANS_SFX = {"whip": ("whoosh", 0.55), "flash": ("flash", 0.5), "iris": ("iris", 0.6), "dip": (None, 0)}


class Timeline:
    def __init__(self, data, clips):
        self.data, self.clips = data, clips

    def dump(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1)

    @property
    def duration(self):
        return self.data["meta"]["duration"]

    @property
    def shots(self):
        return self.data["shots"]


def _lead(s):
    return s.get("silence_lead", {"reveal": 0.75, "scream": 0.6}.get(s["type"], 0))


def _resolve(shot_t0, dur, lines, anchor, default=0.0):
    li, which, off = parse_anchor(anchor)
    if isinstance(anchor, (int, float)):
        return snap(shot_t0 + float(anchor))
    if li is None:
        base = shot_t0 if which == "start" else (shot_t0 + dur if which == "end" else shot_t0 + dur / 2)
    else:
        if li >= len(lines):
            raise ValueError(f"anchor {anchor!r}: shot has only {len(lines)} lines")
        L = lines[li]
        base = {"start": L["t0"], "end": L["t1"], "mid": (L["t0"] + L["t1"]) / 2}[which]
    return snap(base + off)


def _chunks(words, line_t0, speaker, shot_i, style):
    """Group word timings into caption chunks (<=2 short words, break on punctuation)."""
    out, cur = [], []
    def flush():
        if cur:
            out.append(dict(text=" ".join(w for w, _, _ in cur), t0=cur[0][1], t1=cur[-1][2], speaker=speaker, shot=shot_i, style=style))
    for w, a, b in words:
        cur.append((w, line_t0 + a, line_t0 + b))
        brk = w.endswith((",", ".", "!", "?", "…", ";", ":")) or len(cur) >= 2 or sum(len(x[0]) for x in cur) > 13
        if brk:
            flush()
            cur = []
    flush()
    return out


def compile_episode(spec, voices=None, tts_fn=render_line):
    errs = validate(spec)
    if errs:
        raise ValueError("invalid episode spec:\n  - " + "\n  - ".join(errs))
    shots_in = spec["shots"]
    voices = voices or spec.get("voices")
    # silence-before-reveal needs the previous shot to keep an empty tail
    min_tail = {}
    for i, s in enumerate(shots_in):
        lead = _lead(s)
        if lead and i > 0:
            min_tail[i - 1] = lead

    clips, shots_out, captions, sfx, mute, reveals = {}, [], [], [], [], []
    t = 0.0
    for i, s in enumerate(shots_in):
        ty = s["type"]
        t0 = snap(t)
        lines, cur = [], s.get("gap_before", 0.12 if ty != "reveal" else 0.0)
        prev_who = None
        for j, ln in enumerate(s.get("lines", [])):
            clip = tts_fn(ln["text"], ln["who"], style=ln.get("style"), speed=ln.get("speed"), spec_voices=voices)
            gap = ln.get("gap", 0.16 if prev_who == ln["who"] else 0.1) if j else 0.0
            ls = snap(t0 + cur + gap + ln.get("pre", 0.0))
            le = ls + clip.dur - clip.onset          # ls = moment the line becomes AUDIBLE
            lid = f"s{i}l{j}"
            clips[lid] = clip
            lines.append(dict(id=lid, who=ln["who"], text=ln["text"], t0=ls, t1=le, style=ln.get("style"), expr=ln.get("expr"), onset=clip.onset,
                              words=[(w, ls + a - clip.onset, ls + b - clip.onset) for w, a, b in clip.words]))
            cur = le - t0
            prev_who = ln["who"]
            captions += _chunks([(w, a - clip.onset, b - clip.onset) for w, a, b in clip.words], ls, ln["who"], i, ln.get("style"))
        content_end = lines[-1]["t1"] - t0 if lines else 0.0
        pad = s.get("pad", DEFAULT_PAD[ty])
        dur = max(s.get("dur", 0.0), MIN_DUR.get(ty, 0.5) if not s.get("dur") else 0.0, content_end + pad)
        # explicit event anchors may push duration
        ev_end = 0.0
        react, sfx_s, beats = [], [], []
        tmp_dur = max(dur, 0.1)
        for r in s.get("react", []):
            ra = _resolve(t0, tmp_dur, lines, r.get("at", "start"))
            react.append(dict(r, t0=ra, t1=ra + r.get("dur", 0.9)))
            ev_end = max(ev_end, react[-1]["t1"] - t0 - 0.15)
        for e in s.get("sfx", []):
            sfx_s.append(dict(t=_resolve(t0, tmp_dur, lines, e.get("at", 0)), name=e["name"], gain=e.get("gain", 1.0), pan=e.get("pan", 0.0), shot=i))
        for b in s.get("beats", []):
            beats.append(dict(b, t=_resolve(t0, tmp_dur, lines, b.get("at", 0))))
        beats.sort(key=lambda b: b["t"])
        dur = max(dur, ev_end, min_tail.get(i, 0) + max(content_end, max([e["t"] - t0 for e in sfx_s if e["name"] not in ("heartbeat", "tick")] or [0])))
        if i + 1 < len(shots_in) and shots_in[i + 1].get("trans_in") in ("whip", "flash"):
            pass
        t1 = snap(t0 + dur)
        if t1 - t0 < 0.4:
            t1 = snap(t0 + 0.4)
        # auto sfx -------------------------------------------------------
        auto = s.get("auto_sfx", True)
        if auto:
            if i == 0:
                sfx_s.append(dict(t=t0 + 0.04, name="scribble", gain=0.6, pan=0, shot=i))
            if ty == "wide" and not s.get("lines") is None and i == 0:
                sfx_s.append(dict(t=t0 + 0.0, name="knock", gain=0.8, pan=0, shot=i))
            if ty == "blackout":
                sfx_s += [dict(t=t0 + 0.0, name="switch", gain=1.0, pan=0, shot=i), dict(t=t0 + 0.04, name="power_down", gain=0.9, pan=0, shot=i),
                          dict(t=t0 + 0.55, name="pop_big", gain=0.8, pan=0, shot=i)]
            if ty in ("secret", "secret_split", "private"):
                sfx_s.append(dict(t=t0 + 0.03, name="rec_beep", gain=0.7, pan=0, shot=i))
            tr = s.get("trans_in", "cut")
            if tr in TRANS_SFX and TRANS_SFX[tr][0] and i > 0:
                sfx_s.append(dict(t=t0 - 0.0, name=TRANS_SFX[tr][0], gain=TRANS_SFX[tr][1], pan=0, shot=i, peak=True))
            for r in react:
                sfx_s.append(dict(t=r["t0"], name=r.get("sfx", "pop"), gain=0.8, pan=0, shot=i))
        shot_d = dict(i=i, id=s.get("id", f"s{i}"), type=ty, t0=t0, t1=t1, spec=s, lines=lines, react=react, beats=beats,
                      trans=s.get("trans_in", "cut"), energy=s.get("energy", ENERGY[ty]))
        # secret / vote action timing
        if ty in ("secret", "secret_split", "private"):
            shot_d["stamp_t"] = [snap(t0 + (s.get("stamp_at", (t1 - t0) * 0.62)) + k * 0.0) for k in range(1)]
            if ty == "secret_split":
                st = []
                for k, pn in enumerate(s["panels"]):
                    st.append(snap(t0 + pn.get("stamp_at", (t1 - t0) * (0.5 + 0.2 * k))))
                shot_d["stamp_t"] = st
            if auto:
                for st in shot_d["stamp_t"]:
                    act = s.get("action") or "eliminate"
                    sfx_s.append(dict(t=st - 0.45, name="slide", gain=0.6, pan=0, shot=i))
                    kinds = [p.get("action") for p in s.get("panels", [])] or [s.get("action") or ("hand" if ty == "private" else None)]
                    sfx_s.append(dict(t=st, name="stamp" if "eliminate" in kinds else ("flip" if "hand" in kinds else "pop_big"), gain=1.0, pan=0, shot=i))
        if ty == "vote":
            order = s.get("order") or []
            nv = len(s.get("votes", []))
            step = (t1 - t0 - 0.9) / max(1, nv)
            shot_d["vote_t"] = [snap(t0 + 0.5 + k * step) for k in range(nv)]
            if auto:
                for vt in shot_d["vote_t"]:
                    sfx_s.append(dict(t=vt, name="tick_pop", gain=0.9, pan=0, shot=i))
        if ty == "scream":
            reveals.append(dict(t=t0, title="SCREAM", shot=i))
            if auto:
                sfx_s.append(dict(t=t0, name="scream", gain=1.0, pan=0, shot=i))
        if ty == "endcard":
            ec = s.get("timing", {})
            shot_d["ec"] = dict(name=ec.get("name", 0.9), button=ec.get("button", 1.5), cursor=ec.get("cursor", 1.9), click=ec.get("click", 2.6))
            if auto:
                for nm, key, g_ in (("tick_pop", "line1", 0.7), ("pop", "name", 0.8), ("pop", "button", 0.8), ("click", "click", 1.0), ("ding", "click", 0.7)):
                    off = 0.05 if key == "line1" else shot_d["ec"][key]
                    sfx_s.append(dict(t=snap(t0 + off), name=nm, gain=g_, pan=0, shot=i))
        if ty == "reveal":
            reveals.append(dict(t=t0, title=s["title"], shot=i))
            if auto:
                sfx_s += [dict(t=t0, name="boom", gain=1.0, pan=0, shot=i), dict(t=t0, name="sting", gain=0.9, pan=0, shot=i),
                          dict(t=t0 + 0.28, name="gasp", gain=0.8, pan=0, shot=i), dict(t=t0 + 0.35, name="ding", gain=0.8, pan=0, shot=i)]
        sfx += sfx_s
        shots_out.append(shot_d)
        t = t1
    total = snap(t + 0.0)
    # silence windows right before every reveal ------------------------------
    for k, s in enumerate(shots_out):
        lead = _lead(shots_in[k])
        if lead and k > 0:
            mute.append([snap(s["t0"] - lead), s["t0"]])
    # drop anything that would break the silence
    keep = []
    for e in sfx:
        inside = any(a - 1e-6 <= e["t"] < b - 1e-6 for a, b in mute)
        if inside:
            log.info("dropping sfx %s at %.2f inside silence window", e["name"], e["t"])
        else:
            keep.append(e)
    # risers: from a shot flagged riser:true until the next mute window / reveal
    risers = []
    for k, s in enumerate(shots_out):
        if shots_in[k].get("riser"):
            end = next((a for a, b in mute if a > s["t0"]), None)
            if end is None:
                end = next((x["t0"] for x in shots_out[k + 1:] if x["type"] == "reveal"), total)
            risers.append([s["t0"], end])
    # speech windows for music ducking + QA
    speech = [[l["t0"], l["t1"], l["who"], l["id"]] for s in shots_out for l in s["lines"]]
    meta = dict(id=spec["id"], title=spec.get("title", spec["id"]), fps=FPS, duration=total, hook=spec["hook"], cast=spec["cast"],
                music=spec.get("music", {}), n_shots=len(shots_out), series=spec.get("series"), game=spec.get("game"),
                tts_backends=sorted({c.backend for c in clips.values()}))
    data = dict(meta=meta, shots=shots_out, captions=captions, sfx=sorted(keep, key=lambda e: e["t"]), mute=mute, risers=risers,
                reveals=reveals, speech=speech)
    data["fingerprint"] = hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return Timeline(data, clips)
