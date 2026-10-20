"""Episode spec (JSON) validation.  Returns human-readable errors so an LLM /
script generator can be asked to fix its own output (see scriptgen.py)."""
import re
from .art.chars import CAST, EXPR

SHOT_TYPES = {"wide", "two", "close", "blackout", "secret", "secret_split", "vote", "reveal",
              "table", "insert", "private", "scream", "endcard"}
SPEAKERS_EXTRA = {"narrator"}
GAMES = {"poker", "catan"}
PROPS = {"poker", "chips", "score", "board"}
TRANS = {"cut", "whip", "flash", "dip", "iris"}
ACTIONS = {"peek", "eliminate", "hand"}
ANCHOR = re.compile(r"^(?:(?:L(\d+)\.)?(start|end|mid))?\s*([+-]\s*\d+(?:\.\d+)?)?$")
MAX_LINE_WORDS = 14
MAX_DUR = 45.0


def parse_anchor(a):
    """'L1.end+0.2' / 'start' / 0.4  ->  (line_idx|None, which, offset)"""
    if isinstance(a, (int, float)):
        return None, "start", float(a)
    m = ANCHOR.match(str(a).strip())
    if not m or not str(a).strip():
        raise ValueError(f"bad anchor {a!r}")
    li, which, off = m.groups()
    off = float(off.replace(" ", "")) if off else 0.0
    return (int(li) if li is not None else None), (which or "start"), off


def validate(spec):
    errs = []
    E = errs.append
    for k in ("id", "cast", "hook", "shots"):
        if k not in spec:
            E(f"missing top-level key '{k}'")
    if errs:
        return errs
    cast = spec["cast"]
    for c in cast:
        if c not in CAST:
            E(f"unknown cast member '{c}' (known: {sorted(CAST)})")
    if not (isinstance(spec["hook"], dict) and spec["hook"].get("text")):
        E("hook.text is required (the handwritten POV line)")
    if not spec["shots"]:
        E("no shots")
    reveals = 0
    for i, s in enumerate(spec["shots"]):
        p = f"shot[{i}]"
        t = s.get("type")
        if t not in SHOT_TYPES:
            E(f"{p}: bad type {t!r}")
            continue
        if s.get("trans_in", "cut") not in TRANS:
            E(f"{p}: bad trans_in {s.get('trans_in')!r}")
        who_fields = {"close": ["who"], "two": ["pair"], "secret": ["who", "action"], "vote": ["votes"], "reveal": ["who"],
                      "table": ["pair", "game"], "insert": ["prop"], "private": ["who", "hand"], "scream": [], "endcard": ["game"]}
        for f in who_fields.get(t, []):
            if f not in s:
                E(f"{p}: type '{t}' needs '{f}'")
        for w in ([s["who"]] if isinstance(s.get("who"), str) else s.get("who", []) or []) + list(s.get("pair", [])):
            if w not in cast:
                E(f"{p}: '{w}' not in cast")
        if t in ("table", "endcard") and s.get("game") not in GAMES:
            E(f"{p}: game must be one of {sorted(GAMES)}")
        if t == "insert" and s.get("prop", {}).get("kind") not in PROPS:
            E(f"{p}: prop.kind must be one of {sorted(PROPS)}")
        if t == "private" and not s.get("hand"):
            E(f"{p}: private view needs a 'hand' list (e.g. ['As','Ah'] or ['VP','VP'])")
        if t == "secret" and s.get("action") not in ACTIONS:
            E(f"{p}: secret action must be one of {sorted(ACTIONS)}")
        if t == "secret" and s.get("action") == "eliminate" and s.get("target") not in cast:
            E(f"{p}: eliminate needs a valid 'target'")
        if t == "vote":
            for v in s.get("votes", []):
                if v.get("from") not in cast or v.get("to") not in cast:
                    E(f"{p}: bad vote {v}")
        if t == "secret_split":
            if len(s.get("panels", [])) != 2:
                E(f"{p}: secret_split needs exactly 2 panels")
            for pn in s.get("panels", []):
                if pn.get("who") not in cast or pn.get("action") not in ACTIONS:
                    E(f"{p}: bad panel {pn}")
        if t == "reveal":
            reveals += 1
            if not s.get("title"):
                E(f"{p}: reveal needs a 'title'")
        for j, ln in enumerate(s.get("lines", [])):
            if ln.get("who") not in cast and ln.get("who") not in SPEAKERS_EXTRA:
                E(f"{p}.lines[{j}]: speaker '{ln.get('who')}' not in cast")
            txt = ln.get("text", "")
            if not txt.strip():
                E(f"{p}.lines[{j}]: empty text")
            if len(txt.split()) > MAX_LINE_WORDS:
                E(f"{p}.lines[{j}]: line too long for a reel caption ({len(txt.split())} words > {MAX_LINE_WORDS})")
            if ln.get("expr") and ln["expr"] not in EXPR:
                E(f"{p}.lines[{j}]: unknown expr {ln['expr']!r}")
        for key in ("sfx", "react", "beats"):
            for j, ev in enumerate(s.get(key, [])):
                try:
                    parse_anchor(ev.get("at", 0))
                except ValueError as e:
                    E(f"{p}.{key}[{j}]: {e}")
                if key == "beats":
                    if ev.get("who") not in cast:
                        E(f"{p}.beats[{j}]: '{ev.get('who')}' not in cast")
                    ex = ev.get("set", {}).get("expr")
                    if ex and ex not in EXPR:
                        E(f"{p}.beats[{j}]: unknown expr {ex!r}")
        for ch, st in s.get("state", {}).items():
            if ch not in cast:
                E(f"{p}.state: '{ch}' not in cast")
            if st.get("expr") and st["expr"] not in EXPR:
                E(f"{p}.state.{ch}: unknown expr {st['expr']!r}")
        if s.get("silence_lead") and i == 0:
            E(f"{p}: silence_lead on first shot")
    for i, s in enumerate(spec["shots"]):
        if s.get("tag") is not None and not (isinstance(s["tag"], str) and 1 <= len(s["tag"]) <= 14):
            E(f"shot[{i}].tag must be a string of 1-14 characters")
        ty = s.get("tally")
        if ty is not None and not (isinstance(ty, dict) and ty.get("rows") and all(len(r) == 3 for r in ty["rows"]) and (ty.get("to") is None or all(len(r) == 3 for r in ty["to"]))):
            E(f"shot[{i}].tally needs rows [[kind,label,value],...] (and optional 'to' rows)")
        stk = (s.get("prop") or {}).get("stacks")
        if stk is not None and not (len(stk) == 3 and all(isinstance(n, int) and 1 <= n <= 10 for n in stk)):
            E(f"shot[{i}]: prop.stacks must be 3 integers between 1 and 10")
    if "board_seed" in spec and not (isinstance(spec["board_seed"], int) and spec["board_seed"] >= 0):
        E("board_seed must be a non-negative integer")
    for i, s in enumerate(spec["shots"]):
        for j, r in enumerate(s.get("react", []) or []):
            if not str(r.get("text", "")).strip() or len(str(r["text"])) > 14:
                E(f"shot[{i}].react[{j}]: text must be 1-14 characters")
        lab = (s.get("prop") or {}).get("label")
        if lab is not None and (not isinstance(lab, str) or len(lab) > 12):
            E(f"shot[{i}]: prop.label must be a string of at most 12 characters")
    if reveals < 1 and not any(x.get("type") in ("scream", "endcard") for x in spec["shots"]):
        E("episode needs at least one 'reveal' or 'scream' shot")
    if spec["shots"] and spec["shots"][-1].get("type") != "reveal" and not spec["shots"][-1].get("lines"):
        pass
    if spec.get("series") in ("private_view", "rounds") and not errs:
        from .gamelogic import check_series
        errs += check_series(spec)
    return errs
