"""Hand-drawn character rigs: Wren/Red (humans) and GPT/Claude/Gemini (TV-head robots).

A character is drawn as two cached layers (head + body) so the head can bob /
tilt / lip-sync independently.  Everything is deterministic (seeded) so the
same spec always renders the same frames."""
from functools import lru_cache
import numpy as np
from .draw import Cv, blob, rrect, arc_pts, shade, INK

SHIRT_INK = INK
CAST = {
    "wren":   dict(kind="human", name="WREN",   head=(88, 180, 42),  body=(42, 143, 220), seed=11),
    "red":    dict(kind="human", name="RED",    head=(232, 56, 43),  body=(61, 74, 70),   seed=23),
    "gpt":    dict(kind="robot", name="GPT",    body=(64, 80, 74),   eye=(70, 222, 204),  emblem="knot",     ball=(232, 56, 43),  seed=31),
    "claude": dict(kind="robot", name="CLAUDE", body=(207, 122, 88), eye=(244, 169, 138), emblem="burst",    ball=(205, 110, 80), seed=41),
    "gemini": dict(kind="robot", name="GEMINI", body=(122, 94, 200), eye=(178, 160, 255), emblem="star",     ball=(122, 94, 200), seed=53),
}
DARK_RED = (92, 20, 20, 255)
WHITE = (250, 247, 240, 255)
FRAME = (233, 229, 244, 255)
SCREEN = (22, 20, 30, 255)

# expression presets: eyes / brows / mouth / extras / (fixed mouth level for open mouths)
EXPR = {
    "neutral": dict(eyes="dot", brows="none", mouth="flat"),
    "blank":   dict(eyes="dot", brows="none", mouth="flat"),
    "worried": dict(eyes="dot", brows="worried", mouth="frown"),
    "shock":   dict(eyes="wide", brows="raised", mouth="open", level=3),
    "gasp":    dict(eyes="wide", brows="raised", mouth="o"),
    "smug":    dict(eyes="squint", brows="raised", mouth="smirk"),
    "angry":   dict(eyes="dot", brows="angry", mouth="frown"),
    "sus":     dict(eyes="squint", brows="flat", mouth="flat"),
    "happy":   dict(eyes="arc_up", brows="none", mouth="grin"),
    "nervous": dict(eyes="wide", brows="worried", mouth="wavy", extras=("sweat",)),
    "plead":   dict(eyes="wide", brows="worried", mouth="o"),
    "dead":    dict(eyes="x", brows="none", mouth="flat"),
    "sleep":   dict(eyes="closed", brows="none", mouth="tiny"),
    "evil":    dict(eyes="squint", brows="angry", mouth="grin"),
    "deadpan": dict(eyes="dot", brows="flat", mouth="flat"),
    "calm":    dict(eyes="dot", brows="none", mouth="smile"),
    "caught":  dict(eyes="wide", brows="worried", mouth="wavy", extras=("sweat", "fedora")),
    "busted":  dict(eyes="wide", brows="worried", mouth="wavy", extras=("sweat",)),
    "mafia":   dict(eyes="squint", brows="angry", mouth="smirk", extras=("fedora",)),
    "shh":     dict(eyes="squint", brows="none", mouth="tiny", extras=("shh",)),
}

HX0, HX1, HY0, HY1 = -360, 360, -800, 70   # head canvas (design units)


def expr_key(name, level=0, gx=0, gy=0, extra=()):
    e = EXPR.get(name, EXPR["neutral"])
    ex = tuple(sorted(set(e.get("extras", ())) | set(extra)))
    lvl = e.get("level", level) if e["mouth"] == "open" else level
    mouth = e["mouth"]
    if level > 0 and mouth in ("flat", "smirk", "frown", "smile", "tiny", "wavy", "grin"):
        mouth = "open_" + mouth if mouth in ("smirk", "frown", "grin", "smile", "wavy") else "open"
    return (e["eyes"], e["brows"], mouth, int(lvl), int(gx), int(gy), ex)


# ---------------------------------------------------------------- humans
def _mouth_human(c, m, lvl, ink=INK):
    y = -128
    if m == "flat":
        c.line([(-36, y), (-10, y + 3), (36, y)], ink, 9)
    elif m == "tiny":
        c.line([(-18, y), (18, y)], ink, 9)
    elif m in ("open", "open_flat"):
        rx, ry = 40 + lvl * 9, 10 + lvl * 20
        c.d.ellipse([c.P(-rx, y - ry + 8)[0], c.P(-rx, y - ry + 8)[1], c.P(rx, y + ry + 8)[0], c.P(rx, y + ry + 8)[1]],
                    fill=DARK_RED, outline=ink, width=c.lw(8))
    elif m == "o":
        c.circle(0, y + 4, 24, DARK_RED, ink, 8)
    elif m in ("smirk", "open_smirk"):
        c.line([(-44, y + 8), (-10, y + 12), (30, y - 4), (52, y - 28)], ink, 9)
        if m == "open_smirk":
            c.circle(0, y + 8, 22 + 4 * lvl, DARK_RED, ink, 8)
    elif m in ("frown", "open_frown"):
        c.line([(-48, y + 14), (-22, y - 6), (22, y - 6), (48, y + 14)], ink, 9)
        if m == "open_frown":
            c.circle(0, y + 6, 20 + 4 * lvl, DARK_RED, ink, 8)
    elif m in ("smile", "open_smile"):
        c.line(arc_pts(0, y - 30, 56, 44, 25, 155), ink, 9)
        if m == "open_smile":
            c.circle(0, y + 6, 22 + 4 * lvl, DARK_RED, ink, 8)
    elif m in ("wavy", "open_wavy"):
        c.line([(-50, y), (-30, y - 12), (-10, y + 8), (10, y - 12), (30, y + 8), (50, y - 6)], ink, 9)
        if m == "open_wavy":
            c.circle(0, y + 10, 20 + 4 * lvl, DARK_RED, ink, 8)
    elif m in ("grin", "open_grin"):
        pts = arc_pts(0, y - 50, 70, 70, 28, 152)
        c.poly(pts, WHITE if m == "grin" else DARK_RED, ink, 8)
        if m == "grin":
            c.line([(-40, y + 8), (40, y + 8)], ink, 4)


def _eyes_human(c, e, gx, gy, ink=INK, dot=INK):
    for sx in (-1, 1):
        cx, cy = sx * 88, -250
        if e == "dot":
            c.circle(cx + gx * 8, cy + gy * 8, 17 if dot is not INK else 15, dot, None, 0)
        elif e in ("wide", "squint"):
            R = 60 if e == "wide" else 42
            c.circle(cx, cy, R, WHITE, ink, 8)
            c.circle(cx + gx * 16, cy + gy * 14, 22 if e == "wide" else 18, ink, None, 0)
            if e == "squint":  # heavy lid
                c.poly([(cx - R - 4, cy - R - 6), (cx + R + 4, cy - R - 6), (cx + R + 4, cy - 6), (cx - R - 4, cy - 6)],
                       c.img.getpixel((0, 0)) and None, None, 0)
                c.line([(cx - R, cy - 8), (cx + R, cy - 8)], ink, 10)
        elif e == "x":
            c.line([(cx - 34, cy - 34), (cx + 34, cy + 34)], ink, 12)
            c.line([(cx + 34, cy - 34), (cx - 34, cy + 34)], ink, 12)
        elif e == "closed":
            c.line(arc_pts(cx, cy - 22, 36, 30, 30, 150), ink, 10)
        elif e == "arc_up":
            c.line(arc_pts(cx, cy + 22, 36, 34, 205, 335), ink, 11)


def _brows_human(c, b, ink=INK):
    if b == "none":
        return
    for sx in (-1, 1):
        if b == "worried":
            c.line([(sx * 135, -338), (sx * 52, -372)], ink, 10)
        elif b == "angry":
            c.line([(sx * 135, -372), (sx * 52, -330)], ink, 10)
        elif b == "raised":
            c.line(arc_pts(sx * 90, -300, 56, 46, 205, 335), ink, 9)
        elif b == "flat":
            c.line([(sx * 135, -348), (sx * 45, -348)], ink, 10)


def _hat(c, top_y, w):
    """Fedora sitting on the head top (top_y = top of the head)."""
    hat = (43, 43, 54, 255)
    bw = w * 0.80
    cw0, cw1, ch = bw * 0.30, bw * 0.21, 175
    base = top_y + 40
    crown = [(-cw0, base), (-cw1 - 8, base - ch + 30), (-cw1 + 14, base - ch), (-8, base - ch + 26),
             (8, base - ch + 26), (cw1 - 14, base - ch), (cw1 + 8, base - ch + 30), (cw0, base)]
    c.poly(blob(0, base + 6, bw * 0.5, 30, 5, 0, 0.01, 48), hat, INK, 9)       # brim
    c.poly(crown, hat, INK, 9)
    c.line([(-cw0 + 4, base - 34), (cw0 - 4, base - 34)], (178, 40, 40, 255), 34, caps=False)
    c.line(blob(0, base + 6, bw * 0.5, 30, 5, 0, 0.01, 48)[24:60] , hat, 9)


def _hand_shh(c, hand, ink=INK):
    c.line([(34, -70), (6, -165)], ink, 58, True)
    c.line([(34, -70), (6, -165)], hand, 40, True)
    c.circle(40, -62, 46, hand, ink, 9)


def head_human(cd, ex, s, boil, layer="all"):
    eyes, brows, mouth, lvl, gx, gy, extras = ex
    c = Cv(HX0, HY0, HX1, HY1, s)
    col = cd["head"] + (255,)
    sd = cd["seed"]
    C = (0, -230)
    full = layer == "all"
    # hair
    for i, ang in enumerate((-118, -97, -76, -55)[:4 if sd % 2 else 3] if full else []):
        a = np.radians(ang)
        d = np.array([np.cos(a), np.sin(a)])
        j = ((sd + i * 7 + boil * 3) % 5 - 2) * 4
        p0 = np.array(C) + d * 238
        p1 = np.array(C) + d * (292 + j)
        c.line([tuple(p0), tuple(p1)], INK, 11)
    if full:
        c.poly(blob(*C, 250, 250, sd, boil, 0.022), col, INK, 11)
    _eyes_human(c, eyes, gx, gy, INK, INK if full else (225, 238, 255, 255))
    if full:
        _brows_human(c, brows)
        _mouth_human(c, mouth, lvl)
    if not full:
        return c.finish(), int(-HX0 * s), int(-HY0 * s)
    if "sweat" in extras:
        c.poly([(178, -420), (205, -372), (178, -345), (151, -372)], (150, 205, 245, 255), INK, 7)
    if "shh" in extras:
        _hand_shh(c, col)
    if "fedora" in extras:
        _hat(c, -470, 560)
    return c.finish(), int(-HX0 * s), int(-HY0 * s)


# ---------------------------------------------------------------- robots
def _eyes_robot(c, e, col, gx, gy, boil, sd):
    for sx in (-1, 1):
        cx, cy = sx * 96 + gx * 22, -272 + gy * 14
        if e in ("dot", "bars"):
            c.poly(rrect(cx, cy, 76, 42, 18, sd, boil, 0.01), col, None, 0)
        elif e == "wide":
            c.poly(rrect(cx, cy, 72, 76, 26, sd, boil, 0.01), col, None, 0)
            c.circle(cx + gx * 8, cy + gy * 8, 11, SCREEN, None, 0)
        elif e == "squint":
            c.poly(rrect(cx, cy, 90, 16, 8, sd, boil, 0.01), col, None, 0)
        elif e == "x":
            c.line([(cx - 32, cy - 32), (cx + 32, cy + 32)], col, 14)
            c.line([(cx + 32, cy - 32), (cx - 32, cy + 32)], col, 14)
        elif e == "closed":
            c.line(arc_pts(cx, cy - 18, 36, 26, 30, 150), col, 13)
        elif e == "arc_up":
            c.line(arc_pts(cx, cy + 22, 38, 34, 205, 335), col, 14)


def _brows_robot(c, b, col):
    if b in ("none", "flat"):
        if b == "flat":
            for sx in (-1, 1):
                c.line([(sx * 140, -338), (sx * 52, -338)], col, 12)
        return
    for sx in (-1, 1):
        if b == "worried":
            c.line([(sx * 140, -332), (sx * 55, -362)], col, 12)
        elif b == "angry":
            c.line([(sx * 140, -364), (sx * 55, -330)], col, 12)
        elif b == "raised":
            c.line(arc_pts(sx * 96, -318, 54, 40, 210, 330), col, 11)


def _mouth_robot(c, m, lvl, col, boil, sd):
    y = -168
    if m == "flat":
        c.line([(-30, y), (30, y)], col, 13)
    elif m == "tiny":
        c.line([(-14, y), (14, y)], col, 13)
    elif m in ("open", "open_flat"):
        c.poly(rrect(0, y, 66 + lvl * 8, 14 + lvl * 20, 12, sd + 3, boil, 0.01), col, None, 0)
        c.poly(rrect(0, y + 2, 44 + lvl * 6, max(2, 6 + lvl * 12), 6, sd + 4, boil, 0.01), SCREEN, None, 0)
    elif m == "o":
        c.circle(0, y, 24, None, col, 11)
    elif m in ("smirk", "open_smirk"):
        c.line([(-36, y + 10), (-6, y + 12), (30, y + 2), (48, y - 22)], col, 13)
    elif m in ("frown", "open_frown"):
        c.line([(-42, y + 14), (-16, y - 4), (16, y - 4), (42, y + 14)], col, 13)
    elif m in ("smile", "open_smile"):
        c.line(arc_pts(0, y - 38, 50, 44, 28, 152), col, 13)
    elif m in ("wavy", "open_wavy"):
        c.line([(-48, y), (-28, y - 12), (-8, y + 8), (12, y - 12), (32, y + 8), (50, y - 4)], col, 12)
    elif m in ("grin", "open_grin"):
        c.poly(arc_pts(0, y - 54, 74, 66, 30, 150), col, None, 0)


def head_robot(cd, ex, s, boil, layer="all"):
    eyes, brows, mouth, lvl, gx, gy, extras = ex
    c = Cv(HX0, HY0, HX1, HY1, s)
    sd = cd["seed"]
    body = cd["body"] + (255,)
    eye = cd["eye"] + (255,)
    full = layer == "all"
    wob = ((boil % 3) - 1) * 4
    if full:  # antenna (behind frame) + frame + screen
        c.line([(0, -470), (4, -520), (24 + wob, -556)], INK, 12)
        c.circle(26 + wob, -576, 30, cd.get("ball", cd["body"]) + (255,), INK, 9)
        c.poly(rrect(0, -250, 520, 460, 78, sd, boil, 0.006), FRAME, INK, 12)
        c.poly(rrect(0, -250, 392, 340, 52, sd + 1, boil, 0.006), SCREEN, INK, 6)
        c.line([(-170, -380), (-120, -400)], (255, 255, 255, 60), 10)  # glare
    _eyes_robot(c, eyes, eye, gx, gy, boil, sd)
    _brows_robot(c, brows, eye)
    _mouth_robot(c, mouth, lvl, eye, boil, sd)
    if not full:
        return c.finish(), int(-HX0 * s), int(-HY0 * s)
    if "sweat" in extras:
        c.poly([(236, -400), (262, -352), (236, -326), (210, -352)], (150, 205, 245, 255), INK, 7)
    if "shh" in extras:
        _hand_shh(c, (250, 247, 240, 255))
    if "fedora" in extras:
        _hat(c, -478, 600)
    return c.finish(), int(-HX0 * s), int(-HY0 * s)


def _emblem(c, kind):
    x, y = 0, 215
    if kind == "asterisk":
        for a in (90, 30, -30):
            r = np.radians(a)
            p = [(x + 74 * np.cos(r), y + 74 * np.sin(r)), (x - 74 * np.cos(r), y - 74 * np.sin(r))]
            c.line(p, INK, 34)
        for a in (90, 30, -30):
            r = np.radians(a)
            p = [(x + 74 * np.cos(r), y + 74 * np.sin(r)), (x - 74 * np.cos(r), y - 74 * np.sin(r))]
            c.line(p, WHITE, 24)
    elif kind == "star":
        pts = [(0, -92), (24, -24), (92, 0), (24, 24), (0, 92), (-24, 24), (-92, 0), (-24, -24)]
        c.poly([(x + a, y + b) for a, b in pts], WHITE, INK, 7)
    elif kind == "burst":
        for w_, col_ in ((26, INK), (13, WHITE)):
            for i in range(12):
                a = np.radians(i * 30)
                c.line([(x + 26 * np.cos(a), y + 26 * np.sin(a)), (x + 88 * np.cos(a), y + 88 * np.sin(a))], col_, w_)
        c.circle(x, y, 22, WHITE, INK, 6)
    elif kind == "knot":
        for w_, col_ in ((30, INK), (17, WHITE)):
            for k in range(3):
                a = np.radians(60 * k)
                pts = [(x + 86 * np.cos(t) * np.cos(a) - 38 * np.sin(t) * np.sin(a),
                        y + 86 * np.cos(t) * np.sin(a) + 38 * np.sin(t) * np.cos(a)) for t in np.linspace(0, 2 * np.pi, 40)]
                c.line(pts, col_, w_, caps=False)
    elif kind == "hex":
        pts = [(x + 80 * np.cos(np.radians(60 * i + 30)), y + 80 * np.sin(np.radians(60 * i + 30))) for i in range(6)]
        c.poly(pts, None, INK, 34)
        c.poly(pts, None, WHITE, 22)


def _arm_draw(c, cd, side, ang, finger, col, hand, L=400):
    sx = 300 * side
    a = np.radians(ang)
    ex_, ey_ = sx + L * np.cos(a), 130 + L * np.sin(a)
    c.line([(sx, 130), (ex_, ey_)], INK, 96)
    tip = None
    if finger:
        tip = (ex_ + 85 * np.cos(a), ey_ + 85 * np.sin(a))
        c.line([(ex_, ey_), tip], INK, 54)
    c.circle(ex_, ey_, 58, hand, INK, 11)
    c.line([(sx, 130), (ex_, ey_)], col, 74)
    if finger:
        c.line([(ex_, ey_), tip], hand, 34)
    c.circle(ex_, ey_, 50, hand, None, 0)


def torso_impl(cd, s, boil):
    c = Cv(-430, -40, 430, 560, s)
    sd = cd["seed"]
    col = cd["body"] + (255,)
    c.poly(blob(0, 340, 390, 300, sd + 9, boil, 0.018), col, INK, 11)
    if cd["kind"] == "robot":
        _emblem(c, cd["emblem"])
    return c.finish(), int(430 * s), int(40 * s)


def arm_impl(cd, arm, s, boil):
    """arm: ('point', angle_deg, side, reach) | ('up', side) | ('both_up',).  Drawn separately so it sits above the table."""
    L = arm[3] if arm[0] == "point" and len(arm) > 3 else 400
    ext = 420 + L
    c = Cv(-ext, -(L + 200), ext, L + 260, s)
    col = cd["body"] + (255,)
    hand = (cd["head"] + (255,)) if cd["kind"] == "human" else WHITE
    if arm[0] == "point":
        _arm_draw(c, cd, arm[2], arm[1], True, col, hand, L)
    elif arm[0] == "up":
        _arm_draw(c, cd, arm[1], -80 if arm[1] > 0 else -100, False, col, hand, 400)
    elif arm[0] == "both_up":
        _arm_draw(c, cd, -1, -105, False, col, hand, 380)
        _arm_draw(c, cd, 1, -75, False, col, hand, 380)
    return c.finish(), int(ext * s), int((L + 200) * s)


@lru_cache(maxsize=4096)
def head_sprite(who, ex, s, boil, layer="all"):
    cd = CAST[who]
    return (head_human if cd["kind"] == "human" else head_robot)(cd, ex, s, boil, layer)


@lru_cache(maxsize=512)
def body_sprite(who, s, boil):
    return torso_impl(CAST[who], s, boil)


@lru_cache(maxsize=512)
def arm_sprite(who, arm, s, boil):
    return arm_impl(CAST[who], arm, s, boil)


@lru_cache(maxsize=64)
def hat_sprite(who, s):
    cd = CAST[who]
    c = Cv(HX0, HY0, HX1, HY1, s)
    _hat(c, -470 if cd["kind"] == "human" else -478, 560 if cd["kind"] == "human" else 600)
    return c.finish(), int(-HX0 * s), int(-HY0 * s)
