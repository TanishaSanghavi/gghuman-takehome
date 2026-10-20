"""Hand-drawn props: cards, X stamp, skull token, lamp, vote board, stickers."""
from functools import lru_cache
import numpy as np, cv2
from PIL import Image, ImageDraw
from .draw import Cv, blob, rrect, arc_pts, INK
from .chars import CAST, head_sprite, expr_key
from .text import font

PAPER = (250, 246, 236, 255)
RED = (214, 52, 44, 255)


def _pil(img):
    return Image.fromarray(img)


@lru_cache(maxsize=64)
def portrait_card(who, size=1.0, boil=0, out=False):
    """Player card: face + name. Used for 'eliminate' secret-cam moves & vote board."""
    w, h = int(250 * size), int(330 * size)
    c = Cv(0, 0, 250, 330, size)
    c.poly(rrect(125, 165, 232, 312, 24, CAST[who]["seed"], boil, 0.006), PAPER, INK, 8)
    hs, ox, oy = head_sprite(who, expr_key("dead" if out else "neutral"), 0.2 * size, 0)
    im = c.finish()
    pil = _pil(im)
    hp = _pil(hs)
    # head sprite origin = neck; place so the face sits in the card
    pil.alpha_composite(hp, (int(125 * size - ox), int(228 * size - oy + 0)))
    d = ImageDraw.Draw(pil)
    f = font("PatrickHand-Regular.ttf", int(40 * size))
    name = CAST[who]["name"]
    d.text((125 * size - f.getlength(name) / 2, 268 * size), name, font=f, fill=(40, 34, 30, 255))
    return np.asarray(pil).copy()


@lru_cache(maxsize=16)
def role_card(role, boil=0):
    c = Cv(0, 0, 300, 400, 1.0)
    mafia = role == "MAFIA"
    c.poly(rrect(150, 200, 276, 376, 28, 77, boil, 0.006), (34, 30, 38, 255) if mafia else PAPER, INK, 9)
    pil = _pil(c.finish())
    d = ImageDraw.Draw(pil)
    f = font("Bangers-Regular.ttf", 58)
    col = (230, 70, 60, 255) if mafia else (60, 120, 70, 255)
    d.text((150 - f.getlength(role) / 2, 300), role, font=f, fill=col)
    c2 = Cv(0, 0, 300, 400, 1.0)
    if mafia:  # skull
        c2.poly(blob(150, 160, 62, 56, 4, boil, 0.02), (240, 236, 226, 255), INK, 6)
        c2.poly(rrect(150, 214, 56, 36, 10, 5, boil), (240, 236, 226, 255), INK, 6)
        c2.circle(128, 158, 15, (20, 18, 24, 255), None, 0)
        c2.circle(172, 158, 15, (20, 18, 24, 255), None, 0)
        c2.poly([(150, 176), (141, 194), (159, 194)], (20, 18, 24, 255), None, 0)
    else:
        c2.poly([(90, 190), (150, 120), (210, 190)], (232, 120, 90, 255), INK, 8)
        c2.poly(rrect(150, 214, 96, 56, 8, 6, boil), (240, 220, 180, 255), INK, 8)
    pil.alpha_composite(_pil(c2.finish()))
    return np.asarray(pil).copy()


@lru_cache(maxsize=4)
def card_back(boil=0):
    c = Cv(0, 0, 300, 400, 1.0)
    c.poly(rrect(150, 200, 276, 376, 28, 78, boil, 0.006), (178, 156, 206, 255), INK, 9)
    pil = _pil(c.finish())
    d = ImageDraw.Draw(pil)
    f = font("Bangers-Regular.ttf", 130)
    d.text((150 - f.getlength("?") / 2, 120), "?", font=f, fill=(250, 246, 236, 255))
    return np.asarray(pil).copy()


@lru_cache(maxsize=32)
def x_mark(progress=1.0, size=220, boil=0):
    """Red hand-drawn X, stroke by stroke (progress 0..1)."""
    c = Cv(0, 0, size, size, 1.0)
    a, b = (30, 30), (size - 30, size - 30)
    c2, d2 = (size - 30, 34), (30, size - 36)
    p1 = min(1, progress * 2)
    p2 = max(0, progress * 2 - 1)
    if p1 > 0:
        e = (a[0] + (b[0] - a[0]) * p1, a[1] + (b[1] - a[1]) * p1)
        c.line([a, ((a[0] + e[0]) / 2 + 3, (a[1] + e[1]) / 2 - 3), e], INK, 40)
        c.line([a, ((a[0] + e[0]) / 2 + 3, (a[1] + e[1]) / 2 - 3), e], RED, 28)
    if p2 > 0:
        e = (c2[0] + (d2[0] - c2[0]) * p2, c2[1] + (d2[1] - c2[1]) * p2)
        c.line([c2, ((c2[0] + e[0]) / 2 - 3, (c2[1] + e[1]) / 2 - 3), e], INK, 40)
        c.line([c2, ((c2[0] + e[0]) / 2 - 3, (c2[1] + e[1]) / 2 - 3), e], RED, 28)
    return c.finish()


@lru_cache(maxsize=8)
def skull_token(boil=0):
    c = Cv(0, 0, 150, 150, 1.0)
    c.circle(75, 75, 68, (34, 30, 38, 255), INK, 8)
    c.poly(blob(75, 68, 34, 31, 4, boil, 0.02), (240, 236, 226, 255), INK, 4)
    c.poly(rrect(75, 100, 30, 20, 6, 5, boil), (240, 236, 226, 255), INK, 4)
    c.circle(63, 66, 8, (20, 18, 24, 255), None, 0)
    c.circle(87, 66, 8, (20, 18, 24, 255), None, 0)
    return c.finish()


@lru_cache(maxsize=8)
def out_tag(boil=0):
    c = Cv(0, 0, 200, 90, 1.0)
    c.poly(rrect(100, 45, 184, 72, 14, 9, boil, 0.01), RED, INK, 7)
    pil = _pil(c.finish())
    d = ImageDraw.Draw(pil)
    f = font("Bangers-Regular.ttf", 58)
    d.text((100 - f.getlength("OUT") / 2, 10), "OUT", font=f, fill=(255, 255, 255, 255))
    return np.asarray(pil).copy()


@lru_cache(maxsize=16)
def mark(kind, boil=0):
    """Tiny comic marks: '?', '!', 'zzz'."""
    pil = Image.new("RGBA", (140, 160), (0, 0, 0, 0))
    d = ImageDraw.Draw(pil)
    f = font("Bangers-Regular.ttf", 130 if kind != "zzz" else 80)
    col = {"?": (255, 214, 64, 255), "!": (232, 56, 43, 255), "zzz": (190, 205, 255, 255)}[kind]
    txt = {"zzz": "Zzz"}.get(kind, kind)
    d.text((70 - f.getlength(txt) / 2, 0), txt, font=f, fill=col, stroke_width=8, stroke_fill=INK)
    return np.asarray(pil).copy()


@lru_cache(maxsize=32)
def tally_board(names, counts, hi=None, boil=0):
    """Vote tally board.  names: tuple of cast keys."""
    n = len(names)
    w, h = 560, 100 + 96 * n
    pil = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    c = Cv(0, 0, w, h, 1.0)
    c.poly(rrect(w / 2, 40, 300, 66, 10, 1, boil, 0.01), PAPER, INK, 7)
    base = _pil(c.finish())
    d = ImageDraw.Draw(base)
    f = font("PatrickHand-Regular.ttf", 44)
    d.text((w / 2 - f.getlength("VOTES") / 2, 14), "VOTES", font=f, fill=(40, 34, 30, 255))
    for i, who in enumerate(names):
        y = 100 + i * 96
        cc = Cv(0, 0, w, h, 1.0)
        fill = (255, 226, 160, 255) if hi == who else PAPER
        cc.poly(rrect(w / 2, y + 40, 530, 76, 12, 2 + i, boil, 0.008), fill, INK, 6)
        base.alpha_composite(_pil(cc.finish()))
        hs, ox, oy = head_sprite(who, expr_key("neutral"), 0.09, 0)
        base.alpha_composite(_pil(hs), (int(62 - ox), int(y + 62 - oy + 34)))
        d = ImageDraw.Draw(base)
        d.text((110, y + 20), CAST[who]["name"], font=f, fill=(40, 34, 30, 255))
        cs = str(counts[i])
        d.text((505 - f.getlength(cs), y + 20), cs, font=f, fill=(214, 52, 44, 255) if counts[i] else (40, 34, 30, 255))
    return np.asarray(base).copy()


def burst(img, cx, cy, t, n=18, col=(255, 224, 120), r0=60, r1=1100, alpha=0.55):
    """Radial speed-lines behind a reveal (in-place on an RGB frame)."""
    ov = img.copy()
    rot = t * 0.35
    for i in range(n):
        a0 = rot + i * 2 * np.pi / n
        a1 = a0 + np.pi / n * 0.8
        pts = np.array([[cx + r0 * np.cos(a0), cy + r0 * np.sin(a0)], [cx + r1 * np.cos(a0), cy + r1 * np.sin(a0)],
                        [cx + r1 * np.cos(a1), cy + r1 * np.sin(a1)]], np.int32)
        cv2.fillConvexPoly(ov, pts, col, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
