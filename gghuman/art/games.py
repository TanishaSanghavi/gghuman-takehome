"""Hand-drawn game props: playing cards, chips, Catan-style board, score panel, CTA button, cursor."""
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw
from .draw import Cv, blob, rrect, arc_pts, INK
from .text import font

PAPER = (252, 248, 238, 255)
RED = (205, 42, 40, 255)
BLK = (34, 30, 34, 255)
SUITS = {"s": "spade", "h": "heart", "d": "diamond", "c": "club"}
RANKS = {"T": "10"}


def _heart_pts(cx, cy, s, flip=False):
    t = np.linspace(0, 2 * np.pi, 60)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    k = s / 17.0
    return [(cx + a * k, cy + (b * k if flip else -b * k)) for a, b in zip(x, y)]


def draw_suit(c, suit, cx, cy, s, col):
    if suit == "h":
        c.poly(_heart_pts(cx, cy + s * .05, s), col, None, 0)
    elif suit == "s":
        c.poly(_heart_pts(cx, cy - s * .12, s, flip=True), col, None, 0)
        c.poly([(cx - s * .08, cy + s * .25), (cx + s * .08, cy + s * .25), (cx + s * .32, cy + s * .95), (cx - s * .32, cy + s * .95)], col, None, 0)
    elif suit == "d":
        c.poly([(cx, cy - s * 1.0), (cx + s * .72, cy), (cx, cy + s * 1.0), (cx - s * .72, cy)], col, None, 0)
    elif suit == "c":
        for dx, dy in ((0, -.42), (-.46, .2), (.46, .2)):
            c.circle(cx + dx * s, cy + dy * s, s * .42, col, None, 0)
        c.poly([(cx - s * .1, cy + s * .1), (cx + s * .1, cy + s * .1), (cx + s * .3, cy + s * .95), (cx - s * .3, cy + s * .95)], col, None, 0)


@lru_cache(maxsize=128)
def playing_card(code, boil=0):
    """code like 'As', 'Th', '7c' -> 200x280 RGBA card."""
    r, s = code[0], code[1]
    rank = RANKS.get(r, r)
    col = RED if s in "hd" else BLK
    c = Cv(0, 0, 200, 280, 1.0)
    c.poly(rrect(100, 140, 188, 268, 20, ord(r) + ord(s), boil, 0.005), PAPER, INK, 7)
    draw_suit(c, s, 100, 150, 56, col)
    pil = Image.fromarray(c.finish())
    d = ImageDraw.Draw(pil)
    f = font("PatrickHand-Regular.ttf", 52 if len(rank) == 1 else 44)
    d.text((24, 12), rank, font=f, fill=col)
    c2 = Cv(0, 0, 200, 280, 1.0)
    draw_suit(c2, s, 36, 82, 15, col)
    pil.alpha_composite(Image.fromarray(c2.finish()))
    return np.asarray(pil).copy()


@lru_cache(maxsize=8)
def vp_card(boil=0):
    c = Cv(0, 0, 200, 280, 1.0)
    c.poly(rrect(100, 140, 188, 268, 20, 99, boil, 0.005), (250, 240, 214, 255), INK, 7)
    c.poly([(100, 60), (118, 110), (168, 112), (128, 142), (142, 192), (100, 164), (58, 192), (72, 142), (32, 112), (82, 110)], (240, 190, 50, 255), INK, 6)
    pil = Image.fromarray(c.finish())
    d = ImageDraw.Draw(pil)
    f = font("Bangers-Regular.ttf", 72)
    d.text((100 - f.getlength("VP") / 2, 196), "VP", font=f, fill=RED, stroke_width=2, stroke_fill=INK)
    return np.asarray(pil).copy()


def card_face(code, boil=0):
    return vp_card(boil) if code == "VP" else playing_card(code, boil)


@lru_cache(maxsize=16)
def chip_stack(n=6, color=(214, 52, 44), boil=0):
    w, h = 160, 40 * n + 70
    c = Cv(0, 0, w, h, 1.0)
    for i in range(n):
        y = h - 46 - i * 34
        c.poly(blob(80, y, 62, 22, 7 + i, boil, 0.01, 48), color + (255,), INK, 6)
        c.poly(blob(80, y - 10, 62, 22, 7 + i, boil, 0.01, 48), color + (255,), INK, 6)
        c.line([(30, y - 8), (50, y - 8)], (255, 255, 255, 255), 6)
        c.line([(110, y - 8), (130, y - 8)], (255, 255, 255, 255), 6)
    return c.finish()


RES = {"F": ((70, 140, 62), "tree"), "H": ((205, 112, 62), "brick"), "P": ((160, 206, 92), "sheep"),
       "W": ((232, 182, 52), "wheat"), "M": ((150, 156, 168), "mount"), "D": ((226, 206, 152), None)}
LAYOUT = "FPWHMFPDWMHFWPFMHPW"
NUMS = [5, 2, 6, 3, 8, 10, 9, 0, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]


PIECE_COLS = [(36, 120, 220), (214, 52, 44), (236, 140, 40), (250, 250, 244)]


def board_plan(seed=0):
    """Tile layout, number tokens and piece positions for a board.  seed 0 is the original fixed board;
    any other seed gives a different (but valid) layout: one desert, numbers 2-12 as in the base game."""
    layout, nums = list(LAYOUT), list(NUMS)
    pieces = [((-1, -1), (36, 120, 220)), ((1, -2), (214, 52, 44)), ((2, 0), (236, 140, 40)), ((-2, 1), (214, 52, 44)), ((0, 2), (36, 120, 220)), ((1, 1), (236, 140, 40))]
    if seed:
        rr = np.random.default_rng(1000 + int(seed))
        rest = [r for r in layout if r != "D"]
        rr.shuffle(rest)
        tokens = [n for n in nums if n]
        rr.shuffle(tokens)
        layout, nums, ri, ni = [], [], 0, 0
        di = int(rr.integers(0, 19))
        for k in range(19):
            if k == di:
                layout.append("D"); nums.append(0)
            else:
                layout.append(rest[ri]); nums.append(tokens[ni]); ri += 1; ni += 1
        cells = [(q, r) for r in range(-2, 3) for q in range(-2, 3) if abs(q + r) <= 2]
        pick = rr.permutation(len(cells))[:6]
        cols = [PIECE_COLS[int(rr.integers(0, 3))] for _ in range(6)]
        pieces = [(cells[int(i)], c) for i, c in zip(pick, cols)]
    return layout, nums, pieces


@lru_cache(maxsize=16)
def hex_board(boil=0, R=62, seed=0):
    """19-tile Catan-style board (pointy-top hexes) with a few roads/settlements."""
    W, H = int(R * 3.46 * 5.2), int(R * 1.5 * 4 * 2 + R * 2.4)
    c = Cv(0, 0, W, H, 1.0)
    ox, oy = W / 2, H / 2
    rng = np.random.default_rng(4)
    LAYOUT_, NUMS_, pieces = board_plan(seed)
    tiles = []
    k = 0
    for r in range(-2, 3):
        for q in range(-2, 3):
            if abs(q + r) > 2:
                continue
            x = ox + R * 1.732 * (q + r / 2)
            y = oy + R * 1.5 * r
            tiles.append((x, y, LAYOUT_[k], NUMS_[k]))
            k += 1
    for x, y, rt, n in tiles:
        col, icon = RES[rt]
        pts = [(x + (R - 2) * np.cos(np.radians(60 * i - 30)) + rng.normal(0, .8 + boil * .3), y + (R - 2) * np.sin(np.radians(60 * i - 30)) + rng.normal(0, .8)) for i in range(6)]
        c.poly(pts, col + (255,), INK, 4)
        if icon == "tree":
            c.poly([(x, y - 30), (x + 18, y + 4), (x - 18, y + 4)], (36, 96, 44, 255), INK, 3)
        elif icon == "brick":
            c.poly(rrect(x, y - 14, 44, 22, 3, int(x), boil), (170, 70, 40, 255), INK, 3)
        elif icon == "sheep":
            c.poly(blob(x, y - 14, 20, 13, int(y), boil, .02, 32), (250, 250, 244, 255), INK, 3)
        elif icon == "wheat":
            c.circle(x, y - 14, 14, (250, 214, 90, 255), INK, 3)
        elif icon == "mount":
            c.poly([(x, y - 34), (x + 24, y + 2), (x - 24, y + 2)], (112, 118, 130, 255), INK, 3)
        if n:
            c.circle(x, y + 22, 15, (250, 244, 224, 255), INK, 3)
    pil = Image.fromarray(c.finish())
    d = ImageDraw.Draw(pil)
    f = font("PatrickHand-Regular.ttf", 20)
    for x, y, rt, n in tiles:
        if n:
            col = RED[:3] if n in (6, 8) else (40, 34, 30)
            d.text((x - f.getlength(str(n)) / 2, y + 11), str(n), font=f, fill=col + (255,))
    c2 = Cv(0, 0, W, H, 1.0)
    for (q, r), col in pieces:
        x = ox + R * 1.732 * (q + r / 2) + R * .86
        y = oy + R * 1.5 * r - R * .5
        c2.poly([(x - 11, y + 10), (x - 11, y - 4), (x, y - 15), (x + 11, y - 4), (x + 11, y + 10)], col + (255,), INK, 4)
    for (q, r), col in pieces[:4]:
        x = ox + R * 1.732 * (q + r / 2)
        y = oy + R * 1.5 * r
        c2.line([(x - R * .86, y + R * .5), (x, y + R)], INK, 14)
        c2.line([(x - R * .86, y + R * .5), (x, y + R)], col + (255,), 8)
    pil.alpha_composite(Image.fromarray(c2.finish()))
    bb = pil.getchannel('A').point(lambda v: 255 if v > 8 else 0).getbbox()
    return np.asarray(pil.crop(bb)).copy()


@lru_cache(maxsize=32)
def score_panel(rows, title="SCORE", hi=None, boil=0):
    """rows: ((kind, label, value), ...) kind in 'wren','ai:<who>'"""
    from .chars import head_sprite, expr_key
    n = len(rows)
    w, h = 600, 120 + 110 * n
    c = Cv(0, 0, w, h, 1.0)
    c.poly(rrect(w / 2, 50, 300, 78, 10, 1, boil, 0.01), PAPER, INK, 7)
    for i in range(n):
        y = 130 + i * 110
        fill = (255, 226, 160, 255) if hi == i else PAPER
        c.poly(rrect(w / 2, y + 45, 570, 92, 14, 2 + i, boil, 0.008), fill, INK, 7)
    pil = Image.fromarray(c.finish())
    d = ImageDraw.Draw(pil)
    f = font("PatrickHand-Regular.ttf", 52)
    d.text((w / 2 - f.getlength(title) / 2, 20), title, font=f, fill=(40, 34, 30, 255))
    for i, (kind, label, val) in enumerate(rows):
        y = 130 + i * 110
        who = "wren" if kind == "wren" else kind.split(":")[1]
        hs, hx, hy = head_sprite(who, expr_key("neutral"), 0.115, 0)
        pil.alpha_composite(Image.fromarray(hs), (int(66 - hx), int(y + 76 - hy + 14)))
        d = ImageDraw.Draw(pil)
        d.text((118, y + 24), label, font=f, fill=(40, 34, 30, 255))
        vs = str(val)
        d.text((545 - f.getlength(vs), y + 24), vs, font=f, fill=(205, 42, 40, 255) if hi == i else (40, 34, 30, 255))
    return np.asarray(pil).copy()


@lru_cache(maxsize=8)
def cta_button(text="gghuman.ai", pressed=False, boil=0):
    w, h = 470, 130
    c = Cv(0, 0, w, h, 1.0)
    c.poly(rrect(w / 2 + 4, h / 2 + 6, 440, 100, 20, 12, boil, 0.006), (60, 50, 40, 255), None, 0)
    c.poly(rrect(w / 2, h / 2, 440, 100, 20, 12, boil, 0.006), (255, 214, 64, 255) if not pressed else (240, 190, 40, 255), INK, 8)
    pil = Image.fromarray(c.finish())
    d = ImageDraw.Draw(pil)
    f = font("Nunito.ttf", 66, 800)
    d.text((w / 2 - f.getlength(text) / 2, h / 2 - 46), text, font=f, fill=(24, 20, 18, 255))
    return np.asarray(pil).copy()


@lru_cache(maxsize=2)
def cursor():
    c = Cv(0, 0, 90, 110, 1.0)
    pts = [(8, 6), (8, 84), (28, 66), (42, 98), (56, 92), (42, 62), (68, 62)]
    c.poly(pts, (255, 255, 255, 255), INK, 7)
    return c.finish()


@lru_cache(maxsize=2)
def vs_circle():
    c = Cv(0, 0, 120, 120, 1.0)
    c.circle(60, 60, 50, PAPER, INK, 7)
    pil = Image.fromarray(c.finish())
    d = ImageDraw.Draw(pil)
    f = font("PatrickHand-Regular.ttf", 50)
    d.text((60 - f.getlength("VS") / 2, 30), "VS", font=f, fill=(40, 34, 30, 255))
    return np.asarray(pil).copy()


@lru_cache(maxsize=4)
def wordmark(size=58):
    f1 = font("Nunito.ttf", size, 900)
    gw = f1.getlength("GG")
    tw = f1.getlength("GGhuman")
    im = Image.new("RGBA", (int(tw) + 20, int(size * 1.4)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((10, 0), "GG", font=f1, fill=(12, 12, 12, 255))
    d.text((10 + gw, 0), "human", font=f1, fill=(70, 70, 70, 255))
    return np.asarray(im).copy()
