"""Typography: handwritten POV hook, outlined caption words, chunky reaction words."""
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..config import FONTS, W

def font(name, size, var=None):
    f = ImageFont.truetype(str(FONTS / name), size)
    if var is not None:
        try:
            f.set_variation_by_axes([var])
        except Exception:
            pass
    return f

CAPTION_FONT = "PatrickHand-Regular.ttf"
REACT_FONT = "Bangers-Regular.ttf"
HOOK_FONT = "PatrickHand-Regular.ttf"

def _fit(text, fname, size, max_w, stroke=0):
    while size > 20:
        f = font(fname, size)
        l, t, r, b = f.getbbox(text, stroke_width=stroke)
        if r - l <= max_w:
            return f, (l, t, r, b)
        size -= 3
    return f, f.getbbox(text, stroke_width=stroke)

@lru_cache(maxsize=512)
def outlined(text, size=78, fill=(255, 255, 255), stroke_col=(20, 16, 14), stroke=8, max_w=640, fname=CAPTION_FONT):
    """White fill / thick black outline caption word (page style)."""
    f, bb = _fit(text, fname, size, max_w, stroke)
    pad = stroke + 6
    w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((pad - bb[0], pad - bb[1]), text, font=f, fill=fill + (255,), stroke_width=stroke, stroke_fill=stroke_col + (255,))
    return np.asarray(im).copy()

@lru_cache(maxsize=128)
def plain(text, size=56, col=(60, 55, 52), fname=HOOK_FONT, max_w=620, align="center", spacing=8):
    lines = text.split("\n")
    f, _ = _fit(max(lines, key=len), fname, size, max_w)
    boxes = [f.getbbox(t) for t in lines]
    lh = int(f.size * 1.18)
    w = max(b[2] for b in boxes) + 20
    h = lh * len(lines) + 20
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, t in enumerate(lines):
        tw = f.getlength(t)
        x = (w - tw) / 2 if align == "center" else 10
        d.text((x, 6 + i * lh), t, font=f, fill=col + (255,))
    return np.asarray(im).copy()

@lru_cache(maxsize=256)
def chunky(text, size=130, fill=(255, 214, 64), outline=(20, 16, 14), stroke=10, max_w=640, shadow=True):
    """Bangers reaction word with offset shadow."""
    f, bb = _fit(text, REACT_FONT, size, max_w, stroke)
    pad = stroke + 14
    w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if shadow:
        d.text((pad - bb[0] + 7, pad - bb[1] + 9), text, font=f, fill=outline + (255,), stroke_width=stroke, stroke_fill=outline + (255,))
    d.text((pad - bb[0], pad - bb[1]), text, font=f, fill=fill + (255,), stroke_width=stroke, stroke_fill=outline + (255,))
    return np.asarray(im).copy()

@lru_cache(maxsize=8)
def brand():
    """GGhuman / Play vs AI / gghuman.ai footer block (bottom-left)."""
    from ..config import BRAND
    f1 = font("Nunito.ttf", 40, 900); f2 = font("Nunito.ttf", 21, 800)
    im = Image.new("RGBA", (330, 130), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.text((0, 0), "GG", font=f1, fill=(12, 12, 12, 255))
    gw = f1.getlength("GG")
    d.text((gw, 0), "human", font=f1, fill=(70, 70, 70, 255))
    d.text((0, 52), BRAND["tag"], font=f2, fill=(12, 12, 12, 255))
    d.text((0, 80), BRAND["url"], font=f2, fill=(12, 12, 12, 255))
    return np.asarray(im).copy()
