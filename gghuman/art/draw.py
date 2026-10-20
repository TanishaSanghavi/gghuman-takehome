"""Hand-drawn primitives: wobbly outlines, supersampled canvas, 'line boil'."""
import numpy as np
from PIL import Image, ImageDraw

INK = (28, 22, 20, 255)


class Cv:
    """Supersampled RGBA canvas addressed in *design* units (origin-relative)."""

    def __init__(self, x0, y0, x1, y1, s, ss=2):
        self.x0, self.y0, self.s, self.ss = x0, y0, s, ss
        self.k = s * ss
        self.w = max(2, int(round((x1 - x0) * s)))
        self.h = max(2, int(round((y1 - y0) * s)))
        self.img = Image.new("RGBA", (self.w * ss, self.h * ss), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def P(self, x, y):
        return ((x - self.x0) * self.k, (y - self.y0) * self.k)

    def pts(self, ps):
        return [self.P(x, y) for x, y in ps]

    def lw(self, w):
        return max(1, int(round(w * self.k)))

    def line(self, ps, col=INK, w=9, caps=True):
        p = self.pts(ps)
        lw = self.lw(w)
        self.d.line(p, fill=col, width=lw, joint="curve")
        if caps:
            r = lw / 2
            for q in (p[0], p[-1]):
                self.d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=col)

    def poly(self, ps, fill=None, line=INK, w=9):
        p = self.pts(ps)
        if fill is not None:
            self.d.polygon(p, fill=fill)
        if line is not None and w:
            self.line(list(ps) + [ps[0], ps[1]], line, w)

    def circle(self, cx, cy, r, fill=None, line=INK, w=9):
        x, y = self.P(cx, cy)
        rr = r * self.k
        if fill is not None:
            self.d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=fill)
        if line is not None and w:
            self.d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=line, width=self.lw(w))

    def finish(self):
        im = self.img.resize((self.w, self.h), Image.LANCZOS) if self.ss > 1 else self.img
        return np.asarray(im).copy()


def blob(cx, cy, rx, ry, seed, boil=0, amp=0.03, n=84, rot=0.0):
    """Closed wobbly ellipse. `seed` fixes the shape, `boil` jitters it a bit
    (cycling boil 0..2 at ~8fps gives the classic hand-drawn 'line boil')."""
    r0 = np.random.default_rng(seed)
    r1 = np.random.default_rng(seed * 7919 + boil * 104729 + 13)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    off = np.zeros(n)
    for k in (2, 3, 4, 5, 7):
        off += r0.uniform(0.4, 1) * amp / k ** 0.5 * np.sin(k * th + r0.uniform(0, 6.28))
        off += r1.uniform(0.3, 0.9) * amp * 0.5 / k ** 0.5 * np.sin(k * th + r1.uniform(0, 6.28))
    x = rx * (1 + off) * np.cos(th)
    y = ry * (1 + off) * np.sin(th)
    c, s = np.cos(rot), np.sin(rot)
    return [(cx + x[i] * c - y[i] * s, cy + x[i] * s + y[i] * c) for i in range(n)]


def rrect(cx, cy, w, h, r, seed, boil=0, amp=0.012, per=10):
    """Wobbly rounded rectangle polygon."""
    r1 = np.random.default_rng(seed * 31 + boil * 977 + 5)
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    pts = []
    corners = [(x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)]
    for ccx, ccy, a0 in corners:
        for i in range(per + 1):
            a = np.radians(a0 + 90 * i / per)
            pts.append((ccx + r * np.cos(a), ccy + r * np.sin(a)))
    j = r1.normal(0, amp * min(w, h), (len(pts), 2))
    return [(p[0] + j[i, 0], p[1] + j[i, 1]) for i, p in enumerate(pts)]


def wob_line(p0, p1, seed, boil=0, amp=3.0, n=8):
    r = np.random.default_rng(seed * 13 + boil * 131)
    xs = np.linspace(p0[0], p1[0], n)
    ys = np.linspace(p0[1], p1[1], n)
    j = r.normal(0, amp, (n, 2))
    j[0] = j[-1] = 0
    return [(xs[i] + j[i, 0], ys[i] + j[i, 1]) for i in range(n)]


def arc_pts(cx, cy, rx, ry, a0, a1, n=24):
    a = np.radians(np.linspace(a0, a1, n))
    return [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in a]


def shade(rgb, f):
    return tuple(int(max(0, min(255, c * f))) for c in rgb[:3])
