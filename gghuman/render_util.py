"""Fast compositing helpers (premultiplied RGBA sprites -> RGB frame)."""
import numpy as np, cv2
from functools import lru_cache


def prem(rgba):
    a = rgba[..., 3:4].astype(np.uint16)
    out = rgba.copy()
    out[..., :3] = ((rgba[..., :3].astype(np.uint16) * a + 127) // 255).astype(np.uint8)
    return out


def blit(dst, pm, x, y, opacity=1.0):
    h, w = pm.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = pm[y0 - y:y1 - y, x0 - x:x1 - x]
    if opacity < 1.0:
        s = cv2.convertScaleAbs(s, alpha=opacity)
    d = dst[y0:y1, x0:x1]
    inv = 255 - s[..., 3]
    inv3 = cv2.merge((inv, inv, inv))
    tmp = cv2.multiply(np.ascontiguousarray(d), inv3, scale=1 / 255.0)
    d[:] = cv2.add(tmp, np.ascontiguousarray(s[..., :3]))


def place(dst, pm, ax, ay, X, Y, sx=1.0, sy=None, rot=0.0, opacity=1.0):
    """Draw premultiplied sprite with its anchor (ax,ay) at (X,Y), scaled/rotated about the anchor."""
    sy = sx if sy is None else sy
    if abs(sx - 1) < 1e-3 and abs(sy - 1) < 1e-3 and abs(rot) < 1e-4:
        blit(dst, pm, int(round(X - ax)), int(round(Y - ay)), opacity)
        return
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    L = np.array([[c * sx, -s * sy], [s * sx, c * sy]])
    t = np.array([X, Y]) - L @ np.array([ax, ay])
    h, w = pm.shape[:2]
    cs = np.array([[0, 0], [w, 0], [0, h], [w, h]]).T
    tc = L @ cs + t[:, None]
    x0, y0 = int(np.floor(tc[0].min())) - 1, int(np.floor(tc[1].min())) - 1
    x1, y1 = int(np.ceil(tc[0].max())) + 1, int(np.ceil(tc[1].max())) + 1
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(dst.shape[1], x1), min(dst.shape[0], y1)
    if x1c <= x0c or y1c <= y0c:
        return
    M = np.hstack([L, (t - np.array([x0c, y0c]))[:, None]]).astype(np.float32)
    interp = cv2.INTER_AREA if (sx < 0.6 and abs(rot) < 1e-4) else cv2.INTER_LINEAR
    if interp == cv2.INTER_AREA:  # resize first for clean downscale
        nw, nh = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
        r = cv2.resize(pm, (nw, nh), interpolation=cv2.INTER_AREA)
        blit(dst, r, int(round(X - ax * sx)), int(round(Y - ay * sy)), opacity)
        return
    out = cv2.warpAffine(pm, M, (x1c - x0c, y1c - y0c), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    blit(dst, out, x0c, y0c, opacity)


def tint(pm, mul):
    out = pm.copy()
    out[..., :3] = np.clip(pm[..., :3].astype(np.float32) * np.array(mul, np.float32), 0, 255).astype(np.uint8)
    return out


@lru_cache(maxsize=8)
def vignette(h, w, strength=0.55, cy=0.5):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h * cy) / (h / 2)) ** 2
    return (1 - strength * np.clip(r / 1.6, 0, 1))[..., None]


@lru_cache(maxsize=4)
def scanlines(h, w):
    m = np.ones((h, w, 1), np.float32)
    m[::4] = 0.82
    m[1::4] = 0.93
    return m


def ease_out_back(x, k=1.9):
    x = min(1.0, max(0.0, x))
    return 1 + (k + 1) * (x - 1) ** 3 + k * (x - 1) ** 2


def ease_out_bounce(x):
    x = min(1.0, max(0.0, x))
    n1, d1 = 7.5625, 2.75
    if x < 1 / d1:
        return n1 * x * x
    if x < 2 / d1:
        x -= 1.5 / d1
        return n1 * x * x + 0.75
    if x < 2.5 / d1:
        x -= 2.25 / d1
        return n1 * x * x + 0.9375
    x -= 2.625 / d1
    return n1 * x * x + 0.984375


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)
