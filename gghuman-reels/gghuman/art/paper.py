"""Cream paper background + table surfaces (cached)."""
from functools import lru_cache
import numpy as np, cv2
from ..config import W, H, CREAM, TABLE, TABLE_DARK, INK


@lru_cache(maxsize=4)
def paper(w=W, h=H):
    rng = np.random.default_rng(3)
    base = np.zeros((h, w, 3), np.float32)
    base[:] = CREAM
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    vig = 1 - 0.07 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h * 0.45) / (h / 2)) ** 2)
    base *= vig[..., None]
    n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.2) * 2.2
    base += n[..., None]
    return np.clip(base, 0, 255).astype(np.uint8)


@lru_cache(maxsize=8)
def table_close(dark=False, w=W, h=H):
    """Bottom-of-frame brown table ."""
    ss = 1
    img = np.zeros((h, w, 4), np.uint8)
    col = tuple(int(c * (0.55 if dark else 1)) for c in TABLE)
    cy, ry, rx = 1620, 800, 900
    cv2.ellipse(img, (w // 2, cy), (rx, ry), 0, 0, 360, col + (255,), -1, cv2.LINE_AA)
    cv2.ellipse(img, (w // 2, cy), (rx, ry), 0, 200, 340, INK + (255,), 9, cv2.LINE_AA)
    rng = np.random.default_rng(5)
    n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.0) * 2.0
    img[..., :3] = np.clip(img[..., :3].astype(np.float32) + n[..., None] * (img[..., 3:] > 0), 0, 255).astype(np.uint8)
    return img


def table_wide(cx=360, cy=790, rx=335, ry=150, dark=False, scale=1.0, color=None):
    img = np.zeros((H, W, 4), np.uint8)
    col = tuple(int(c * (0.5 if dark else 1)) for c in (color or TABLE))
    rx, ry = int(rx * scale), int(ry * scale)
    cv2.ellipse(img, (cx, cy + 14), (rx, ry), 0, 0, 360, tuple(int(c * 0.7) for c in col) + (255,), -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, col + (255,), -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, INK + (255,), 7, cv2.LINE_AA)
    return img
