"""Global constants – the single place where video/audio format lives."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
FONTS = ASSETS / "fonts"
MODELS = ASSETS / "models"
CACHE = ROOT / ".cache"

W, H = 720, 1280          # 9:16 vertical video
FPS = 30
SR = 48000                # audio sample rate (AAC)

CREAM = (236, 230, 216)
INK = (28, 22, 20)
TABLE = (139, 108, 94)
TABLE_DARK = (92, 70, 62)
NIGHT = (9, 14, 22)

# Layout zones
CAPTION_Y = 915           # centre of the outlined caption word
TABLE_TOP = 830           # y where the close-up table edge starts
BRAND = {"name": "GGhuman", "tag": "Play vs AI", "url": "gghuman.ai"}

def snap(t: float) -> float:
    """Snap a time to the video frame grid so sound & picture share a frame."""
    return round(t * FPS) / FPS
