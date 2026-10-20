import json
import numpy as np
from gghuman.spec import validate
from gghuman.timeline import compile_episode
from gghuman import tts
from gghuman.config import SR, FPS
from conftest import ROOT


def spec(name="ep01_poker_bluff_trust"):
    return json.loads((ROOT / "episodes" / f"{name}.json").read_text(encoding="utf-8"))


def fake_tts(text, who, style=None, speed=None, spec_voices=None):
    n = len(text.split())
    dur = 0.35 * n + 0.1
    t = np.arange(int(dur * SR)) / SR
    wav = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    words = [(w, 0.35 * i, 0.35 * i + 0.3) for i, w in enumerate(text.split())]
    return tts.Clip(wav, dur, words, np.ones(int(dur / 0.01) + 1, np.float32), "fake", text)


def test_validator_messages_are_actionable():
    d = spec(); d["cast"] = ["wren", "bob"]
    assert any("unknown cast member 'bob'" in e for e in validate(d))
    d = spec(); d["shots"][0]["lines"][0]["text"] = "word " * 20
    assert any("too long" in e for e in validate(d))
    d = spec(); next(x for x in d["shots"] if x["id"] == "r0_bet")["prop"]["label"] = "WAY TOO LONG A LABEL"
    assert any("label" in e for e in validate(d))
    d = spec(); d["shots"][1]["tag"] = "THIS TAG IS FAR TOO LONG"
    assert any("tag" in e for e in validate(d))
    d = spec(); d["shots"][1]["tally"] = {"rows": [["wren", "YOU"]]}
    assert any("tally" in e for e in validate(d))
    d = spec(); next(x for x in d["shots"] if x["id"] == "r0_bet")["prop"]["stacks"] = [1, 2, 99]
    assert any("stacks" in e for e in validate(d))
    d = spec(); d["board_seed"] = -3
    assert any("board_seed" in e for e in validate(d))
    d = spec(); d["shots"][-2]["react"] = [dict(at=0.5, text="", dur=1)]
    assert any("react" in e for e in validate(d))


def test_timeline_is_frame_snapped_and_monotonic():
    tl = compile_episode(spec(), tts_fn=fake_tts)
    ts = [s["t0"] for s in tl.shots]
    assert ts == sorted(ts)
    for s in tl.shots:
        assert abs(s["t0"] * FPS - round(s["t0"] * FPS)) < 1e-6 and abs(s["t1"] * FPS - round(s["t1"] * FPS)) < 1e-6
    assert tl.data["meta"]["tts_backends"] == ["fake"]
    assert 18 < tl.duration < 30


def test_silence_window_before_scream_keeps_sfx_out():
    tl = compile_episode(spec(), tts_fn=fake_tts)
    scream = next(s for s in tl.shots if s["type"] == "scream")
    for a, b in tl.data["mute"]:
        assert not [e for e in tl.data["sfx"] if a <= e["t"] < b - 1e-6]
    assert any(e["name"] == "scream" and abs(e["t"] - scream["t0"]) < 1e-6 for e in tl.data["sfx"])


def test_every_spoken_word_has_a_caption():
    tl = compile_episode(spec("ep02_catan_behind"), tts_fn=fake_tts)
    caps = " ".join(c["text"].lower() for c in tl.data["captions"])
    for s in tl.shots:
        for l in s["lines"]:
            for w in l["text"].lower().replace(".", "").replace(",", "").split():
                assert w in caps
