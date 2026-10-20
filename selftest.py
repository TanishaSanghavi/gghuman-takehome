#!/usr/bin/env python3
"""~1 minute smoke test (no TTS model needed): validates every script, proves bad game state is rejected,
compiles with the babble voice, renders a frame of EVERY shot, mixes audio, and checks the batch generator."""
import os, sys, json, tempfile
os.environ["GGH_TTS"] = "blip"
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gghuman.spec import validate
from gghuman.timeline import compile_episode
from gghuman.render import Renderer
from gghuman.mixer import mix

files = sorted(ROOT.glob("episodes/*.json"))
assert files, "no episodes/*.json"
for f in files:
    spec = json.loads(f.read_text(encoding="utf-8"))
    assert not validate(spec), (f.name, validate(spec))
    tl = compile_episode(spec)
    R = Renderer(tl, spec)
    for s in tl.shots:
        fr = R.frame(int((s["t0"] + 0.4 * (s["t1"] - s["t0"])) * 30))
        assert fr.shape == (1280, 720, 3), s["id"]
    with tempfile.TemporaryDirectory() as d:
        info = mix(tl, Path(d) / "a.wav")
    print(f"ok  {f.parent.name + '/' + f.name:46} {tl.duration:5.1f}s {len(tl.shots):2d} shots  peak {info['peak']:.2f}")

# game logic must reject impossible state (this is what protects LLM-written briefs and hand-edited scripts)
d = json.loads((ROOT / "episodes/ep01_poker_bluff_trust.json").read_text(encoding="utf-8"))
d["rounds"][1]["board"][0] = d["rounds"][1]["wren"][0]
assert any("duplicate" in e for e in validate(d)), "duplicate card not rejected"
d = json.loads((ROOT / "episodes/ep01_poker_bluff_trust.json").read_text(encoding="utf-8"))
d["rounds"][2] = json.loads(json.dumps(d["rounds"][0]))
assert any("round 3" in e for e in validate(d)), "a round-3 'lie' that is not a lie was not rejected"
d = json.loads((ROOT / "episodes/ep02_catan_behind.json").read_text(encoding="utf-8"))
d["rounds"][2]["hidden"] = 6
assert validate(d), "bad Catan arithmetic not rejected"
print("ok  impossible deal / fake lie / wrong score are rejected")
from gghuman.scriptgen import generate_many
with tempfile.TemporaryDirectory() as d:
    assert generate_many(12, 3, Path(d)) == 12      # raises if the batch is repetitive
print("ok  generator: 12 validated scripts")
print("selftest passed")
