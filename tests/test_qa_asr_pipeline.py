import json, os
import numpy as np
import pytest
from gghuman import qa, asr, pipeline
from gghuman.config import SR
from conftest import ROOT


# ------------------------------------------------------------------ ASR similarity
def test_similarity_tolerates_near_misses_but_not_garbage():
    assert asr.similarity("Badly.", "Bad lady") >= 0.75
    assert asr.similarity("Easy win.", "Easy when") >= 0.75
    assert asr.similarity("Eight to five.", "8 to 5") == 1.0                    # digits vs words
    assert asr.similarity("I have nothing.", "[Music]") < 0.3
    assert asr.similarity("Pocket aces.", "(wind blowing)") < 0.3
    assert asr.similarity("Nine versus six.", "9 versus 6.") == 1.0
    assert asr.similarity("Nine six", "96") == 1.0                              # Whisper writes digit runs as one number
    assert asr.similarity("Call.", "") == 0.0


needs_asr = pytest.mark.skipif(not asr.available(), reason="run setup_assets.py")


@needs_asr
def test_asr_check_rejects_babble_voice_and_accepts_neural_voice():
    """Negative + positive control on the real models: the babble fallback must FAIL, Kokoro must PASS."""
    from gghuman import tts
    babble, neural = [], []
    for be_name, sink in (("blip", babble), ("kokoro", neural)):
        old = os.environ.get("GGH_TTS"); os.environ["GGH_TTS"] = be_name
        tts._active = None
        try:
            if be_name == "kokoro" and tts.get_backend().name != "kokoro":
                pytest.skip("kokoro model missing")
            for txt in ("I have nothing.", "Easy money."):
                c = tts.render_line(txt, "wren", cache=False)
                x = np.concatenate([np.zeros(SR // 4, np.float32), c.wav.astype(np.float32), np.zeros(SR // 4, np.float32)])
                sink.append(asr.similarity(txt, asr.transcribe(asr._resample16(x, SR))))
        finally:
            tts._active = None
            os.environ.pop("GGH_TTS", None) if old is None else os.environ.__setitem__("GGH_TTS", old)
    assert max(babble) < 0.5 and min(neural) >= 0.85, (babble, neural)


# ------------------------------------------------------------------ QA gates
def test_babble_voice_is_never_ok():
    tl = {"meta": {"tts_backends": ["blip"]}}
    assert qa.check_voice_backend(tl)[0]["status"] == "fail"
    assert qa.check_voice_backend({"meta": {"tts_backends": ["kokoro", "blip"]}})[0]["status"] == "fail"       # one bad line is enough
    assert qa.check_voice_backend({"meta": {"tts_backends": ["kokoro"]}})[0]["status"] == "pass"
    assert qa.check_voice_backend(tl, allow_fallback=True)[0]["status"] == "pass"


def test_missing_asr_model_warns_or_fails_in_strict_mode(monkeypatch):
    monkeypatch.setattr(asr, "available", lambda: False)
    assert qa.check_speech_asr(np.zeros(10, np.float32), {}, strict=False)[0]["status"] == "warn"
    assert qa.check_speech_asr(np.zeros(10, np.float32), {}, strict=True)[0]["status"] == "fail"


def test_first_line_check_can_actually_fail():
    tl = {"captions": [], "shots": [{"type": "table", "lines": []}], "speech": [[2.5, 3.0, "gpt", "s0l0"]], "meta": {"hook": {"text": "x"}}}
    c = {x["name"]: x for x in qa.check_script(tl)}
    assert c["first_line_within_1s"]["status"] == "fail"      # the old check returned pass for every non-'wide' first shot


# ------------------------------------------------------------------ pipeline behaviour
def test_invalid_spec_is_rejected_before_any_rendering(tmp_path):
    d = json.loads((ROOT / "episodes" / "ep01_poker_bluff_trust.json").read_text())
    d["rounds"][0]["board"][0] = d["rounds"][0]["wren"][0]
    r = pipeline.build_episode(d, tmp_path)
    assert r["status"] == "invalid_spec" and any("duplicate" in e for e in r["errors"])
    assert not (tmp_path / d["id"] / "video_silent.mp4").exists()


def test_crash_is_retried_once_then_reported_not_raised(tmp_path, monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append(1)
        return dict(id="x", status="error", error="RuntimeError: ffmpeg died")
    monkeypatch.setattr(pipeline, "_build_once", boom)
    r = pipeline.build_episode({"id": "x"}, tmp_path, retries=1)
    assert r["status"] == "error" and len(calls) == 2 and r["build_tries"] == 2


def test_validation_failure_is_not_retried(tmp_path, monkeypatch):
    calls = []
    real = pipeline._build_once

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(pipeline, "_build_once", counting)
    pipeline.build_episode({"id": "bad", "cast": [], "hook": {}, "shots": []}, tmp_path, retries=3)
    assert len(calls) == 1


def test_cache_key_changes_with_pipeline_version(monkeypatch):
    s = json.loads((ROOT / "episodes" / "ep01_poker_bluff_trust.json").read_text())
    h1 = pipeline.spec_hash(s)
    monkeypatch.setattr(pipeline, "PIPELINE_VERSION", "999")
    assert pipeline.spec_hash(s) != h1


def test_publish_queue_contains_only_verified_videos(tmp_path):
    import run
    for i, st in (("a", "ok"), ("b", "degraded"), ("c", "qa_failed"), ("d", "error")):
        (tmp_path / i).mkdir()
        (tmp_path / i / "caption.txt").write_text("cap " + i)
    res = [dict(id=i, status=st, duration=20.0) for i, st in (("a", "ok"), ("b", "degraded"), ("c", "qa_failed"), ("d", "error"))]
    assert run.write_manifest(res, tmp_path) == 1
    rows = (tmp_path / "publish_queue.csv").read_text().splitlines()
    assert len(rows) == 2 and rows[1].startswith("a,")


# ------------------------------------------------------------------ integration (real render, ~30-90 s each): pytest --runslow
MINI = {"id": "mini", "cast": ["wren", "gpt"], "hook": {"pov": "POV:", "text": "The AI says hello"},
        "shots": [{"id": "table", "type": "table", "pair": ["wren", "gpt"], "game": "poker", "table_cards": ["Ad", "7c", "2h", "9s", "4d"], "dur": 2.3,
                   "lines": [{"who": "gpt", "text": "I have nothing.", "expr": "plead"}]},
                  {"id": "scream", "type": "scream", "who": "wren", "dur": 1.7}]}


@pytest.mark.slow
def test_mini_video_builds_with_neural_voice_and_is_marked_ok(tmp_path):
    r = pipeline.build_episode(MINI, tmp_path)
    assert r["status"] in ("ok", "qa_failed"), r
    assert r["tts_backends"] == ["kokoro"]
    q = json.loads((tmp_path / "mini" / "qa.json").read_text())
    names = {c["name"]: c["status"] for c in q["checks"]}
    assert names["neural_voice_used"] == "pass" and names["voice_onset_vs_timeline_ms"] == "pass" and names["speech_intelligible_asr"] == "pass"


@pytest.mark.slow
def test_babble_fallback_build_is_degraded_not_ok(tmp_path, monkeypatch):
    from gghuman import tts
    monkeypatch.setenv("GGH_TTS", "blip"); tts._active = None
    try:
        r = pipeline.build_episode(MINI, tmp_path)
        assert r["status"] == "degraded" and "neural_voice_used" in r["qa"]["failed"]
        r2 = pipeline.build_episode(MINI, tmp_path / "b", allow_fallback_voice=True, asr=True, force=True)
        assert "speech_intelligible_asr" in r2["qa"]["failed"]        # even if the voice is allowed, the ASR check still rejects babble
    finally:
        tts._active = None
