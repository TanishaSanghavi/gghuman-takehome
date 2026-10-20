import json
import pytest
from gghuman import scriptgen as sg
from gghuman.spec import validate
from gghuman.art import games
from conftest import ROOT


def test_generator_is_deterministic_per_seed():
    a = json.dumps(sg.generate(24, seed=5), sort_keys=True)
    assert a == json.dumps(sg.generate(24, seed=5), sort_keys=True)
    assert a != json.dumps(sg.generate(24, seed=6), sort_keys=True)


def test_100_scripts_are_valid_and_diverse():
    eps = sg.generate(100, seed=7)
    assert len(eps) == 100
    assert all(validate(e) == [] for e in eps)
    d = sg.diversity(eps)
    assert d["passed"], d
    assert d["distinct_hooks"] >= 80 and d["distinct_structures"] >= 8 and d["distinct_catan_boards"] >= 20
    assert set(d["stories"]) == set(sg.KINDS)


def test_every_generated_video_is_a_rule_of_three_with_a_real_lie():
    for e in sg.generate(30, seed=2):
        assert e["series"] == "rounds" and len(e["rounds"]) == 3
        assert [s["tag"][:4] for s in e["shots"] if s["id"].endswith("_claim")] in (["HAND"] * 3, ["TURN"] * 3)
        assert any(s["type"] == "scream" for s in e["shots"]) and e["shots"][-1]["type"] == "endcard"
        assert 13 <= len(e["shots"]) <= 18                     # density: many scene changes, not 9


def test_no_single_word_lines_except_none():
    for e in sg.generate(40, seed=4):
        for s in e["shots"]:
            for l in s.get("lines", []):
                assert len(l["text"].split()) >= 2, l


def test_diversity_gate_rejects_a_repetitive_batch():
    one = sg.generate(1, seed=1)[0]
    batch = [json.loads(json.dumps(one)) for _ in range(30)]
    d = sg.diversity(batch)
    assert not d["passed"] and d["distinct_hooks"] == 1


def test_generate_many_raises_on_repetitive_batch(tmp_path, monkeypatch):
    one = sg.generate(1, seed=1)[0]
    monkeypatch.setattr(sg, "generate", lambda c, s=1: [json.loads(json.dumps(one)) for _ in range(c)])
    with pytest.raises(RuntimeError, match="repetitive"):
        sg.generate_many(30, 1, tmp_path)


def test_catan_board_seeds():
    base_layout, base_nums, _ = games.board_plan(0)
    assert base_layout == list(games.LAYOUT) and base_nums == list(games.NUMS)      # seed 0 = the original board
    seen = set()
    for seed in range(1, 30):
        layout, nums, pieces = games.board_plan(seed)
        assert layout.count("D") == 1 and sorted(layout) == sorted(games.LAYOUT)     # same tile mix, one desert
        assert sorted(n for n in nums if n) == sorted(n for n in games.NUMS if n)    # same number tokens
        assert nums[layout.index("D")] == 0 and len(pieces) == 6
        seen.add((tuple(layout), tuple(nums)))
    assert len(seen) >= 28


# ------------------------------------------------------------------ briefs
def good(**kw):
    return dict(sg.EXAMPLE_BRIEF, **kw)


def test_brief_validation_messages_are_actionable():
    assert sg.validate_brief(good()) == []
    assert any("story" in e for e in sg.validate_brief(good(story="chess")))
    assert any("opener" in e for e in sg.validate_brief(good(opener="Hi.")))                       # single-word lines are rejected
    assert any("claims" in e for e in sg.validate_brief(good(claims=["I'm bluffing.", "Again.", "Bluff."])))
    assert any("at least 2 of the 3 claims" in e for e in sg.validate_brief(good(claims=["Nice weather.", "Still nice.", "Very bluff."])))
    assert any("quips" in e for e in sg.validate_brief(good(quips=[None, None])))
    assert any("hook" in e for e in sg.validate_brief(good(hook="Short")))


def test_episode_from_brief_is_deterministic_and_seeded():
    a = sg.episode_from_brief(good(seed=3)); b = sg.episode_from_brief(good(seed=3)); c = sg.episode_from_brief(good(seed=4))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) and a["rounds"] != c["rounds"]
    with pytest.raises(ValueError, match="invalid brief"):
        sg.episode_from_brief(good(story="nope"))


# ------------------------------------------------------------------ LLM stage, with an injected fake model (no network)
def test_llm_loop_feeds_validator_errors_back_until_valid():
    bad = good(claims=["I'm bluffing.", "Bluffing again.", "Bluff."])                          # last claim is one word
    replies, seen = [json.dumps(bad), "sorry, here you go: not json", json.dumps(good(hook="The AI bluffs and you fall for it"))], []

    def fake(system, msgs):
        seen.append(msgs[-1]["content"])
        return replies[len(seen) - 1]
    ep = sg.llm_generate("any idea", call=fake)
    assert ep["series"] == "rounds" and ep["hook"]["text"].startswith("The AI bluffs") and len(seen) == 3
    assert "claims must be exactly 3" in seen[1] and "not valid JSON" in seen[2]               # the errors were sent back to the model


def test_llm_gives_up_after_max_tries():
    with pytest.raises(RuntimeError, match="still invalid"):
        sg.llm_generate("x", call=lambda s, m: "nope", max_tries=2)


def test_llm_novelty_rejects_reused_hook_then_accepts_new_one():
    replies = iter([json.dumps(good(hook="The AI keeps saying it's bluffing")), json.dumps(good(hook="The AI cries wolf at the poker table"))])
    ep = sg.llm_generate("x", call=lambda s, m: next(replies), used_hooks=["the AI keeps saying its bluffing!"])
    assert ep["hook"]["text"].endswith("poker table")


def test_llm_batch_isolates_failures_and_logs(tmp_path):
    def fake(system, msgs):
        return "garbage" if "BROKEN" in msgs[0]["content"] else json.dumps(good(hook="The AI cries wolf at the poker table"))
    written, failed = sg.llm_batch(["idea one", "BROKEN idea"], tmp_path, call=fake, max_tries=2)
    assert len(written) == 1 and len(failed) == 1
    log = [json.loads(l) for l in (tmp_path / "llm_log.jsonl").read_text().splitlines()]
    assert [l["ok"] for l in log] == [True, False]
