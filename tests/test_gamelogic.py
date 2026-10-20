import copy, json
from gghuman.gamelogic import evaluate, hand_name
from gghuman.spec import validate
from conftest import ROOT

NAMES = ("ep01_poker_bluff_trust", "ep02_catan_behind", "ep03_poker_nothing_trust")


def ep(name):
    return json.loads((ROOT / "episodes" / f"{name}.json").read_text(encoding="utf-8"))


def test_hand_categories_and_names():
    assert hand_name(["As", "Ks", "Qs", "Js", "Ts", "2d", "3c"]) == "Royal flush."
    assert hand_name(["9h", "8h", "7h", "6h", "5h", "2d", "3c"]) == "Straight flush."
    assert hand_name(["Ah", "Ad", "Ac", "Kd", "Ks", "2c", "3c"]) == "Full house."
    assert hand_name(["Ah", "Ad", "Qc", "Kd", "Js", "2c", "3c"]) == "Pair of aces."
    assert hand_name(["2h", "7d", "9c", "Kd", "Js", "4c", "3c"]) == "High card."


def test_wheel_straight_is_a_straight_not_ace_high():
    cat, tb = evaluate(["Ah", "2d", "3c", "4s", "5h", "9d", "Kc"])
    assert cat == 4 and tb == (5,)


def test_best_five_of_seven_and_kicker_ordering():
    assert evaluate(["Ah", "Kd", "Qc", "9s", "4h", "2d", "3c"]) < evaluate(["Ah", "Kd", "Qc", "9s", "5h", "2d", "3c"])
    assert evaluate(["Kh", "Kd", "2c", "7s", "9h", "4d", "3c"]) < evaluate(["Ah", "Ad", "2c", "7s", "9h", "4d", "3c"])


def test_submission_scripts_are_valid():
    for n in NAMES:
        assert validate(ep(n)) == [], n


def test_rounds_data_matches_a_rule_of_three():
    d = ep("ep01_poker_bluff_trust")
    assert [r["winner"] for r in d["rounds"]] == ["wren", "wren", "ai"]
    for r in d["rounds"][:2]:                                  # honest rounds: the AI really has only a high card
        assert evaluate(r["board"] + r["ai"])[0] == 0
    assert evaluate(d["rounds"][2]["board"] + d["rounds"][2]["ai"])[0] >= 5      # bluff_trust: the lie is a flush or better


def test_duplicate_card_in_a_round_rejected():
    d = ep("ep01_poker_bluff_trust"); d["rounds"][1]["board"][0] = d["rounds"][1]["wren"][0]
    assert any("duplicate" in e for e in validate(d))


def test_recorded_winner_must_match_the_cards():
    d = ep("ep01_poker_bluff_trust"); d["rounds"][0]["winner"] = "ai"
    assert any("cards say" in e for e in validate(d))


def test_honest_rounds_require_the_ai_to_really_have_nothing():
    d = ep("ep01_poker_bluff_trust")
    d["rounds"][0]["ai"] = [c for c in ("Ah", "Ad", "As", "Ac") if c not in d["rounds"][0]["board"] + d["rounds"][0]["wren"]][:2]      # AI now has a pair of aces
    errs = validate(d)
    assert any("must" in e and "round 1" in e for e in errs), errs


def test_round_three_lie_must_be_real_and_win():
    d = ep("ep01_poker_bluff_trust")
    d["rounds"][2] = copy.deepcopy(d["rounds"][0])             # round 3 is now an honest loss: nothing to be shocked about
    errs = validate(d)
    assert any("round 3" in e for e in errs), errs
    from gghuman.scriptgen import episode_from_brief
    b = ep("ep03_poker_nothing_trust")["brief"]
    for seed in range(60):                                      # find a deal where the AI wins round 3 with exactly three of a kind
        d = episode_from_brief(dict(b, seed=seed))
        if evaluate(d["rounds"][2]["board"] + d["rounds"][2]["ai"])[0] == 3:
            break
    assert validate(d) == []
    d["story"] = "bluff_trust"                                  # trips are enough for 'nothing', but not for a 'bluff' lie: needs a flush or better
    assert any("flush" in e for e in validate(d))


def test_showdown_shots_must_show_their_rounds_cards():
    d = ep("ep01_poker_bluff_trust")
    sd = next(s for s in d["shots"] if s["id"] == "r2_showdown")
    sd["prop"]["hands"][1]["cards"] = ["2c", "3d"]
    assert any("does not show that round" in e for e in validate(d))
    d = ep("ep01_poker_bluff_trust")
    sd = next(s for s in d["shots"] if s["id"] == "r0_showdown")
    sd["prop"]["hands"][0]["win"], sd["prop"]["hands"][1]["win"] = False, True
    assert any("wrong winner" in e for e in validate(d))


def test_catan_arithmetic_and_rule_of_three():
    d = ep("ep02_catan_behind")
    assert [r["hidden"] for r in d["rounds"]] == [0, 0, d["rounds"][2]["hidden"]]
    assert d["rounds"][2]["va"] + d["rounds"][2]["hidden"] == 10
    d["rounds"][2]["hidden"] = 6
    assert any("1-5" in e or "!= 10" in e for e in validate(d))
    d = ep("ep02_catan_behind"); d["rounds"][1]["va"] = d["rounds"][0]["vw"]           # AI would be level with you in round 2
    assert any("rise in round 2" in e for e in validate(d))
    d = ep("ep02_catan_behind")
    sc = next(s for s in d["shots"] if s["id"] == "r1_score"); sc["prop"]["rows"][1][2] += 1
    assert any("does not show" in e for e in validate(d))
    d = ep("ep02_catan_behind"); d["game"] = "poker"
    assert any("needs game" in e for e in validate(d))
