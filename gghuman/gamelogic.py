"""Game state is computed / verified by CODE, never trusted from a language model:
7-card poker evaluation, card de-duplication, Catan score arithmetic."""
from collections import Counter
from itertools import combinations

RANKS = "23456789TJQKA"
RW = {"A": "aces", "K": "kings", "Q": "queens", "J": "jacks", "T": "tens", "9": "nines", "8": "eights", "7": "sevens", "6": "sixes", "5": "fives", "4": "fours", "3": "threes", "2": "twos"}
CAT = ["High card", "Pair", "Two pair", "Three of a kind", "Straight", "Flush", "Full house", "Four of a kind", "Straight flush"]


def _eval5(c):
    ranks = sorted((RANKS.index(x[0]) + 2 for x in c), reverse=True)
    flush = len({x[1] for x in c}) == 1
    uniq = sorted(set(ranks), reverse=True)
    sh = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            sh = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:
            sh = 5
    g = sorted(Counter(ranks).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    if sh and flush: return (8, (sh,))
    if g[0][1] == 4: return (7, (g[0][0], g[1][0]))
    if g[0][1] == 3 and g[1][1] == 2: return (6, (g[0][0], g[1][0]))
    if flush: return (5, tuple(ranks))
    if sh: return (4, (sh,))
    if g[0][1] == 3: return (3, (g[0][0],) + tuple(r for r in ranks if r != g[0][0]))
    if g[0][1] == 2 and g[1][1] == 2: return (2, (g[0][0], g[1][0], g[2][0]))
    if g[0][1] == 2: return (1, (g[0][0],) + tuple(r for r in ranks if r != g[0][0]))
    return (0, tuple(ranks))


def evaluate(cards):
    """best 5 of up to 7 cards ('As','Td',...) -> (category, tiebreak)"""
    return max(_eval5(c) for c in combinations(cards, 5))


def hand_name(cards):
    cat, tb = evaluate(cards)
    rk = lambda v: RANKS[v - 2]
    if cat == 8: return "Royal flush." if tb[0] == 14 else "Straight flush."
    if cat == 7: return f"Four {RW[rk(tb[0])]}."
    if cat == 6: return "Full house."
    if cat == 5: return "A flush."
    if cat == 4: return "A straight."
    if cat == 3: return f"Three {RW[rk(tb[0])]}."
    if cat == 2: return "Two pair."
    if cat == 1: return f"Pair of {RW[rk(tb[0])]}."
    return "High card."


def _valid(c):
    return isinstance(c, str) and len(c) == 2 and c[0] in RANKS and c[1] in "shdc"


STORIES = {"sandbag", "bluff_lie", "truth", "hidden_vp"}


def check_story(spec, board, ai):
    """The joke only works if what the AI SAYS is really a lie (or, for 'truth', really true).  Verified by code."""
    story, errs = spec.get("story"), []
    if story is None:
        return errs
    if story not in STORIES:
        return [f"story must be one of {sorted(STORIES)}"]
    cat = evaluate(board + ai)[0]
    if story == "sandbag" and cat < 3:
        errs.append(f"story 'sandbag': the AI says it has nothing, so it must really hold three of a kind or better (it has {hand_name(board + ai)})")
    if story == "bluff_lie" and cat < 5:
        errs.append(f"story 'bluff_lie': the AI says it is bluffing, so it must really hold a flush or better (it has {hand_name(board + ai)})")
    if story == "truth":
        want = (spec.get("claim") or {}).get("pocket")
        if not (isinstance(want, str) and len(want) == 1 and want in RANKS):
            errs.append("story 'truth' needs claim.pocket = the pocket-pair rank the AI announces, e.g. 'A'")
        elif not (ai[0][0] == want and ai[1][0] == want):
            errs.append(f"story 'truth': the AI announces pocket {RW[want]} but holds {ai}")
        elif any(c[0] == want for c in board):
            errs.append("story 'truth': the board must not contain the announced rank (the claim should be exactly pocket cards)")
    return errs


ROUND_STORIES = {"bluff_trust", "nothing_trust", "catan_behind"}


def check_rounds(spec):
    """'rounds' series: three hands/turns, rule of three.  The AI makes the SAME kind of claim every round; in rounds 1-2 the claim is honest and the AI
    loses, in round 3 it is a lie and the AI wins big.  Everything is verified by code (hand ranking, scores) and the shots must match the round data."""
    errs, story, game, rounds = [], spec.get("story"), spec.get("game"), spec.get("rounds")
    if story not in ROUND_STORIES:
        return [f"series 'rounds' needs story in {sorted(ROUND_STORIES)}"]
    if not isinstance(rounds, list) or len(rounds) != 3:
        return ["rounds must be a list of exactly 3 rounds"]
    if (game == "catan") != (story == "catan_behind"):
        errs.append("story 'catan_behind' needs game 'catan' (and the poker stories need game 'poker')")
        return errs
    if game == "poker":
        for i, r in enumerate(rounds):
            cards = list(r.get("board", [])) + list(r.get("wren", [])) + list(r.get("ai", []))
            if len(r.get("board", [])) != 5 or len(r.get("wren", [])) != 2 or len(r.get("ai", [])) != 2 or not all(_valid(c) for c in cards):
                errs.append(f"round {i + 1}: needs a 5-card board and two 2-card hands, all valid cards like 'As','Td'")
                continue
            if len(set(cards)) != len(cards):
                errs.append(f"round {i + 1}: duplicate card in {cards}")
                continue
            a, y = evaluate(r["board"] + r["ai"]), evaluate(r["board"] + r["wren"])
            real = "ai" if a > y else "wren" if y > a else "tie"
            if r.get("winner") != real:
                errs.append(f"round {i + 1}: winner is recorded as {r.get('winner')!r} but the cards say {real!r}")
            if i < 2:
                if real != "wren":
                    errs.append(f"round {i + 1}: the AI's claim is honest here, so the AI must lose this round")
                if a[0] != 0:
                    errs.append(f"round {i + 1}: the AI claims it has nothing/is bluffing, so it must really hold only a high card (it has {hand_name(r['board'] + r['ai'])})")
            else:
                need = 5 if story == "bluff_trust" else 3
                if real != "ai":
                    errs.append("round 3: the AI must WIN the last round (the claim is the lie)")
                if a[0] < need:
                    errs.append(f"round 3: the lie must be real: the AI must hold {'a flush' if need == 5 else 'three of a kind'} or better (it has {hand_name(r['board'] + r['ai'])})")
        for s in spec["shots"]:
            pr = s.get("prop") or {}
            if s.get("type") == "insert" and pr.get("hands"):
                i = pr.get("round")
                if not isinstance(i, int) or not 0 <= i < 3:
                    errs.append("showdown insert needs prop.round 0-2")
                    continue
                r = rounds[i]
                got = [h["cards"] for h in pr["hands"]]
                if pr.get("board") != r["board"] or got != [r["wren"], r["ai"]]:
                    errs.append(f"showdown for round {i + 1} does not show that round's cards")
                if [bool(h.get("win")) for h in pr["hands"]] != [r["winner"] == "wren", r["winner"] == "ai"]:
                    errs.append(f"showdown for round {i + 1} highlights the wrong winner")
    else:
        vw = [r.get("vw") for r in rounds]; va = [r.get("va") for r in rounds]; hid = [r.get("hidden", 0) for r in rounds]
        if not all(isinstance(x, int) for x in vw + va + hid):
            return ["catan rounds need integer vw, va, hidden"]
        if len(set(vw)) != 1 or vw[0] >= 10:
            errs.append("catan: your score must stay the same and under 10")
        if not (va[0] < va[1] <= va[2] and va[1] < vw[0]):
            errs.append("catan: AI visible score must rise in round 2 but stay below yours in rounds 1-2")
        if hid[0] != 0 or hid[1] != 0:
            errs.append("catan: hidden VP only appear in round 3")
        if not 1 <= hid[2] <= 5:
            errs.append("catan: hidden VP cards must be 1-5 (the deck only has 5)")
        if va[2] + hid[2] != 10:
            errs.append(f"catan: AI visible {va[2]} + {hid[2]} hidden VP != 10")
        for s in spec["shots"]:
            pr = s.get("prop") or {}
            if s.get("type") == "insert" and pr.get("kind") == "score" and "round" in pr:
                i = pr["round"]
                if not isinstance(i, int) or not 0 <= i < 3:
                    errs.append("score insert prop.round must be 0-2")
                    continue
                if [r[2] for r in pr["rows"]] != [vw[i], va[i]]:
                    errs.append(f"score insert for round {i + 1} does not show {vw[i]} to {va[i]}")
                if i == 2 and pr.get("to") != [vw[2], 10]:
                    errs.append("round 3 score insert must flip to 10")
    return errs


def check_series(spec):
    """Semantic validation for the 'private_view' and 'rounds' series. Returns a list of human-readable errors."""
    if spec.get("series") == "rounds":
        return check_rounds(spec)
    errs = []
    shots = spec["shots"]
    game = spec.get("game")
    if game == "poker":
        table = next((s for s in shots if s["type"] == "table"), {})
        board = table.get("table_cards") or []
        priv = next((s for s in shots if s["type"] == "private"), {})
        ai = priv.get("hand", [])
        you = next((s["prop"].get("hand") for s in shots if s["type"] == "insert" and s["prop"].get("kind") == "poker" and s["prop"].get("hand_label") == "YOU"), None)
        final = [s["prop"] for s in shots if s["type"] == "insert" and s["prop"].get("kind") == "poker" and s["prop"].get("hand_label") != "YOU"]
        cards = board + ai + (you or [])
        if len(board) != 5 or any(c is None or not _valid(c) for c in board):
            errs.append("poker: table.table_cards must be 5 valid cards like 'As','Td'")
        if len(ai) != 2 or not all(_valid(c) for c in ai):
            errs.append("poker: private.hand must be 2 valid cards")
        if not errs:
            if len(set(cards)) != len(cards):
                errs.append(f"poker: duplicate card in {cards}")
            else:
                a = evaluate(board + ai)
                if you:
                    y = evaluate(board + you)
                    if not a > y:
                        errs.append(f"poker: the AI's hand ({hand_name(board + ai)}) must beat yours ({hand_name(board + you)}) – that's the joke")
                elif a[0] < 1:
                    errs.append("poker: AI hand must at least make a pair")
                for f in final:
                    if f.get("hand") != ai or f.get("board") != board:
                        errs.append("poker: final reveal insert must show the same board and AI hand as the table/private shots")
                errs += check_story(spec, board, ai)
    elif game == "catan":
        if spec.get("story") not in (None, "hidden_vp"):
            errs.append("catan scripts only support story 'hidden_vp'")
        priv = next((s for s in shots if s["type"] == "private"), {})
        hidden = sum(1 for c in priv.get("hand", []) if c == "VP")
        sc = next((s["prop"] for s in shots if s["type"] == "insert" and s["prop"].get("kind") == "score" and s["prop"].get("to")), None)
        if sc is None:
            errs.append("catan: need a score insert with 'to' (the final score)")
        else:
            (vw, va), (tw, ta) = [r[2] for r in sc["rows"]], sc["to"]
            if hidden < 1 or hidden > 5: errs.append("catan: hidden VP cards must be 1-5 (the deck only has 5)")
            if va + hidden != ta: errs.append(f"catan: AI score {va} + {hidden} hidden VP != final {ta}")
            if ta < 10: errs.append("catan: AI must reach 10 VP to win")
            if not (vw < 10 and vw > va): errs.append("catan: you must be ahead on visible score (and under 10)")
            if tw != vw: errs.append("catan: your score must not change")
    return errs
