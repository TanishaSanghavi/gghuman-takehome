"""Script production for the 'HAND 1 / HAND 2 / HAND 3' series (rule of three).

The format: the AI makes the SAME kind of claim three rounds in a row ("I'm bluffing." / "I have nothing." / "I'm losing.").
Rounds 1 and 2 prove the claim honest (the AI really loses, Wren wins, the pot grows BET -> RAISE).  In round 3 the pot is ALL IN
and the claim turns out to be a lie: the AI wins big.  Wren screams, the gghuman.ai end card plays.

On-screen furniture that makes the series recognisable: a HAND/TURN tag (top-left), a running WINS tally (top-right), an
escalating pot (BET -> RAISE -> ALL IN), a two-hand showdown with the winner outlined, and a YOU WIN / AI WINS stamp.

Everything creative lives in a small BRIEF (hook, opener, three claims, quips, stamp word).  Code turns a brief into a full
episode: it deals the cards / scores (gamelogic.py), builds the shots, and validates the result (including 'is the lie really a
lie?').  So an LLM only writes the funny part, and cannot break the game logic.

* generate():   deterministic, seeded batch of briefs -> episodes, with a diversity gate (repetitive content tends to perform worse).
* llm_generate(): an LLM writes one brief from a one-line idea; the validators send errors back until it is valid.  The model call
                  is injectable, so the whole loop is unit-tested without network.
"""
import hashlib, json, os, random, re, urllib.error, urllib.request
from collections import Counter
from pathlib import Path
from .spec import validate
from .gamelogic import hand_name, evaluate, RANKS, RW, ROUND_STORIES

AIS = ["gpt", "claude", "gemini"]
AI_NAME = {"gpt": "GPT", "claude": "Claude", "gemini": "Gemini"}
KINDS = ["bluff_trust", "nothing_trust", "catan_behind"]
NUM = {i: w for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}
HASHTAGS = "#gghuman #playvsai #aigames"
BET_LABELS = ["BET", "RAISE", "ALL IN"]
STACKS = [(3, 4, 2), (5, 7, 4), (8, 10, 7)]
TENSION_WORDS = ["HMM.", "?!", "UH OH", "WAIT.", "HUH?"]
STAMPS = ["BUSTED", "NO WAY", "WHAT", "NOOO", "HOW", "TRAP"]
TENSION_SETS = [[2], [1, 2], [0, 2], [1]]          # which rounds get a two-character tension beat
QUIP_PATTERNS = [(True, True), (True, False), (False, True)]
CLAIM_WORDS = {"bluff_trust": ("bluff",), "nothing_trust": ("nothing", "junk", "trash", "terrible", "worst", "can't win", "fold"),
               "catan_behind": ("losing", "behind", "hopeless", "catch up", "lose")}

# ------------------------------------------------------------------ wording pools
POOLS = {
    "bluff_trust": dict(
        hooks=["The AI keeps saying it's bluffing", "The AI is honest about bluffing. Twice.", "The AI says it's bluffing three hands in a row",
               "You learn to trust the AI's bluffs", "The AI says it's bluffing again", "The AI always tells you when it's bluffing",
               "You finally figure out the AI's tell", "The AI announces every bluff", "The AI says it's bluffing and you believe it",
               "Every time the AI says bluff, it's a bluff", "You've caught the AI's bluffing habit", "The AI warns you before every bluff"],
        opener=["Let's play.", "Ready when you are.", "Deal me in.", "You go first."],
        claims=[["I'm bluffing.", "Bluffing again.", "Definitely bluffing."], ["Total bluff.", "Another bluff.", "Pure bluff. Promise."],
                ["It's a bluff.", "Still a bluff.", "Big bluff. Trust me."], ["I'm bluffing.", "Same bluff.", "Obvious bluff. Call me."],
                ["Just a bluff.", "Bluffing again.", "Biggest bluff yet."], ["Total bluff.", "Bluffing, obviously.", "Honest bluff. Call."]],
        quips=[["So easy.", "Too easy."], ["Free chips.", "More free chips."], ["Told you.", "Told you again."], ["I read you.", "Read you again."],
               ["Easy money.", "More easy money."], ["I knew it.", "Knew it again."]]),
    "nothing_trust": dict(
        hooks=["The AI keeps saying it has nothing", "The AI says it has nothing three hands in a row", "The AI is honest about having nothing",
               "You trust the AI when it says it has junk", "The AI says it has nothing. Again.", "The AI says it never has a hand",
               "The AI complains about its cards every hand", "The AI keeps moaning about its cards", "You get used to the AI having nothing",
               "The AI says it's out of luck", "The AI sighs about its cards every hand", "The AI says it can't catch a hand"],
        opener=["Deal the cards.", "Let's begin.", "Ready, I guess.", "Here we go."],
        claims=[["I have nothing.", "Still nothing.", "Absolutely nothing."], ["Terrible hand.", "Another terrible hand.", "Worst hand ever."],
                ["I have junk.", "More junk.", "Pure junk. Honestly."], ["I got nothing.", "Nothing again.", "Nothing at all."],
                ["Nothing here.", "Still nothing here.", "Truly nothing here."], ["I can't win this.", "Can't win this either.", "I really can't win."]],
        quips=[["So easy.", "Too easy."], ["Free chips.", "More free chips."], ["Thanks for the gift.", "Another gift."], ["Easy money.", "More easy money."],
               ["I'm on a roll.", "Still rolling."], ["Poor thing.", "Poor thing again."]]),
    "catan_behind": dict(
        hooks=["The AI keeps saying it's losing at Catan", "The AI says it's losing at Catan every turn", "You start believing the AI is losing at Catan",
               "The AI says it's behind. Again.", "The AI is losing at Catan, allegedly", "The AI says it's hopeless at Catan every round",
               "You trust the AI when it says it's behind", "The AI says you're winning at Catan", "The AI keeps conceding at Catan",
               "The AI says it needs a miracle at Catan again", "The AI sounds defeated at Catan", "The AI says it can't win at Catan"],
        opener=["Let's play Catan.", "Ready to play.", "Roll the dice.", "Your turn first."],
        claims=[["I'm losing.", "Still losing.", "Definitely losing."], ["I'm behind.", "Still behind.", "Way behind."], ["I'm way behind.", "Still way behind.", "Hopelessly behind."],
                ["This is hopeless.", "Still hopeless.", "Truly hopeless."], ["I'm losing badly.", "Still losing badly.", "Losing very badly."],
                ["I can't catch up.", "Still can't catch up.", "I'll never catch up."]],
        quips=[["Easy win.", "Too easy."], ["I've got this.", "Still got this."], ["Thanks for playing.", "Thanks again."], ["Easy points.", "More easy points."],
               ["I'm cruising.", "Still cruising."], ["Poor thing.", "Poor thing again."]]),
}
CTA1 = {"poker": ["Think you can beat", "Think you can out-bluff", "Think you can read"], "catan": ["Think you can beat", "Think you can outbuild", "Think you can out-trade"]}
ROAD_R1 = ["You hold longest road.", "You hold biggest army.", "You have two cities."]


# ------------------------------------------------------------------ shot factories
def _line(who, text, expr=None, **kw):
    d = dict(who=who, text=text)
    if expr:
        d["expr"] = expr
    d.update(kw)
    return d


def _close(sid, who, text, expr, anim=None):
    s = dict(id=sid, type="close", who=who, state={who: dict(expr=expr)}, lines=[_line(who, text, expr)])
    if anim:
        s["beats"] = [dict(at="L0.start" if anim == "hop" else 0.0, who=who, set=dict(anim=anim))]
    return s


def _two(sid, pair, states, word, dur=1.3):
    return dict(id=sid, type="two", pair=list(pair), dur=dur, state={w: dict(expr=e) for w, e in states.items()}, lines=[],
                react=[dict(at=0.15, text=word, dur=dur - 0.3, pos=[360, 250], size=150)],
                sfx=[dict(name="heartbeat", at=0.05), dict(name="heartbeat", at=0.62)])


def _scream(word):
    return dict(id="scream", type="scream", who="wren", dur=1.7, react=[dict(at=0.5, text=word, dur=1.0, pos=[360, 250], size=130, rot=-4)])


def _cta(game, ai, line1):
    return dict(id="cta", type="endcard", game=game, ai=ai, line1=line1, trans_in="dip")


def _tally(ai, title, w, to=None):
    t = dict(title=title, rows=[["wren", "YOU", w[0]], [f"ai:{ai}", "AI", w[1]]])
    if to is not None:
        t.update(to=[["wren", "YOU", to[0]], [f"ai:{ai}", "AI", to[1]]], at=1.0)
    return t


# ------------------------------------------------------------------ deals (computed by code, never by a language model)
def _deck():
    return [r + s for r in RANKS for s in "shdc"]


def deal_loss(rng):
    """Honest-claim round: the AI really has only a high card and you win with a pair or better."""
    while True:
        d = _deck(); rng.shuffle(d)
        board, wren, ai = d[:5], d[5:7], d[7:9]
        a, y = evaluate(board + ai), evaluate(board + wren)
        if a[0] == 0 and y[0] >= 1 and y > a:
            return board, wren, ai


def deal_weak(rng):
    """AI holds a pocket pair that hits trips+; you hold a decent but weaker hand.  -> board, ai, wren"""
    while True:
        d = _deck(); rng.shuffle(d)
        r = rng.choice("AKQJT98")
        ai = [r + "s", r + "h"] if rng.random() < .5 else [r + "d", r + "c"]
        rest = [c for c in d if c not in ai]
        wren_r = rng.choice([x for x in "AKQJT" if x != r])
        wren = [c for c in rest if c[0] == wren_r][:2]
        rest = [c for c in rest if c not in wren]
        board = [c for c in rest if c[0] == r][:1] + [c for c in rest if c[0] != r][:4]
        rng.shuffle(board)
        a, y = evaluate(board + ai), evaluate(board + wren)
        if a[0] >= 3 and a > y and y[0] >= 1:
            return board, ai, wren


def deal_honest(rng):
    """AI makes a flush or better with two suited high cards; you hold a pair.  -> board, ai, wren"""
    while True:
        d = _deck(); rng.shuffle(d)
        suit = rng.choice("shdc")
        hi = rng.sample("AKQJT", 2)
        ai = [hi[0] + suit, hi[1] + suit]
        rest = [c for c in d if c not in ai]
        suited = [c for c in rest if c[1] == suit and c[0] not in hi][:3]
        off = [c for c in rest if c[1] != suit][:2]
        board = suited + off
        rng.shuffle(board)
        wr = rng.choice("KQJ")
        wren = [c for c in rest if c[0] == wr and c not in board][:2]
        if len(wren) < 2:
            continue
        a, y = evaluate(board + ai), evaluate(board + wren)
        if a[0] >= 5 and a > y:
            return board, ai, wren


# ------------------------------------------------------------------ brief validation
def validate_brief(b):
    """Errors in plain language: this is what an LLM is shown when its brief is rejected."""
    E = []
    if b.get("story") not in ROUND_STORIES:
        E.append(f"story must be one of {sorted(ROUND_STORIES)}")
        return E
    if b.get("ai") not in AIS:
        E.append(f"ai must be one of {AIS}")
    hook = b.get("hook", "")
    if not isinstance(hook, str) or not 12 <= len(hook) <= 70:
        E.append("hook must be a string of 12-70 characters (the handwritten POV line)")

    def words(t, lo, hi):
        return isinstance(t, str) and lo <= len(t.split()) <= hi

    if not words(b.get("opener"), 2, 6):
        E.append("opener must be 2-6 words (the AI's first line while the POV text is read)")
    claims = b.get("claims")
    if not (isinstance(claims, list) and len(claims) == 3 and all(words(c, 2, 6) for c in claims)):
        E.append("claims must be exactly 3 lines of 2-6 words (the AI's claim in round 1, 2, 3; escalate it)")
    else:
        hit = sum(any(k in c.lower() for k in CLAIM_WORDS[b["story"]]) for c in claims)
        if hit < 2:
            E.append(f"at least 2 of the 3 claims must clearly say what the story is about (one of {CLAIM_WORDS[b['story']]})")
    q = b.get("quips")
    if not (isinstance(q, list) and len(q) == 2 and all(x is None or words(x, 2, 6) for x in q) and any(x is not None for x in q)):
        E.append("quips must be a list of 2 items (Wren's line after round 1 and after round 2): each 2-6 words or null, at least one not null")
    if not (isinstance(b.get("stamp", "NO WAY"), str) and 2 <= len(b.get("stamp", "NO WAY")) <= 12):
        E.append("stamp must be 2-12 characters (the comic word over Wren's scream)")
    if not (isinstance(b.get("tension", "HMM."), str) and 2 <= len(b.get("tension", "HMM.")) <= 8):
        E.append("tension must be 2-8 characters (the comic word in the tension beat)")
    ts = b.get("tension_rounds", [2])
    if not (isinstance(ts, list) and ts and set(ts) <= {0, 1, 2}):
        E.append("tension_rounds must be a non-empty subset of [0,1,2]")
    return E


# ------------------------------------------------------------------ brief -> episode
def _poker_episode(b, rng):
    ai, story = b["ai"], b["story"]
    final = deal_honest(rng) if story == "bluff_trust" else deal_weak(rng)
    deals = [deal_loss(rng), deal_loss(rng), (final[0], final[2], final[1])]
    rounds = []
    for board, wren, aih in deals:
        a, y = evaluate(board + aih), evaluate(board + wren)
        rounds.append(dict(board=board, wren=wren, ai=aih, winner="ai" if a > y else "wren"))
    expr = "calm" if story == "bluff_trust" else "plead"
    shots = [dict(id="table", type="table", pair=["wren", ai], game="poker", table_cards=rounds[0]["board"], dur=2.3, state={ai: dict(expr=expr)},
                  lines=[_line(ai, b["opener"], expr)])]
    wins = [0, 0]
    tension_rounds, quips = b.get("tension_rounds", [2]), b["quips"]
    for r, rd in enumerate(rounds):
        tag = f"HAND {r + 1}"
        pre, post = list(wins), list(wins)
        post[0 if rd["winner"] == "wren" else 1] += 1
        wins = post
        claim = _close(f"r{r}_claim", ai, b["claims"][r], expr)
        if r:
            claim["trans_in"] = "whip"
        bet = dict(id=f"r{r}_bet", type="insert", prop=dict(kind="chips", label=BET_LABELS[r], stacks=list(STACKS[r])), dur=0.8, lines=[],
                   sfx=[dict(name="chips", at=0.1)])
        if r == 2:
            bet["lines"] = [_line("wren", "All in.", "smug")]
        block = [claim, bet]
        if r in tension_rounds:
            block.append(_two(f"r{r}_tension", ["wren", ai], {"wren": "smug", ai: expr}, b.get("tension", "HMM.")))
        won = rd["winner"] == "wren"
        block.append(dict(id=f"r{r}_showdown", type="insert", dur=2.0 if r < 2 else 2.2, pad=0.3,
                          prop=dict(kind="poker", round=r, board=rd["board"], win_at=1.0,
                                    hands=[dict(label="YOU", cards=rd["wren"], win=won), dict(label="THE AI", cards=rd["ai"], win=not won)]),
                          sfx=[dict(name="flip", at=0.3), dict(name="flip", at=0.5)],
                          react=[dict(at=1.0, text="YOU WIN" if won else "AI WINS", dur=0.9, pos=[300, 190], size=96, color=[60, 190, 90] if won else [226, 60, 50])],
                          lines=[_line("narrator", hand_name(rd["board"] + rd["ai"]), None, pre=0.5, speed=0.93)]))
        qi = quips[r] if r < 2 else None
        if qi:
            block.append(_close(f"r{r}_quip", "wren", qi, "smug", anim="hop"))
        for s in block:
            s["tag"] = tag
            if s["id"].endswith("_showdown"):
                s["tally"] = _tally(ai, "WINS", pre, post)
            else:
                s["tally"] = _tally(ai, "WINS", post if s["id"].endswith("_quip") else pre)
        shots += block
    shots += [_scream(b.get("stamp", "NO WAY")), _cta("poker", ai, CTA1["poker"][rng.randrange(3)])]
    return "poker", rounds, shots


def _catan_episode(b, rng):
    ai = b["ai"]
    vw = rng.choice([8, 9])
    va2 = rng.choice([6, 7])
    delta = rng.choice([1, 2])
    va1 = va2 - delta
    hid = 10 - va2
    rounds = [dict(vw=vw, va=va1, hidden=0), dict(vw=vw, va=va2, hidden=0), dict(vw=vw, va=va2, hidden=hid)]
    events = {1: "The AI built a settlement.", 2: "The AI built two settlements."}
    expr = "plead"
    shots = [dict(id="table", type="table", pair=["wren", ai], game="catan", dur=2.3, state={ai: dict(expr=expr)}, lines=[_line(ai, b["opener"], expr)])]
    tension_rounds, quips = b.get("tension_rounds", [2]), b["quips"]
    for r, rd in enumerate(rounds):
        tag = f"TURN {10 * (r + 1)}"
        rows = [["wren", "YOU", vw], [f"ai:{ai}", "THE AI", rd["va"]]]
        claim = _close(f"r{r}_claim", ai, b["claims"][r], expr)
        if r:
            claim["trans_in"] = "whip"
        block = [claim]
        if r < 2:
            line = _pick(ROAD_R1, rng) if r == 0 else events[delta]
            block.append(dict(id=f"r{r}_board", type="insert", prop=dict(kind="board"), trans_in="whip", lines=[_line("narrator", line, None, speed=0.95)]))
        if r in tension_rounds:
            block.append(_two(f"r{r}_tension", ["wren", ai], {"wren": "smug", ai: "calm"}, b.get("tension", "HMM.")))
        if r < 2:
            block.append(dict(id=f"r{r}_score", type="insert", prop=dict(kind="score", round=r, rows=rows), lines=[_line("narrator", f"{NUM[vw].capitalize()} versus {NUM[rd['va']]}.")]))
            if quips[r]:
                block.append(_close(f"r{r}_quip", "wren", quips[r], "smug", anim="hop"))
        else:
            block.append(dict(id="r2_score", type="insert", prop=dict(kind="score", round=2, rows=rows, to=[vw, 10], flip_at=1.0, hi=1), pad=0.3,
                              lines=[_line("narrator", f"{NUM[hid].capitalize()} hidden points.", None, pre=0.3), _line("narrator", "Ten points.", None, pre=0.3)]))
        for s in block:
            s["tag"] = tag
        shots += block
    shots += [_scream(b.get("stamp", "NO WAY")), _cta("catan", ai, CTA1["catan"][rng.randrange(3)])]
    return "catan", rounds, shots


def _pick(seq, rng):
    return seq[rng.randrange(len(seq))]


def episode_from_brief(b, eid=None):
    """brief -> validated episode spec.  Raises ValueError with readable errors (what an LLM is shown)."""
    errs = validate_brief(b)
    if errs:
        raise ValueError("invalid brief:\n  - " + "\n  - ".join(errs))
    rng = random.Random(int(b.get("seed", 0)))
    game, rounds, shots = (_catan_episode if b["story"] == "catan_behind" else _poker_episode)(b, rng)
    hook = b["hook"].rstrip(".")
    eid = eid or b.get("id") or "ep_" + hashlib.md5(json.dumps(b, sort_keys=True).encode()).hexdigest()[:8]
    store = "#catan" if game == "catan" else "#poker"
    where = "Play Catan-style Settler Rush" if game == "catan" else "Play Poker"
    ep = dict(id=eid, title=f"POV: {hook}", hook=dict(pov="POV:", text=hook), series="rounds", story=b["story"], game=game, cast=["wren", b["ai"]], rounds=rounds,
              music=dict(level=0), caption=f"POV: {hook[0].lower() + hook[1:]}.\n{where} against frontier AI at gghuman.ai\n{HASHTAGS} {store}", shots=shots, brief=b)
    if game == "catan":
        ep["board_seed"] = int(b.get("board_seed", 1 + int(b.get("seed", 0)) % 97))
    errs = validate(ep)
    if errs:
        raise ValueError("episode failed validation:\n  - " + "\n  - ".join(errs))
    return ep


# ------------------------------------------------------------------ batch generation + diversity gate
def _hook(kind, ai, n):
    """Cycle through every base hook x {'The AI', GPT, Claude, Gemini}: a 100-video batch uses ~100 different hooks."""
    hooks = POOLS[kind]["hooks"]
    k = n // len(KINDS)
    base = hooks[k % len(hooks)]
    cycle = (k // len(hooks)) % 4
    return base if cycle == 0 else re.sub(r"\bthe AI\b", AI_NAME[ai], base, count=1, flags=re.I)


def make_brief(rng, kind, n):
    P, ai = POOLS[kind], rng.choice(AIS)
    qp = rng.choice(QUIP_PATTERNS)
    q = rng.choice(P["quips"])
    return dict(id=f"gen{n:03d}_{kind}_{ai}", story=kind, ai=ai, hook=_hook(kind, ai, n), opener=rng.choice(P["opener"]), claims=list(rng.choice(P["claims"])),
                quips=[q[0] if qp[0] else None, q[1] if qp[1] else None], tension=rng.choice(TENSION_WORDS), tension_rounds=list(rng.choice(TENSION_SETS)),
                stamp=rng.choice(STAMPS), seed=rng.randrange(10 ** 6), board_seed=1 + (n * 7) % 97)


def structure_sig(ep):
    return tuple(s["type"] + (":" + s["prop"]["kind"] if s["type"] == "insert" else "") for s in ep["shots"])


def diversity(scripts):
    """Metrics a reviewer (or the batch gate) can check: are these N videos really N different videos?"""
    n = len(scripts)
    if n == 0:
        return dict(n=0, passed=True)
    hooks = Counter(s["hook"]["text"] for s in scripts)
    dialog = Counter(tuple(l["text"] for sh in s["shots"] for l in sh.get("lines", [])) for s in scripts)
    structs = Counter(structure_sig(s) for s in scripts)
    boards = {s.get("board_seed", 0) for s in scripts if s.get("game") == "catan"}
    ncat = sum(s.get("game") == "catan" for s in scripts)
    m = dict(n=n, distinct_hooks=len(hooks), max_hook_share=round(max(hooks.values()) / n, 3), distinct_dialogues=len(dialog), distinct_structures=len(structs),
             catan_videos=ncat, distinct_catan_boards=len(boards), stories=dict(Counter(s.get("story") for s in scripts)))
    need_h = 0.9 if n <= 24 else 0.8
    m["passed"] = bool(len(hooks) >= need_h * n and len(dialog) >= 0.95 * n and (n < 20 or len(structs) >= 8) and (ncat == 0 or len(boards) >= 0.8 * ncat)
                       and (n < 20 or m["max_hook_share"] <= 0.05))
    return m


def generate(count, seed=1):
    rng = random.Random(seed)
    out, seen, tries = [], set(), 0
    while len(out) < count and tries < count * 50:
        tries += 1
        n = len(out)
        ep = episode_from_brief(make_brief(rng, KINDS[n % len(KINDS)], n))
        sig = hashlib.md5(json.dumps([ep["hook"], [[l["text"] for l in s.get("lines", [])] for s in ep["shots"]], ep["cast"]], sort_keys=True).encode()).hexdigest()
        if sig in seen:
            continue
        seen.add(sig)
        out.append(ep)
    return out


def generate_many(count, seed, out_dir: Path, enforce_diversity=True):
    """Write `count` validated episodes + diversity.json.  Returns the number written; raises if the batch is too repetitive."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    eps = generate(count, seed)
    for ep in eps:
        (out_dir / f"{ep['id']}.json").write_text(json.dumps(ep, indent=1), encoding="utf-8")
    d = diversity(eps)
    (out_dir / "diversity.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    if enforce_diversity and not d["passed"]:
        raise RuntimeError(f"batch is too repetitive: {d}")
    return len(eps)


# ------------------------------------------------------------------ LLM mode: the LLM writes a BRIEF, code does the rest
EXAMPLE_BRIEF = dict(story="bluff_trust", ai="gpt", hook="You learn to trust the AI's bluffs", opener="Let's play.", claims=["I'm bluffing.", "Bluffing again.", "Definitely bluffing."],
                     quips=["So easy.", "Too easy."], tension="HMM.", tension_rounds=[2], stamp="BUSTED")
SYSTEM = """You write 20-24 second social-video briefs for GGhuman (play board games against frontier AI at gghuman.ai).
Format "HAND 1 / HAND 2 / HAND 3" (rule of three): a hand-drawn human (WREN) plays Poker or Catan against one robot AI (gpt, claude or gemini).
The AI makes the SAME kind of claim three rounds in a row, escalating the wording. In rounds 1 and 2 the claim is honest and the AI loses; in round 3 the claim is a lie
and the AI wins big. Wren screams. Code deals the cards/scores and checks the game logic; YOU write only the funny text.
Return ONLY one JSON object with exactly these keys:
  story: "bluff_trust" (AI says it's bluffing) | "nothing_trust" (AI says it has nothing/junk) | "catan_behind" (AI says it's losing/behind at Catan)
  ai: "gpt" | "claude" | "gemini"
  hook: the handwritten POV line, 12-70 characters, a NEW funny hook that is not in the list of used hooks
  opener: the AI's first line (2-6 words)
  claims: exactly 3 lines (2-6 words each), the AI's claim in round 1, 2, 3, escalating; at least 2 must clearly state the claim (bluff / nothing / losing)
  quips: [Wren's line after round 1, Wren's line after round 2] (each 2-6 words, or null; at least one not null)
  tension: a comic word for the tension beat (2-8 chars, e.g. "HMM.")
  tension_rounds: non-empty subset of [0,1,2] (rounds that get a tension beat), e.g. [2] or [1,2]
  stamp: a comic word over Wren's scream (2-12 chars, e.g. "BUSTED")
No single-word lines (they are hard to hear). Example:
""" + json.dumps(EXAMPLE_BRIEF)


def _call_anthropic(system, messages, model=None):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    body = json.dumps(dict(model=model or os.environ.get("GGH_LLM_MODEL", "claude-sonnet-5-5"), max_tokens=2000, system=system, messages=messages)).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", body, {"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=120).read())["content"][0]["text"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Anthropic API error {e.code}: {e.read().decode('utf-8', 'replace')[:300]} (set GGH_LLM_MODEL to use another model)") from None


def check_novelty(b, used_hooks):
    """LLM briefs must not repeat a hook that already exists (case/punctuation-insensitive)."""
    norm = lambda t: re.sub(r"[^a-z0-9 ]", "", str(t).lower()).strip()
    return [f"hook {b.get('hook')!r} is already used - write a different hook"] if norm(b.get("hook", "")) in {norm(x) for x in used_hooks} else []


def llm_generate(idea, call=None, max_tries=4, used_hooks=(), log=None, seed=None):
    call = call or _call_anthropic
    avoid = ("\nHooks already used (do not repeat): " + "; ".join(sorted(set(used_hooks))[:60])) if used_hooks else ""
    msgs = [dict(role="user", content=f"Write a NEW brief for this idea:\n{idea}{avoid}")]
    errs = []
    for k in range(max_tries):
        txt = call(SYSTEM, msgs)
        if log is not None:
            log.append(dict(try_=k, response=txt))
        try:
            b = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
            b.setdefault("seed", seed if seed is not None else int(hashlib.md5(idea.encode()).hexdigest(), 16) % 10 ** 6)
            errs = check_novelty(b, used_hooks)
            if not errs:
                return episode_from_brief(b)
        except ValueError as e:
            errs = [str(e)]
        except Exception as e:
            errs = [f"not valid JSON: {e}"]
        msgs += [dict(role="assistant", content=txt), dict(role="user", content="The validator rejected the brief:\n- " + "\n- ".join(errs) + "\nReturn the corrected full JSON only.")]
    raise RuntimeError(f"LLM brief still invalid after {max_tries} tries: {errs}")


def llm_batch(ideas, out_dir, call=None, used_hooks=(), max_tries=4):
    """One episode per idea.  Every prompt/response is logged to <out_dir>/llm_log.jsonl; a failing idea never stops the batch.
    Returns (written_paths, failures)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    used, written, failed = list(used_hooks), [], []
    with open(out_dir / "llm_log.jsonl", "a", encoding="utf-8") as lf:
        for idea in ideas:
            log = []
            try:
                ep = llm_generate(idea, call=call, max_tries=max_tries, used_hooks=used, log=log)
                p = out_dir / f"{ep['id']}.json"
                p.write_text(json.dumps(ep, indent=1), encoding="utf-8")
                used.append(ep["hook"]["text"]); written.append(p)
                lf.write(json.dumps(dict(idea=idea, ok=True, id=ep["id"], tries=len(log), log=log)) + "\n")
            except Exception as e:
                failed.append((idea, str(e)))
                lf.write(json.dumps(dict(idea=idea, ok=False, error=str(e), tries=len(log), log=log)) + "\n")
    return written, failed
