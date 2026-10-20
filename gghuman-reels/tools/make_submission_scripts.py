#!/usr/bin/env python3
"""The three submission episodes.  Each is a BRIEF written by an LLM (Claude) from the series description; code (scriptgen.episode_from_brief) deals the
cards / scores, builds the shots and validates the game logic (including that the AI's round-3 claim really is a lie).
Edit the briefs below and re-run to regenerate episodes/*.json."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gghuman.scriptgen import episode_from_brief

BRIEFS = [
    dict(id="ep01_poker_bluff_trust", story="bluff_trust", ai="gpt", seed=11, hook="The AI always tells you when it's bluffing", opener="Deal me in.",
         claims=["I'm bluffing.", "Bluffing again.", "Definitely bluffing."], quips=["So easy.", "Too easy."], tension="HMM.", tension_rounds=[2], stamp="BUSTED"),
    dict(id="ep02_catan_behind", story="catan_behind", ai="claude", seed=7, board_seed=17, hook="You trust the AI when it says it's behind", opener="Roll the dice.",
         claims=["I'm losing.", "Still losing.", "Definitely losing."], quips=["Easy win.", "Too easy."], tension="?!", tension_rounds=[2], stamp="WHAT"),
    dict(id="ep03_poker_nothing_trust", story="nothing_trust", ai="gemini", seed=23, hook="The AI keeps saying it has nothing", opener="Here we go.",
         claims=["I have nothing.", "Still nothing.", "Absolutely nothing."], quips=["Free chips.", None], tension="UH OH", tension_rounds=[0, 2], stamp="NO WAY"),
]
for b in BRIEFS:
    ep = episode_from_brief(b)
    (ROOT / "episodes" / f"{ep['id']}.json").write_text(json.dumps(ep, indent=1), encoding="utf-8")
    print("wrote", ep["id"], len(ep["shots"]), "shots")
