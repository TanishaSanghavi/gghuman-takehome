#!/usr/bin/env python3
"""Builds GGhuman_Takehome_Round2.pdf (needs reportlab: pip install reportlab)."""
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

out = sys.argv[1] if len(sys.argv) > 1 else "GGhuman_Takehome_Round2.pdf"
ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6, leading=13, spaceAfter=4)
H = ParagraphStyle("h", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, spaceBefore=9, spaceAfter=3, textColor=colors.HexColor("#1b3a5c"))
T = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=18, spaceAfter=2, alignment=0)
C = ParagraphStyle("c", parent=B, fontSize=8.6, leading=11, spaceAfter=0)
CB = ParagraphStyle("cb", parent=C, fontName="Helvetica-Bold")
BL = ParagraphStyle("bl", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2)


def P(t, s=B): return Paragraph(t, s)


def table(rows, widths, head=True):
    data = [[Paragraph(str(c), CB if (head and i == 0) else C) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2cc")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if head:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6edf4")))
    t.setStyle(TableStyle(st))
    return t


s = [P("GGhuman Take-Home: Round 2", T), P("Tanisha Sanghavi, MSCS, UMass Amherst", B)]

s += [P("1. What changed after your feedback", H),
      P("Round 1 was judged not near the quality you can post. I compared it with five real reels from the page: mine used a dark UI board and no characters, so it did not look like GGhuman at all. "
        "Round 2 uses the page's own visual language (Wren and the three robots, hand-drawn props, word-by-word captions, footer, scream, end card) and builds a <b>new, recognisable series format</b> "
        "on top of it, with far more happening per second: 14 to 15 shots per video instead of 9 to 10. I built it with Claude as a coding and creative assistant.")]

s += [P("2. The format: \"HAND 1 / HAND 2 / HAND 3\" (rule of three)", H),
      P("The AI makes the <b>same claim three rounds in a row</b>. In rounds 1 and 2 the claim is honest: the AI really loses and the pot grows (BET, then RAISE). In round 3 the pot is ALL IN and the claim "
        "turns out to be a lie. Wren screams, the gghuman.ai end card plays. It shows the site's promise directly (an AI that sets you up, not just plays you) and invites comments (\"I'd have fallen for it too\"). "
        "Recognisable on screen in every video: a HAND / TURN tag (top-left), a running WINS tally (top-right), an escalating pot, a two-hand showdown with the winner outlined and a YOU WIN / AI WINS stamp, "
        "whip transitions between rounds, and a comic word over the scream. Each video is 24 to 25 s, 720x1280, 30 fps."),
      table([["#", "Video", "The AI says (x3)", "Rounds 1-2", "Round 3: the lie"],
             ["1", "ep01_poker_bluff_trust", "\"I'm bluffing.\" / \"Bluffing again.\" / \"Definitely bluffing.\"", "High card, you win", "A flush"],
             ["2", "ep02_catan_behind", "\"I'm losing.\" / \"Still losing.\" / \"Definitely losing.\"", "You lead 9 to 4, then 9 to 6", "4 hidden victory points: 6 becomes 10"],
             ["3", "ep03_poker_nothing_trust", "\"I have nothing.\" / \"Still nothing.\" / \"Absolutely nothing.\"", "High card, you win", "A full house"]],
            [0.3, 1.7, 2.4, 1.3, 1.4])]

s += [P("3. Workflow and AI tools", H),
      table([["Stage", "Tool", "What it does"],
             ["Creative text", "Claude (LLM)", "Writes a small <b>brief</b> (hook, opener, three escalating claims, quips, stamp word) from a one-line idea. <b>python run.py llm / llm-batch</b> automates this; the three submission briefs were written this way."],
             ["Voices", "Kokoro v0.19 neural TTS (offline)", "One voice per character, with word timings for captions and lip-sync."],
             ["Quality check", "Whisper tiny.en ASR (offline)", "Transcribes every spoken line from the <i>finished mp4</i> and compares it with the script."],
             ["Cards, scores, shots, drawing, sound, mix", "Python, NumPy, ffmpeg", "Deterministic: the same brief always renders the same frames, so 100 videos stay consistent."]],
            [1.4, 1.9, 3.7]),
      P("The LLM only writes the funny part. Code deals every hand and score, builds the shots, and decides whether the result is allowed to be produced at all, so the model cannot break the game logic.")]

s += [P("4. How I check quality", H),
      P("<b>Before rendering:</b> the brief is validated (lengths, no single-word lines, claims that really state the claim). The episode is validated by code: poker hands are ranked (7-card evaluation, wheel straights, "
        "no duplicate cards); in rounds 1-2 the AI must really hold only a high card and lose; in round 3 the lie must be real and win (a flush or better for \"bluffing\", three of a kind or better for \"nothing\"); "
        "Catan scores must add up (visible + hidden VP = 10, at most 5 VP cards exist, you stay ahead early); every showdown and score shot must show exactly its round's data. "
        "With an LLM, errors are sent back for a rewrite (up to 4 tries), plus a novelty check against existing hooks."),
      P("<b>On the finished mp4 (30 checks):</b> format, fps, loudness (about -14 LUFS), no clipping; silence before the scream and a loud hit after it; each voice starts on its caption frame (within 20 ms); "
        "cuts and captions land on their scheduled frames; no frozen or black frames; hook visible in frame 0; first line within 1 s; no dead air over 1.7 s; every spoken word captioned; "
        "ends on the gghuman.ai button; a neural voice for every line; and the speech check: every line transcribed back from the final audio."),
      P("<b>What the checks caught while I was building:</b>"),
      P("the speech check heard \"Nine to six.\" as \"96\" (the word \"to\" was swallowed), so I changed the narration to \"Nine versus six.\"; earlier it caught \"Badly.\" heard as \"Bad lady\" and a one-word \"Call.\" "
        "heard as \"Co-owner\", so the series now forbids single-word lines;", BL),
      P("my own generator failed its diversity gate twice (a rotation bug, then hooks that never varied), and the validator rejected my first brief for a one-word line;", BL),
      P("a silent weakness: if the neural voice fails to load, the old pipeline quietly used a babble voice and still marked the video ok. Those builds are now <b>degraded</b> and never published. "
        "As a negative control, the babble voice scores 0.10 to 0.15 on the speech check and Kokoro scores 0.8 or higher.", BL),
      P("A person still skims each video and caption before posting; the checks decide what is <i>allowed</i> to reach that person.")]

s += [P("5. Failure handling", H),
      table([["Failure", "What happens"],
             ["Invalid brief, impossible hand, a \"lie\" that isn't one", "Rejected before rendering with a readable reason; with an LLM, the reason is sent back for a rewrite"],
             ["Fallback voice, or speech not intelligible", "<b>degraded</b> or <b>qa_failed</b>: never ok, never in the publish queue"],
             ["Loudness or audio offset out of range", "Remixed or re-joined with corrected settings, then re-checked"],
             ["Crash (ffmpeg, killed worker)", "Whole video retried once; deterministic failures are not retried"],
             ["One video fails in a batch", "The batch continues; listed in batch_report.json; finished, unchanged videos are skipped on re-run"],
             ["Interrupted model download", "setup_assets.py verifies sizes before installing, so a truncated model cannot be used"]],
            [2.5, 4.5]),
      P("The repo has 38 automated tests (plus integration tests that render real video and prove a babble-voice build is rejected).")]

s += [P("6. Scaling to 100 videos", H),
      P("<font face='Courier'>python run.py gen --count 100 --seed 7</font> then <font face='Courier'>python run.py batch episodes_gen --jobs 4</font>. "
        "The generator has three story types, 12 hooks each (every one also in a GPT / Claude / Gemini version), six escalating claim sets, six quip pairs, four tension placements and a different Catan board per video, "
        "and is checked by a <b>diversity gate</b> that refuses a repetitive batch (repetitive content tends to perform worse). On the seed-7 batch: 100 episodes, 94 distinct hooks, 97 distinct dialogues, "
        "24 distinct shot structures, 33 different Catan boards."),
      P("<b>Throughput:</b> 65 to 115 s per video on a single CPU core, so 100 videos is about 2 to 3 hours on one core, divided by cores (<i>--jobs</i>) or machines (<i>--shard i/n</i>). "
        "The batch writes <i>publish_queue.csv</i> containing only videos that passed every check, with caption and cover. For more variety, <i>llm-batch</i> turns a list of ideas into validated briefs, with every prompt and reply logged.")]

s += [P("7. Limitations and next steps", H),
      P("The art is drawn in code in the page's style, not generated by an image model: that keeps 100 videos consistent, but the page's hand-animated reels still have richer one-off gags, so more story types "
        "and props are the main next step. Whisper tiny is a coarse judge; a larger model would be stricter. "
        "The LLM stage was tested with an injected fake model, not a live API call. Posting to Instagram through the Graph API is the missing step toward the page's 95% automation; I have not built it. "
        "After that, engagement data (completion rate, clicks) should decide which stories and hooks get more videos.")]

SimpleDocTemplate(out, pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                  title="GGhuman Take-Home: Round 2", author="Tanisha Sanghavi").build(s)
print("wrote", out)
