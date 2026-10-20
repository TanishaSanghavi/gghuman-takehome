# GGhuman Reel Generator (v4: "HAND 1 / HAND 2 / HAND 3")

Generates vertical (9:16, 720x1280, 30 fps) Instagram reels for **@gghumanai** in the page's own hand-drawn style, in one repeatable, recognisable format:
**rule of three**. The AI makes the *same* claim three rounds in a row ("I'm bluffing." / "I have nothing." / "I'm losing."). Rounds 1 and 2 prove it honest
(the AI really loses, the pot grows BET -> RAISE). In round 3 the pot is ALL IN and the claim turns out to be a lie. Wren screams, the gghuman.ai end card plays.

What makes the series recognisable on screen: a **HAND / TURN tag** (top-left), a running **WINS tally** (top-right), an **escalating pot** (BET -> RAISE -> ALL IN),
a **two-hand showdown** with the winner outlined and a **YOU WIN / AI WINS** stamp, whip transitions between rounds, tension beats, and a comic word over the scream.
Characters, props, captions and footer are the page's own. Each video has 14-15 shots in 22-25 s.

A video is a short JSON episode built from a small creative **brief** (hook, opener, three claims, quips, stamp word). Code deals the cards and scores, builds the shots,
validates the game logic (including that the round-3 lie is *really* a lie), voices it with a neural TTS model, renders, mixes, and then **checks the finished file**:
timing, loudness, frames, and, with an offline speech-recognition model, that the video really *says* what the script says.

## The three submission videos (`episodes/`)

| Episode | Story | Hook | Rounds 1-2 (honest) | Round 3 (the lie) |
| --- | --- | --- | --- | --- |
| `ep01_poker_bluff_trust` | `bluff_trust` | The AI always tells you when it's bluffing | "I'm bluffing" and it really is: high card, you win | "Definitely bluffing" with a flush |
| `ep02_catan_behind` | `catan_behind` | You trust the AI when it says it's behind | "I'm losing" and it really is: you lead on points | "Definitely losing" with hidden victory points: 7 + 3 = 10 |
| `ep03_poker_nothing_trust` | `nothing_trust` | The AI keeps saying it has nothing | "I have nothing" and it really does | "Absolutely nothing" with three of a kind or better |

The briefs were written by an LLM (Claude) from the series description (`tools/make_submission_scripts.py`); `python run.py llm` / `llm-batch` automate that step.

## Quick start

Requirements: Python 3.12 and ffmpeg (with ffprobe).

```bash
# ffmpeg: brew install ffmpeg (macOS) | sudo apt install ffmpeg (Ubuntu) | winget install Gyan.FFmpeg (Windows)
python3 -m venv .venv && source .venv/bin/activate        # Windows: py -3.12 -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python setup_assets.py          # downloads two offline AI models, once: Kokoro TTS (~320 MB) + Whisper tiny.en ASR (~120 MB)
python run.py doctor            # everything must say ok
python -m pytest -q             # fast tests, ~9 s   (add --runslow to also render real video)
python run.py all               # builds the three videos into out/   (about 1-2 minutes per video on ONE CPU core)
```

Each video lands in `out/<id>/`: `<id>.mp4`, `qa.json` (every check), `asr_transcript.json` (what the speech recognizer heard for every line),
`sync_report.png` (waveform vs every cue), `cover.jpg`, `caption.txt`, `timeline.json`, `result.json`. Verified copies of the three submission videos
and their reports are in `submission/`.

## Commands

| Command | What it does |
| --- | --- |
| `python run.py build episodes/ep01_poker_bluff_trust.json` | Build one episode |
| `python run.py all` | Build every episode in `episodes/` |
| `python run.py gen --count 100 --seed 7` | Generate 100 validated episodes into `episodes_gen/` and **refuse the batch if it is repetitive** (diversity gate) |
| `python run.py batch episodes_gen --jobs 4` | Build a folder of episodes, 4 videos in parallel; finished unchanged videos are skipped; writes `batch_report.json` and `publish_queue.csv` |
| `python run.py batch episodes_gen --shard 0/4` | Every 4th episode, to split a batch across machines |
| `python run.py llm "POV: ..." --build` | LLM writes one brief (needs `ANTHROPIC_API_KEY`); validated + novelty-checked; code builds the episode; then the video |
| `python run.py llm-batch ideas.txt` | One LLM brief per line of `ideas.txt`; every prompt and reply logged to `episodes_llm/llm_log.jsonl` |
| `python run.py qa out/<id>/<id>.mp4 out/<id>/timeline.json` | Re-run the checks on an existing video |
| `python run.py doctor` | Check the environment (ffmpeg, fonts, both AI models, TTS backend) |

Flags: `--workers N` (frames in parallel inside one video), `--jobs N` (videos in parallel), `--force`, `--out DIR`,
`--allow-fallback-voice` (accept non-neural voices; off by default), `--no-asr`, `--strict-asr` (fail instead of warn if the ASR model is missing).

## AI tools used, and what each one does

| Stage | Tool | Role |
| --- | --- | --- |
| Creative text | **Claude** (LLM) | Writes the brief from a one-line idea (`llm` / `llm-batch`; the three submission briefs were written this way) |
| Voices | **Kokoro v0.19** neural TTS, offline | One distinct voice per character, word timings for captions and lip-sync |
| Quality check | **Whisper tiny.en** ASR, offline | Transcribes every spoken line *from the finished mp4* and compares it with the script |
| Cards, scores, shots, drawing, sound effects, mix | Plain Python / NumPy / ffmpeg | Deterministic: the same brief always renders the same frames |

The LLM only writes the funny part; it cannot break the game logic because code deals the cards and validates the result. Characters and props are drawn in code
(the page's look), not generated by an image model; that is what keeps 100 videos consistent.

## How a video is checked (30 checks per video, all on the FINAL mp4)

1. **Brief, before anything else**: hook length, 2-6 word lines (no single-word lines: they are hard to hear and to recognise), claims that actually state the story's claim.
2. **Episode, before any rendering** (`spec.py`, `gamelogic.py`): 7-card poker ranking computed by code; no duplicate cards; in rounds 1-2 the AI must really hold only a
   high card and lose; in round 3 the lie must be real and win (`bluff_trust`: flush or better, `nothing_trust`: three of a kind or better); Catan: you stay level and ahead,
   the AI's visible score rises, visible + hidden VP = 10, at most 5 VP cards exist; every showdown / score shot must show exactly its round's data.
3. **Timeline**: every voice line, caption, sound effect and cut is placed on the 30 fps grid once; renderer and mixer both read it.
4. **Final file**: format, fps, AAC 48 kHz, duration, size; loudness about -14 LUFS, no clipping; true silence before the scream, then a loud hit;
   voice starts on its caption frame (<=20 ms); cuts and captions land on their scheduled frames; no frozen or black frames; hook visible in frame 0;
   first line within 1 s; no dead air over 1.7 s; every spoken word captioned; ends on the gghuman.ai button.
5. **A neural voice was used for every line** (`neural_voice_used`): a video voiced by the babble/espeak fallback is marked `degraded`, never `ok`.
6. **Speech recognition** (`speech_intelligible_asr`): each line is cut out of the final audio, transcribed, and compared with the script (tolerant of near-misses
   like "badly" -> "bad lady"; a babble voice scores 0.10-0.15 and fails).

## Failure handling

| Situation | Behaviour |
| --- | --- |
| Invalid brief / impossible game state / a "lie" that isn't one | Rejected before rendering (`invalid_spec`) with a readable reason; with an LLM the reasons are sent back for a rewrite (up to 4 tries) |
| LLM reuses an existing hook | Rejected by the novelty check and sent back |
| Neural voice missing or a line falls back to babble | `degraded`: never ok, never in `publish_queue.csv` |
| Voice not intelligible (ASR) | `qa_failed` |
| Loudness, peak or audio offset out of range | Remixed / re-joined with corrected settings, then re-checked |
| A frame fails to render | Previous frame held; the video fails if more than 1% fail |
| Unexpected crash (ffmpeg, killed worker) | Whole video retried once; a deterministic failure (bad script, failed QA) is not retried |
| One video fails in a batch | Batch continues; listed in `batch_report.json` |
| Batch re-run or interrupted | Finished, unchanged videos are skipped; changing the checks (`PIPELINE_VERSION`) invalidates old "ok" results |
| Model download interrupted | `setup_assets.py` extracts to a temp folder and verifies sizes before moving into place |

## Producing 100 videos

```bash
python run.py gen --count 100 --seed 7        # 100 validated episodes; fails if the batch is repetitive
python run.py batch episodes_gen --jobs 4     # build them; verified videos are listed in out/publish_queue.csv
```

Three story types, each with 12 hooks x {"The AI", GPT, Claude, Gemini}, 6 claim escalations, 6 quip pairs, 4 openers, 3 quip patterns, 4 tension placements, a different Catan
board per video, and every deal computed in code. Measured on the seed-7 batch of 100: 94 distinct hooks, 97 distinct dialogues, 24 distinct shot structures,
33 different Catan boards, no hook used more than twice.

Throughput: about 1-2 minutes per video on one CPU core (65-115 s measured per 24 s video). 100 videos is about 2-3 h on one core, roughly divided by the number of cores
(`--jobs`) or machines (`--shard i/n`). `publish_queue.csv` (id, video, cover, caption, duration) is the hand-off to a scheduler. **Instagram posting itself is not built**
(it needs a Meta app and Graph API access); the queue only ever contains videos that passed every check.

## Tests

`python -m pytest -q` runs 38 fast tests (hand ranking incl. wheel straights, the rule-of-three validation, Catan arithmetic, brief validation, deterministic generator,
diversity gate, LLM loop with an injected fake model, ASR positive/negative control, QA gates, retry/caching logic, publish queue). `python -m pytest -q --runslow`
additionally renders a real mini video and proves a babble-voice build is `degraded`, not `ok`.

## Layout

```
run.py              CLI                         setup_assets.py   downloads both models (verified)
selftest.py         1-minute smoke test         ideas.txt         example ideas for llm-batch
episodes/           the 3 submission episodes   submission/       verified videos + QA reports + write-up
tools/              make_submission_scripts.py (briefs) make_writeup.py          tests/   pytest suite
gghuman/  spec.py gamelogic.py (validation)  scriptgen.py (briefs, generator, diversity gate, LLM stage)  timeline.py tts.py asr.py
          render.py art/ (drawing)  sfx.py mixer.py music.py  qa.py (final-file checks)  pipeline.py (build, retry, status)  encode.py viz.py
assets/fonts/       bundled fonts (SIL OFL)
```

## Credits

Voices: Kokoro (Apache-2.0) and speech recognition: Whisper tiny.en (MIT), both run through sherpa-onnx. Fonts: Patrick Hand, Bangers, Nunito (SIL OFL).
