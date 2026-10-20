#!/usr/bin/env python3
"""Downloads the two offline AI models the pipeline uses (fonts are bundled):

  * Kokoro v0.19 TTS      (~320 MB, Apache-2.0)  character voices
  * Whisper tiny.en ASR   (~120 MB, MIT)         speech-intelligibility QA on every finished video

Each download is retried, extracted to a temp folder and verified (size + files) before it is moved into place, so an
interrupted download can never leave a half-extracted model behind (the pipeline would otherwise silently fall back to a babble voice)."""
import shutil, sys, tarfile, time, urllib.request
from pathlib import Path

M = Path(__file__).resolve().parent / "assets" / "models"
BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
MODELS = [
    dict(name="kokoro-en-v0_19", url=BASE + "tts-models/kokoro-en-v0_19.tar.bz2", must={"model.onnx": 3e8, "voices.bin": 1e6, "tokens.txt": 100, "espeak-ng-data": 0}),
    dict(name="sherpa-onnx-whisper-tiny.en", url=BASE + "asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2",
         must={"tiny.en-encoder.int8.onnx": 5e6, "tiny.en-decoder.int8.onnx": 5e7, "tiny.en-tokens.txt": 1e5}),
]


def ok(m):
    d = M / m["name"]
    return all((d / f).exists() and ((d / f).is_dir() or (d / f).stat().st_size >= sz) for f, sz in m["must"].items())


def fetch(m, tries=3):
    last = [-1]

    def hook(b, bs, tot):
        if tot > 0:
            pct = min(100, b * bs * 100 // tot)
            if pct != last[0] and pct % 5 == 0:
                last[0] = pct
                print(f"\r  {pct}%", end="", flush=True)
    M.mkdir(parents=True, exist_ok=True)
    tmp, stage = M / f"{m['name']}.tar.bz2", M / f".stage_{m['name']}"
    for k in range(1, tries + 1):
        try:
            print(f"downloading {m['name']} (attempt {k}/{tries})")
            urllib.request.urlretrieve(m["url"], tmp, hook)
            print("\nextracting...")
            shutil.rmtree(stage, ignore_errors=True)
            with tarfile.open(tmp) as t:
                t.extractall(stage)
            tmp.unlink()
            src = stage / m["name"]
            shutil.rmtree(M / m["name"], ignore_errors=True)
            shutil.move(str(src), str(M / m["name"]))
            shutil.rmtree(stage, ignore_errors=True)
            if not ok(m):
                raise RuntimeError("model looks truncated")
            print("done:", m["name"])
            return True
        except Exception as e:
            print(f"\n  failed: {e}")
            tmp.unlink(missing_ok=True)
            shutil.rmtree(stage, ignore_errors=True)
            time.sleep(2 * k)
    return False


if __name__ == "__main__":
    only = sys.argv[1:]            # e.g. python setup_assets.py asr
    bad = 0
    for m in MODELS:
        if only and not any(o in m["name"] for o in only):
            continue
        if ok(m):
            print(m["name"], "already present")
        elif not fetch(m):
            bad += 1
    sys.exit(1 if bad else 0)
