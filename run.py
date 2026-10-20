#!/usr/bin/env python3
"""GGhuman reel generator CLI.

  python run.py doctor                          check environment
  python run.py build episodes/ep01_poker_weak_hand.json [more.json ...]
  python run.py all                             build every script in episodes/
  python run.py gen --count 100 --seed 7        generate 100 validated scripts (deals/scores computed by code)
  python run.py batch episodes_gen --workers 4 [--shard 0/4] [--limit N]
  python run.py llm "POV: ..." [--build]        LLM-written script, optionally built (needs ANTHROPIC_API_KEY)
  python run.py llm-batch ideas.txt             one LLM script per line of ideas.txt (validated, novelty-checked, logged)
  python run.py batch episodes_gen --jobs 4     build videos in parallel processes (production)
  python run.py qa out/<id>/<id>.mp4 out/<id>/timeline.json   check an existing video
"""
import argparse, json, logging, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def cmd_doctor(a):
    import shutil
    from gghuman import config
    ok = True
    print("ffmpeg/ffprobe :", "ok" if shutil.which("ffmpeg") and shutil.which("ffprobe") else "MISSING"); ok &= bool(shutil.which("ffmpeg"))
    for f in ("PatrickHand-Regular.ttf", "Bangers-Regular.ttf", "Nunito.ttf"):
        e = (config.FONTS / f).exists(); ok &= e
        print(f"font {f:26}:", "ok" if e else "MISSING")
    m = config.MODELS / "kokoro-en-v0_19" / "model.onnx"
    print("kokoro model   :", "ok" if m.exists() else "missing -> python setup_assets.py (falls back to blip voices without it)")
    from gghuman import asr
    print("asr model      :", "ok" if asr.available() else "missing -> python setup_assets.py (speech-intelligibility QA is skipped without it)")
    from gghuman import tts
    try:
        b = tts.get_backend(); print("TTS backend    :", b.name)
    except Exception as e:
        print("TTS backend    : FAILED", e); ok = False
    print("cpu cores      :", os.cpu_count())
    return 0 if ok else 1


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _summary(results):
    print("\n{:<26} {:<11} {:>6} {:>7} {:>6}  {}".format("episode", "status", "dur", "LUFS", "secs", "qa"))
    for r in results:
        q = r.get("qa", {})
        print("{:<26} {:<11} {:>6} {:>7} {:>6}  {}".format(r["id"], r["status"], round(r.get("duration", 0), 1), q.get("lufs", ""), r.get("total_seconds", ""),
              f'{q.get("n_pass", 0)} pass / {q.get("n_warn", 0)} warn / {q.get("n_fail", 0)} fail' + (" CACHED" if r.get("cached") else "")))
        if r["status"] not in ("ok",):
            print("    ->", r.get("error") or r.get("errors") or q.get("failed"))


def _kw(a):
    return dict(allow_fallback_voice=a.allow_fallback_voice, asr=not a.no_asr, strict_asr=a.strict_asr)


def cmd_build(a):
    from gghuman.pipeline import build_episode
    res = []
    for p in a.specs:
        res.append(build_episode(_load(p), a.out, a.workers, a.force, a.crf, a.preset, **_kw(a)))
    _summary(res)
    return 0 if all(r["status"] == "ok" for r in res) else 2


def cmd_all(a):
    a.specs = sorted(str(p) for p in (ROOT / "episodes").glob("*.json"))
    return cmd_build(a)


def cmd_gen(a):
    from gghuman.scriptgen import generate_many
    try:
        n = generate_many(a.count, a.seed, Path(a.dir))
    except RuntimeError as e:
        print("ERROR:", e)
        return 3
    d = json.loads((Path(a.dir) / "diversity.json").read_text(encoding="utf-8"))
    print(f"wrote {n} validated scripts to {a.dir}")
    print("diversity:", {k: v for k, v in d.items() if k != "stories"}, "stories:", d.get("stories"))
    return 0


def cmd_llm_batch(a):
    from gghuman.scriptgen import llm_batch
    ideas = [l.strip() for l in Path(a.ideas).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    used = [json.loads(p.read_text(encoding="utf-8"))["hook"]["text"] for p in (ROOT / "episodes").glob("*.json")]
    written, failed = llm_batch(ideas, a.dir, used_hooks=used)
    print(f"{len(written)} scripts written to {a.dir}; {len(failed)} failed (see {a.dir}/llm_log.jsonl)")
    for idea, err in failed:
        print("  FAILED:", idea, "->", err[:160])
    return 0 if not failed else 2


def cmd_llm(a):
    from gghuman.scriptgen import llm_generate
    ep = llm_generate(a.idea)
    d = Path(a.dir); d.mkdir(parents=True, exist_ok=True)
    f = d / f"{ep['id']}.json"; f.write_text(json.dumps(ep, indent=1), encoding="utf-8"); print("wrote", f)
    if a.build:
        from gghuman.pipeline import build_episode
        r = build_episode(ep, a.out, a.workers, a.force, a.crf, a.preset, **_kw(a))
        _summary([r])
        return 0 if r["status"] == "ok" else 2
    return 0


def _job(args):
    spec_path, out, workers, force, crf, preset, kw = args
    from gghuman.pipeline import build_episode
    return build_episode(_load(spec_path), out, workers, force, crf, preset, **kw)


def write_manifest(results, out_dir):
    """publish_queue.csv lists ONLY verified (status ok) videos with their caption: the hand-off to the scheduler.
    Anything degraded / qa_failed / error stays out of it and in batch_report.json for a human."""
    import csv
    rows = []
    for r in results:
        if r["status"] != "ok":
            continue
        d = Path(out_dir) / r["id"]
        cap = (d / "caption.txt").read_text(encoding="utf-8") if (d / "caption.txt").exists() else ""
        rows.append(dict(id=r["id"], video=str(d / f"{r['id']}.mp4"), cover=str(d / "cover.jpg"), caption=cap.replace("\n", " | "), duration_s=round(r.get("duration", 0), 1)))
    with open(Path(out_dir) / "publish_queue.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "video", "cover", "caption", "duration_s"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def cmd_batch(a):
    files = sorted(Path(a.dir).glob("*.json"))
    files = [f for f in files if f.name != "diversity.json"]
    if a.shard:
        i, n = map(int, a.shard.split("/")); files = files[i::n]
    if a.limit:
        files = files[:a.limit]
    res, t0 = [], time.time()
    jobs = [(str(f), a.out, a.workers if a.jobs <= 1 else 1, a.force, a.crf, a.preset, _kw(a)) for f in files]
    if a.jobs > 1:                           # parallel across videos: one process per video, each renders in-process
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            for k, r in enumerate(pool.imap_unordered(_job, jobs), 1):
                res.append(r)
                print(f"[{k}/{len(files)}] {r['id']}: {r['status']}  ({time.time() - t0:.0f}s elapsed)", flush=True)
    else:
        for k, j in enumerate(jobs, 1):
            r = _job(j)
            res.append(r)
            print(f"[{k}/{len(files)}] {r['id']}: {r['status']}  ({time.time() - t0:.0f}s elapsed)", flush=True)
    res.sort(key=lambda r: r["id"])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rep = Path(a.out) / "batch_report.json"
    by = {}
    for r in res:
        by[r["status"]] = by.get(r["status"], 0) + 1
    queued = write_manifest(res, a.out)
    rep.write_text(json.dumps(dict(total=len(res), ok=by.get("ok", 0), by_status=by, queued_for_publishing=queued, wall_seconds=round(time.time() - t0, 1),
                                   failed=[dict(id=r["id"], status=r["status"], why=r.get("error") or r.get("qa", {}).get("failed") or r.get("errors")) for r in res if r["status"] != "ok"]),
                              indent=1, default=str), encoding="utf-8")
    _summary(res)
    print("report:", rep, "| publish queue:", queued, "videos")
    return 0 if all(r["status"] == "ok" for r in res) else 2


def cmd_qa(a):
    from gghuman.qa import run_qa
    tl = json.loads(Path(a.timeline).read_text(encoding="utf-8"))
    stem = Path(a.timeline).parent / "stems" / "voice.wav"
    q, _ = run_qa(Path(a.video), tl, stem, allow_fallback_voice=a.allow_fallback_voice, asr=not a.no_asr, strict_asr=a.strict_asr)
    for c in q["checks"]:
        print(f'{c["status"].upper():5} {c["name"]:42} {str(c["value"]):>22}   expect {c["expect"]}')
    print("PASSED" if q["passed"] else "FAILED")
    return 0 if q["passed"] else 2


def main():
    for stream in (sys.stdout, sys.stderr):      # Windows consoles default to a legacy code page
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("--out", default=str(ROOT / "out"))
        p.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
        p.add_argument("--force", action="store_true", help="ignore cached results")
        p.add_argument("--crf", type=int, default=18)
        p.add_argument("--preset", default="veryfast")
        p.add_argument("--allow-fallback-voice", action="store_true", help="accept espeak/babble voices (otherwise such videos are marked 'degraded')")
        p.add_argument("--no-asr", action="store_true", help="skip the speech-recognition check")
        p.add_argument("--strict-asr", action="store_true", help="fail (instead of warn) when the ASR model is not installed")
    sub.add_parser("doctor").set_defaults(f=cmd_doctor)
    p = sub.add_parser("build"); p.add_argument("specs", nargs="+"); common(p); p.set_defaults(f=cmd_build)
    p = sub.add_parser("all"); common(p); p.set_defaults(f=cmd_all)
    p = sub.add_parser("gen"); p.add_argument("--count", type=int, default=100); p.add_argument("--seed", type=int, default=1); p.add_argument("--dir", default=str(ROOT / "episodes_gen")); p.set_defaults(f=cmd_gen)
    p = sub.add_parser("llm"); p.add_argument("idea"); p.add_argument("--dir", default=str(ROOT / "episodes_llm")); p.add_argument("--build", action="store_true", help="also build the video"); common(p); p.set_defaults(f=cmd_llm)
    p = sub.add_parser("batch"); p.add_argument("dir"); p.add_argument("--shard"); p.add_argument("--limit", type=int); p.add_argument("--jobs", type=int, default=1, help="parallel videos (processes)"); common(p); p.set_defaults(f=cmd_batch)
    p = sub.add_parser("llm-batch"); p.add_argument("ideas"); p.add_argument("--dir", default=str(ROOT / "episodes_llm")); p.set_defaults(f=cmd_llm_batch)
    p = sub.add_parser("qa"); p.add_argument("video"); p.add_argument("timeline"); p.add_argument("--allow-fallback-voice", action="store_true"); p.add_argument("--no-asr", action="store_true"); p.add_argument("--strict-asr", action="store_true"); p.set_defaults(f=cmd_qa)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    sys.exit(a.f(a))


if __name__ == "__main__":
    main()
