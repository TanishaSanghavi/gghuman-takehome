"""Episode build pipeline: validate -> timeline(TTS) -> mix -> render (parallel chunks)
-> mux -> QA -> auto-fix/retry -> artefacts.  Never raises: always returns a result dict."""
import hashlib, json, logging, multiprocessing as mp, shutil, time, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .config import FPS, W, H
from .spec import validate
from . import encode

log = logging.getLogger("gghuman.pipeline")
_R = None


# ---------------------------------------------------------------- render workers
PIPELINE_VERSION = "3"      # bump when checks change: invalidates cached "ok" results so old videos are re-verified


def _init_worker(data, envs, rspec):
    global _R
    from .render import Renderer
    tl = SimpleNamespace(data=data, clips={k: SimpleNamespace(env=v) for k, v in envs.items()})
    _R = Renderer(tl, rspec)


def _render_chunk(args):
    idx, f0, f1, path, crf, preset = args
    last, fails = None, 0
    for attempt in range(2):
        try:
            w = encode.open_writer(path, crf, preset)
            fails = 0
            for fi in range(f0, f1):
                try:
                    fr = _R.frame(fi)
                    last = fr
                except Exception:  # hold previous frame rather than lose the video
                    fails += 1
                    log.error("frame %d failed:\n%s", fi, traceback.format_exc())
                    fr = last if last is not None else np.zeros((H, W, 3), np.uint8)
                w.stdin.write(np.ascontiguousarray(fr).tobytes())
            w.stdin.close()
            if w.wait() != 0:
                raise RuntimeError("ffmpeg chunk encode failed")
            return idx, fails
        except Exception:
            log.error("chunk %d attempt %d failed:\n%s", idx, attempt, traceback.format_exc())
    return idx, 10 ** 6


def render_video(tl, spec, out_mp4, workdir, workers=1, crf=18, preset="veryfast", chunk_s=4.0, progress=True):
    n = int(round(tl.duration * FPS))
    step = max(FPS, int(chunk_s * FPS))
    bounds = [(i, a, min(n, a + step)) for i, a in enumerate(range(0, n, step))]
    workdir.mkdir(parents=True, exist_ok=True)
    jobs = [(i, a, b, str(workdir / f"chunk_{i:03d}.mp4"), crf, preset) for i, a, b in bounds]
    envs = {k: c.env for k, c in tl.clips.items()}
    rspec = {"seats": spec.get("seats"), "board_seed": spec.get("board_seed", 0)}
    total_fail, done = 0, 0
    t0 = time.time()
    if workers <= 1:                                   # in-process: portable, no multiprocessing
        _init_worker(tl.data, envs, rspec)
        results = (_render_chunk(j) for j in jobs)
        for idx, fails in results:
            total_fail += fails
            done += 1
            if progress:
                log.info("  rendered chunk %d/%d (%.0fs elapsed)", done, len(jobs), time.time() - t0)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(tl.data, envs, rspec)) as pool:
            for idx, fails in pool.imap_unordered(_render_chunk, jobs):
                total_fail += fails
                done += 1
                if progress:
                    log.info("  rendered chunk %d/%d (%.0fs elapsed)", done, len(jobs), time.time() - t0)
    if total_fail > max(2, 0.01 * n):
        raise RuntimeError(f"{total_fail} frames failed to render (>1%)")
    encode.concat([j[3] for j in jobs], out_mp4)
    for j in jobs:
        Path(j[3]).unlink(missing_ok=True)
    return dict(frames=n, failed_frames=total_fail, seconds=round(time.time() - t0, 1))


# ---------------------------------------------------------------- build
def spec_hash(spec):
    return hashlib.sha1((PIPELINE_VERSION + json.dumps(spec, sort_keys=True)).encode()).hexdigest()[:12]


def build_episode(spec, out_root, workers=1, force=False, crf=18, preset="veryfast", max_attempts=2, keep_work=False,
                  allow_fallback_voice=False, asr=True, strict_asr=False, retries=1):
    """Build one video.  Never raises.  An unexpected crash (ffmpeg hiccup, out-of-memory kill of a worker, ...) is retried
    `retries` times from scratch; validation failures and QA failures are deterministic and are NOT retried."""
    res = None
    for k in range(retries + 1):
        res = _build_once(spec, out_root, workers, force or k > 0, crf, preset, max_attempts, keep_work, allow_fallback_voice, asr, strict_asr)
        if res["status"] != "error":
            break
        log.warning("episode %s crashed (%s); retry %d/%d", res.get("id"), res.get("error"), k + 1, retries)
    res["build_tries"] = k + 1
    return res


def _build_once(spec, out_root, workers, force, crf, preset, max_attempts, keep_work, allow_fallback_voice, asr, strict_asr):
    from .timeline import compile_episode
    from .mixer import mix
    from .qa import run_qa
    from PIL import Image
    out_root = Path(out_root)
    eid = spec.get("id", "unnamed")
    out = out_root / eid
    out.mkdir(parents=True, exist_ok=True)
    h = spec_hash(spec)
    res_path = out / "result.json"
    if res_path.exists() and not force:
        try:
            old = json.loads(res_path.read_text(encoding="utf-8"))
            if old.get("spec_hash") == h and old.get("status") == "ok" and (out / f"{eid}.mp4").exists():
                old["cached"] = True
                return old
        except Exception:
            pass
    res = dict(id=eid, spec_hash=h, status="error", stages={}, attempts=0)
    t_all = time.time()
    try:
        errs = validate(spec)
        if errs:
            res.update(status="invalid_spec", errors=errs)
            return _finish(res, res_path, t_all)
        t = time.time()
        tl = compile_episode(spec)
        tl.dump(out / "timeline.json")
        res["stages"]["timeline_tts"] = round(time.time() - t, 1)
        res["tts_backends"] = sorted({c.backend for c in tl.clips.values()})
        res["duration"] = tl.duration
        target_lufs, mux_offset = -14.0, 0.0
        import gghuman.mixer as _M
        _M.CEILING = _M.DEFAULT_CEILING      # per-episode reset (auto-fixes must not leak into the next video)
        vid = out / "video_silent.mp4"
        t = time.time()
        res["render"] = render_video(tl, spec, vid, out / "_chunks", workers, crf, preset)
        res["stages"]["render"] = round(time.time() - t, 1)
        final = out / f"{eid}.mp4"
        for attempt in range(1, max_attempts + 1):
            res["attempts"] = attempt
            t = time.time()
            import gghuman.mixer as M
            M.TARGET_LUFS = target_lufs
            res["mix"] = mix(tl, out / "audio.wav", out / "stems")
            encode.mux(vid, out / "audio.wav", final, offset=mux_offset)
            res["stages"][f"mix_mux_{attempt}"] = round(time.time() - t, 1)
            t = time.time()
            qa, first = run_qa(final, tl.data, out / "stems" / "voice.wav", allow_fallback_voice=allow_fallback_voice, asr=asr, strict_asr=strict_asr)
            res["stages"][f"qa_{attempt}"] = round(time.time() - t, 1)
            (out / "qa.json").write_text(json.dumps(qa, indent=1, default=str), encoding="utf-8")
            if qa["passed"] or not qa["fixes"]:
                break
            log.warning("QA failed (%s); applying auto-fix %s", [c["name"] for c in qa["checks"] if c["status"] == "fail"], qa["fixes"])
            if "gain_db" in qa["fixes"]:
                target_lufs = -14.0 + qa["fixes"]["gain_db"] * 0.0 + qa["fixes"]["gain_db"]
                target_lufs = max(-16.0, min(-12.0, target_lufs))
            if "ceiling_scale" in qa["fixes"]:
                import gghuman.mixer as M2
                M2.CEILING *= qa["fixes"]["ceiling_scale"]
            if "mux_offset_s" in qa["fixes"]:
                mux_offset += qa["fixes"]["mux_offset_s"]
        res["qa"] = dict(passed=qa["passed"], n_pass=qa["n_pass"], n_warn=qa["n_warn"], n_fail=qa["n_fail"], lufs=round(qa["lufs"], 2),
                         failed=[c["name"] for c in qa["checks"] if c["status"] == "fail"], warnings=[c["name"] for c in qa["checks"] if c["status"] == "warn"])
        # artefacts
        Image.fromarray(first[..., ::-1]).save(out / "cover.jpg", quality=92)
        try:
            from .viz import sync_report
            sync_report(final, tl.data, out / "sync_report.png")
        except Exception as e:  # report is a nicety, never fail the episode for it
            log.warning("sync report skipped: %s", e)
        (out / "caption.txt").write_text(spec.get("caption", spec.get("title", "")), encoding="utf-8")
        asr_chk = next((c for c in qa["checks"] if c["name"] == "speech_intelligible_asr" and c.get("lines")), None)
        if asr_chk:
            (out / "asr_transcript.json").write_text(json.dumps(asr_chk["lines"], indent=1), encoding="utf-8")
        res["video"] = str(final)
        failed_names = [c["name"] for c in qa["checks"] if c["status"] == "fail"]
        if qa["passed"]:
            res["status"] = "ok"
        elif "neural_voice_used" in failed_names:
            res["status"] = "degraded"       # built with a fallback voice: never published automatically
        else:
            res["status"] = "qa_failed"
        if not keep_work:
            vid.unlink(missing_ok=True)
            shutil.rmtree(out / "_chunks", ignore_errors=True)
    except Exception as e:
        res.update(status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
        log.error("episode %s failed: %s", eid, e)
    return _finish(res, res_path, t_all)


def _finish(res, path, t0):
    res["total_seconds"] = round(time.time() - t0, 1)
    Path(path).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return res
