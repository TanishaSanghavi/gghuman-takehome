"""sync_report.png: audio waveform + every scheduled event on one time axis."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import SR
from . import encode


def sync_report(final_mp4, tl, out_png):
    x = encode.decode_audio(final_mp4, SR)
    d = tl["meta"]["duration"]
    t = np.arange(len(x)) / SR
    k = 240
    env = np.abs(x[:len(x) // k * k]).reshape(-1, k).max(axis=1)
    te = t[:len(env) * k:k]
    fig, ax = plt.subplots(5, 1, figsize=(15, 8.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1, 1, 1, 1], hspace=0.08))
    ax[0].fill_between(te, -env, env, color="#3b6fb6", lw=0)
    ax[0].set_ylabel("final mix"); ax[0].set_ylim(-1, 1)
    for a, b in tl["mute"]:
        ax[0].axvspan(a, b, color="#d33", alpha=.25)
        ax[0].text(a, .8, "SILENCE", color="#a00", fontsize=8)
    for r in tl["reveals"]:
        ax[0].axvline(r["t"], color="#d33", lw=2)
        ax[0].text(r["t"] + .05, -.9, "REVEAL", color="#a00", fontsize=9)
    ax[0].set_title(f'{tl["meta"]["id"]}  –  one timeline drives picture, voice, captions, sfx, music', fontsize=11)
    for s in tl["shots"]:
        for a in ax:
            a.axvline(s["t0"], color="#999", lw=.5, ls=":")
        ax[0].text(s["t0"] + .03, .88, s["id"], fontsize=6.5, rotation=0, color="#444")
    cols = {"wren": "#58b42a", "red": "#e8382b", "gpt": "#3e9c8e", "claude": "#cf7a58", "gemini": "#5a64d6"}
    for sp in tl["speech"]:
        ax[1].barh(0, sp[1] - sp[0], left=sp[0], color=cols.get(sp[2], "#888"), height=.7)
        ax[1].text(sp[0], 0, sp[2][:2].upper(), fontsize=7, va="center", color="white")
    ax[1].set_ylabel("voice"); ax[1].set_yticks([])
    for c in tl["captions"]:
        ax[2].barh(0, c["t1"] - c["t0"], left=c["t0"], color="#222", height=.7)
        ax[2].text(c["t0"], 0, c["text"], fontsize=6, va="center", color="white", clip_on=True)
    ax[2].set_ylabel("captions"); ax[2].set_yticks([])
    for e in tl["sfx"]:
        ax[3].plot([e["t"], e["t"]], [0, 1], color="#e0a000", lw=1.5)
        ax[3].text(e["t"], 1.02, e["name"], fontsize=5.5, rotation=90, va="bottom")
    ax[3].set_ylabel("sfx"); ax[3].set_yticks([]); ax[3].set_ylim(0, 1.9)
    for s in tl["shots"]:
        ax[4].barh(0, s["energy"] and (s["t1"] - s["t0"]), left=s["t0"], color=plt.cm.magma(s["energy"]), height=.8)
    for a, b in tl["risers"]:
        ax[4].plot([a, b], [0.55, 0.55], color="cyan", lw=3)
    ax[4].set_ylabel("music\nenergy"); ax[4].set_yticks([]); ax[4].set_xlim(0, d); ax[4].set_xlabel("seconds")
    fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
