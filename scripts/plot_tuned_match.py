#!/usr/bin/env python3
"""Plot the SYSID-tuned IsaacSim arm vs the real-world chirp data points.

Reads logs/sysid/fit/tuned_traj.npz (from replay_tuned_arm.py): per joint q_cmd / q_real / q_sim.
Makes three figures in logs/sysid/fit/:
  tuned_match_time.png  -- full 60 s overlay: real as POINTS, tuned sim as line, cmd faint
  tuned_match_zoom.png  -- two zoom windows per joint (low-freq start, high-freq end)
  tuned_match_bode.png  -- empirical frequency response, real vs tuned sim (gain + phase)
Pure numpy/scipy/matplotlib (isaaclab venv).
"""

import argparse
import os

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def clean(y):
    y = np.asarray(y, float).copy()
    nan = np.isnan(y)
    if nan.any():
        idx = np.arange(len(y))
        y[nan] = np.interp(idx[nan], idx[~nan], y[~nan])
    return y


def metrics(real, sim):
    e = sim - real
    rms = float(np.sqrt(np.mean(e ** 2)))
    vaf = float(100.0 * (1.0 - np.var(e) / np.var(real)))
    return rms, vaf


def frf(u, y, dt, nperseg=2048):
    fs = 1.0 / dt
    f, Puu = signal.welch(u, fs=fs, nperseg=nperseg)
    _, Puy = signal.csd(u, y, fs=fs, nperseg=nperseg)
    H = Puy / Puu
    return f, 20 * np.log10(np.abs(H)), np.degrees(np.angle(H))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", default="/home/cning/simtoolreal_isaaclab/logs/sysid/fit/tuned_traj.npz")
    ap.add_argument("--out", default="/home/cning/simtoolreal_isaaclab/logs/sysid/fit")
    args = ap.parse_args()

    d = np.load(args.traj)
    dt = float(d["dt"])
    joints = sorted({k.rsplit("_", 1)[0] for k in d.files if k.endswith("_sim")})
    n = len(joints)
    t = None

    # ---- Fig 1: full overlay, real as points ----------------------------------------------------
    fig, ax = plt.subplots(n, 1, figsize=(12, 2.0 * n), sharex=True, squeeze=False)
    for i, j in enumerate(joints):
        cmd, real, sim = d[f"{j}_cmd"], clean(d[f"{j}_real"]), d[f"{j}_sim"]
        t = np.arange(len(cmd)) * dt
        rms, vaf = metrics(real, sim)
        sub = slice(None, None, 12)  # subsample real points so they're visible
        ax[i, 0].plot(t, cmd, color="0.7", lw=0.8, label="q_cmd (command)", zorder=1)
        ax[i, 0].plot(t[sub], real[sub], "o", ms=2.4, color="tab:orange", label="real q_state", zorder=3)
        ax[i, 0].plot(t, sim, "-", lw=1.3, color="tab:blue", label="tuned sim", zorder=2)
        ax[i, 0].set_ylabel(f"{j}\n[rad]")
        ax[i, 0].grid(alpha=0.3)
        ax[i, 0].text(0.995, 0.04, f"RMS={1e3*rms:.1f} mrad   VAF={vaf:.1f}%",
                      ha="right", va="bottom", transform=ax[i, 0].transAxes, fontsize=8,
                      bbox=dict(fc="white", ec="0.8", alpha=0.8))
        if i == 0:
            ax[i, 0].legend(loc="upper right", fontsize=8, ncol=3)
    ax[-1, 0].set_xlabel("time [s]  (chirp 0.1 -> 2 Hz)")
    fig.suptitle("SYSID-tuned IsaacSim arm vs real data — full 60 s chirp", y=0.999)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "tuned_match_time.png"), dpi=130)

    # ---- Fig 2: zoom windows (low-freq start, high-freq end) ------------------------------------
    N = len(d[f"{joints[0]}_cmd"])
    w = int(6.0 / dt)                  # 6 s window
    wins = [(0, w, "start (~0.1-0.3 Hz)"), (N - w, N, "end (~1.7-2 Hz)")]
    fig2, ax2 = plt.subplots(n, 2, figsize=(13, 1.9 * n), squeeze=False)
    for i, j in enumerate(joints):
        cmd, real, sim = d[f"{j}_cmd"], clean(d[f"{j}_real"]), d[f"{j}_sim"]
        t = np.arange(len(cmd)) * dt
        for c, (a, b, lab) in enumerate(wins):
            sl = slice(a, b)
            ax2[i, c].plot(t[sl], cmd[sl], color="0.75", lw=0.9, label="q_cmd")
            ax2[i, c].plot(t[sl], real[sl], "o", ms=2.6, color="tab:orange", label="real q_state")
            ax2[i, c].plot(t[sl], sim[sl], "-", lw=1.4, color="tab:blue", label="tuned sim")
            ax2[i, c].grid(alpha=0.3)
            if i == 0:
                ax2[i, c].set_title(f"chirp {lab}")
                if c == 0:
                    ax2[i, c].legend(fontsize=7, loc="upper right")
        ax2[i, 0].set_ylabel(f"{j}\n[rad]")
    for c in range(2):
        ax2[-1, c].set_xlabel("time [s]")
    fig2.suptitle("tuned sim vs real — zoom (low-freq start vs high-freq end)", y=0.999)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.out, "tuned_match_zoom.png"), dpi=130)

    # ---- Fig 3: Bode, real vs tuned sim ---------------------------------------------------------
    fig3, ax3 = plt.subplots(n, 2, figsize=(12, 1.9 * n), squeeze=False)
    for i, j in enumerate(joints):
        base = np.mean(d[f"{j}_cmd"])
        u = d[f"{j}_cmd"] - base
        yr = clean(d[f"{j}_real"]) - base
        ys = d[f"{j}_sim"] - base
        f, gr, pr = frf(u, yr, dt)
        _, gs, ps = frf(u, ys, dt)
        band = (f >= 0.1) & (f <= 2.0)
        ax3[i, 0].semilogx(f[band], gr[band], "o-", ms=3, color="tab:orange", label="real")
        ax3[i, 0].semilogx(f[band], gs[band], "-", color="tab:blue", label="tuned sim")
        ax3[i, 0].set_ylabel(f"{j}\n|H| [dB]"); ax3[i, 0].grid(alpha=0.3, which="both")
        ax3[i, 1].semilogx(f[band], pr[band], "o-", ms=3, color="tab:orange")
        ax3[i, 1].semilogx(f[band], ps[band], "-", color="tab:blue")
        ax3[i, 1].set_ylabel("phase [deg]"); ax3[i, 1].grid(alpha=0.3, which="both")
        if i == 0:
            ax3[i, 0].set_title("gain"); ax3[i, 1].set_title("phase"); ax3[i, 0].legend(fontsize=8)
    ax3[-1, 0].set_xlabel("Hz"); ax3[-1, 1].set_xlabel("Hz")
    fig3.suptitle("frequency response: real vs tuned sim (excited band 0.1-2 Hz)", y=0.999)
    fig3.tight_layout()
    fig3.savefig(os.path.join(args.out, "tuned_match_bode.png"), dpi=130)

    print("saved -> tuned_match_time.png, tuned_match_zoom.png, tuned_match_bode.png")
    print(f"\n{'joint':10s} {'RMS[mrad]':>10s} {'VAF%':>7s}")
    for j in joints:
        rms, vaf = metrics(clean(d[f"{j}_real"]), d[f"{j}_sim"])
        print(f"{j:10s} {1e3*rms:10.2f} {vaf:7.2f}")


if __name__ == "__main__":
    main()
