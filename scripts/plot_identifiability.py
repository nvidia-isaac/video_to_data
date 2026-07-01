#!/usr/bin/env python3
"""Visualize the sysid parameter ambiguity: the flat (wn,zeta,delay) valley of near-perfect fits.

(a) VAF vs zeta (fix zeta, refit wn & delay) for every joint -> a flat plateau = many equally-good
    solutions.  (b) the wn(zeta) & delay(zeta) trade-off along that plateau (joint j1).  (c) three
    very different family members (zeta=2.0 / 2.85 / 6.0) overlaid on the real high-freq data points
    for j1 -> visually indistinguishable. Pure numpy/scipy/matplotlib.
"""
import glob, os
import numpy as np
from scipy import signal
from scipy.optimize import least_squares
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/home/cning/simtoolreal_isaaclab/logs/sysid/arm_sysid_amp_0.35_dur_60"
OUT = "/home/cning/simtoolreal_isaaclab/logs/sysid/fit/identifiability.png"


def clean(y):
    y = np.asarray(y, float).copy(); nan = np.isnan(y)
    if nan.any():
        i = np.arange(len(y)); y[nan] = np.interp(i[nan], i[~nan], y[~nan])
    return y


def sim2(u, dt, wn, zeta, T):
    n = len(u); t = np.arange(n) * dt
    ud = np.interp(t - T, t, u, left=u[0])
    _, y, _ = signal.lsim(([wn * wn], [1., 2 * zeta * wn, wn * wn]), U=ud, T=t)
    return y


def fit_fixed_zeta(u, y, dt, zeta):
    r = least_squares(lambda p: sim2(u, dt, p[0], zeta, p[1]) - y, (2 * np.pi, .01),
                      bounds=([2 * np.pi * .05, 0.], [2 * np.pi * 20, .1]), max_nfev=2000)
    wn, T = r.x
    return wn, T, 100. * (1 - np.var(y - sim2(u, dt, wn, zeta, T)) / np.var(y))


def load():
    out = {}
    for p in sorted(glob.glob(os.path.join(DATA, "L_arm_j*.npz"))):
        d = np.load(p, allow_pickle=True); nm = str(d["joint_name"]); base = float(d["probe_center"])
        u = np.asarray(d["q_cmd"], float) - base; y = clean(np.asarray(d["q_state"], float)) - base
        if np.std(y) < 0.05 * np.std(u):
            continue
        out[nm] = (u, y, 1. / float(d["rate"]), base)
    return out


def main():
    data = load()
    zs = np.linspace(1.0, 6.0, 16)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

    # (a) VAF vs zeta, all joints
    for nm, (u, y, dt, base) in data.items():
        vafs = [fit_fixed_zeta(u, y, dt, z)[2] for z in zs]
        ax[0].plot(zs, vafs, "o-", ms=3, label=nm)
    ax[0].set_xlabel("zeta (fixed)"); ax[0].set_ylabel("best VAF % (refit wn, delay)")
    ax[0].set_title("(a) flat valley: VAF ~ const for zeta >= ~1.5"); ax[0].set_ylim(98.5, 100.02)
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7, ncol=2)

    # (b) wn & delay trade-off along the valley (j1)
    nm0 = "L_arm_j1"; u, y, dt, base = data[nm0]
    wns, Ts = [], []
    for z in zs:
        wn, T, _ = fit_fixed_zeta(u, y, dt, z); wns.append(wn / (2 * np.pi)); Ts.append(1e3 * T)
    ax[1].plot(zs, wns, "s-", color="tab:blue", label="wn [Hz]")
    ax[1].set_xlabel("zeta"); ax[1].set_ylabel("wn [Hz]", color="tab:blue"); ax[1].tick_params(axis="y", labelcolor="tab:blue")
    axb = ax[1].twinx(); axb.plot(zs, Ts, "^-", color="tab:red", label="delay [ms]")
    axb.set_ylabel("delay [ms]", color="tab:red"); axb.tick_params(axis="y", labelcolor="tab:red")
    ax[1].set_title(f"(b) {nm0}: wn & delay slide together along the valley"); ax[1].grid(alpha=0.3)

    # (c) three family members overlaid on real high-freq data (j1)
    N = len(u); w = int(6.0 / dt); sl = slice(N - w, N); t = np.arange(N) * dt
    ax[2].plot(t[sl], y[sl] + base, "o", ms=3, color="0.4", label="real q_state", zorder=5)
    for z, col in [(2.0, "tab:green"), (2.85, "tab:blue"), (6.0, "tab:red")]:
        wn, T, v = fit_fixed_zeta(u, y, dt, z)
        ym = sim2(u, dt, wn, z, T)
        ax[2].plot(t[sl], ym[sl] + base, "-", lw=1.3, color=col,
                   label=f"zeta={z}, wn={wn/(2*np.pi):.2f}Hz, {1e3*T:.0f}ms (VAF {v:.2f}%)")
    ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("rad")
    ax[2].set_title(f"(c) {nm0}: 3 very different params, same fit (high-freq end)")
    ax[2].grid(alpha=0.3); ax[2].legend(fontsize=7, loc="upper right")

    fig.suptitle("Parameter ambiguity: a 1-D family of (wn, zeta, delay) fits the chirp near-perfectly", y=1.0)
    fig.tight_layout(); fig.savefig(OUT, dpi=130)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
