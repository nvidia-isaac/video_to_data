#!/usr/bin/env python3
"""Zoom into the STEP EDGE (first ~250 ms) -- the high-frequency part that carries the fast-pole +
delay info -- and measure the fit there, not just globally.

Global VAF is dominated by the ~0.8 s slow rise; it can hide an edge mismatch. Here we:
  - overlay the current chirp-fit + step-refit + two fast-pole family members (zeta=2 vs zeta=6,
    each refit to the step) on the real onset data points,
  - report VAF computed ONLY on the 250 ms edge window,
so we can see (a) does the current fit match at the edge? (b) do different fast poles diverge there,
i.e. does the step actually resolve the ambiguity? Pure numpy/scipy/matplotlib.
"""
import json, os
import numpy as np
from scipy import signal
from scipy.optimize import least_squares
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/cning/simtoolreal_isaaclab/logs/sysid"
fit = json.load(open(f"{ROOT}/fit/fit_params.json"))


def clean(y):
    y = np.asarray(y, float).copy(); n = np.isnan(y)
    if n.any():
        i = np.arange(len(y)); y[n] = np.interp(i[n], i[~n], y[~n])
    return y


def sim2(u, dt, wn, zeta, T):
    t = np.arange(len(u)) * dt
    ud = np.interp(t - T, t, u, left=u[0])
    _, y, _ = signal.lsim(([wn * wn], [1., 2 * zeta * wn, wn * wn]), U=ud, T=t)
    return y


def vaf_win(y, ym, sl):
    e = (y - ym)[sl]
    return 100. * (1 - np.var(e) / np.var(y[sl]))


def refit(u, y, dt, zeta=None):
    if zeta is None:
        r = least_squares(lambda p: sim2(u, dt, *p) - y, (2 * np.pi, 1., .02),
                          bounds=([.3, .05, 0.], [200, 8, .15]), max_nfev=4000)
        return r.x
    r = least_squares(lambda p: sim2(u, dt, p[0], zeta, p[1]) - y, (2 * np.pi, .02),
                      bounds=([.3, 0.], [200, .15]), max_nfev=4000)
    return r.x[0], zeta, r.x[1]


joints = [f"L_arm_j{i}" for i in range(1, 7)]
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
print(f"{'joint':10s} {'edge-VAF cur':>12s} {'refit':>7s} {'z=2':>7s} {'z=6':>7s}   "
      f"onset-delay(data vs curfit)")
for k, j in enumerate(joints):
    p = f"{ROOT}/SystemID/1 step/arm_step_{j.split('_')[-1]}/{j}.npz"
    d = np.load(p, allow_pickle=True)
    dt = 1. / float(d["rate"]); base = float(d["probe_center"])
    u = np.asarray(d["q_cmd"], float) - base
    y = clean(np.asarray(d["q_state"], float)) - base
    t = np.arange(len(u)) * dt
    step_i = int(np.argmax(np.abs(u) > 0.5 * np.max(np.abs(u))))   # first big command move
    edge = slice(step_i, step_i + 125)                            # 250 ms window
    fp = fit[j]
    cur = sim2(u, dt, fp["wn_rad"], fp["zeta"], fp["delay_s"])
    wr, zr, Tr = refit(u, y, dt); yr = sim2(u, dt, wr, zr, Tr)
    w2, _, T2 = refit(u, y, dt, 2.0); y2 = sim2(u, dt, w2, 2.0, T2)
    w6, _, T6 = refit(u, y, dt, 6.0); y6 = sim2(u, dt, w6, 6.0, T6)

    # data onset delay: first sample past step_i exceeding 2% of final move
    fin = np.median(y[step_i + 400:step_i + 800]) if step_i + 800 < len(y) else y[-1]
    thr = step_i + np.argmax(np.abs(y[step_i:]) > 0.02 * abs(fin))
    data_delay = (thr - step_i) * dt
    print(f"{j:10s} {vaf_win(y,cur,edge):12.2f} {vaf_win(y,yr,edge):7.2f} "
          f"{vaf_win(y,y2,edge):7.2f} {vaf_win(y,y6,edge):7.2f}   "
          f"{1e3*data_delay:.0f}ms vs {1e3*fp['delay_s']:.0f}ms")

    a = ax[k // 3, k % 3]
    t0 = t[step_i]
    win = slice(step_i - 25, step_i + 150)
    a.plot((t[win] - t0) * 1e3, u[win], color="0.7", lw=1.0, label="q_cmd")
    a.plot((t[win] - t0) * 1e3, y[win], "o", ms=3, color="tab:orange", label="real", zorder=5)
    a.plot((t[win] - t0) * 1e3, cur[win], "-", lw=1.6, color="tab:blue", label="current fit (chirp)")
    a.plot((t[win] - t0) * 1e3, y2[win], "--", lw=1.1, color="tab:green", label=f"z=2 (fast p {w2*(2-np.sqrt(3))/(2*np.pi):.1f}Hz)")
    a.plot((t[win] - t0) * 1e3, y6[win], ":", lw=1.4, color="tab:red", label=f"z=6 (fast p {w6*(6+np.sqrt(35))/(2*np.pi):.1f}Hz)")
    a.set_title(f"{j}: edge-VAF cur={vaf_win(y,cur,edge):.1f}%", fontsize=9)
    a.set_xlabel("ms after step"); a.set_ylabel("rad (rel)"); a.grid(alpha=0.3)
    if k == 0:
        a.legend(fontsize=6.5, loc="lower right")
fig.suptitle("Step EDGE zoom: does the fit match the high-frequency onset? (250 ms after step)", y=1.0)
fig.tight_layout(); fig.savefig(f"{ROOT}/fit/step_edge.png", dpi=130)
print(f"\nsaved -> {ROOT}/fit/step_edge.png")
