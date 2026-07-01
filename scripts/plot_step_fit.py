#!/usr/bin/env python3
"""Cross-validate the chirp-derived arm fit against the NEW step-response data.

Overlays our CURRENT parameter fit (logs/sysid/fit/fit_params.json: wn, zeta, delay -- fit from the
0.1->2 Hz chirp) onto the new 500 Hz step-response data points (logs/sysid/SystemID/.../L_arm_j*.npz),
per joint. A step is broadband, so it excites the ~6 Hz fast pole the chirp missed -> this both tests
generalization AND shows whether the step prefers different params (the ambiguity we flagged). For
context each panel also draws the best-fit-to-THIS-step (dashed).

Figures in logs/sysid/fit/:  step_fit_single.png (clean single step, zoom on transient),
step_fit_multi.png (30 s multi-step).  Pure numpy/scipy/matplotlib.
"""
import glob, json, os
import numpy as np
from scipy import signal
from scipy.optimize import least_squares
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/cning/simtoolreal_isaaclab/logs/sysid"
FITJSON = f"{ROOT}/fit/fit_params.json"


def clean(y):
    y = np.asarray(y, float).copy(); nan = np.isnan(y)
    if nan.any():
        i = np.arange(len(y)); y[nan] = np.interp(i[nan], i[~nan], y[~nan])
    return y


def sim2(u, dt, wn, zeta, T):
    t = np.arange(len(u)) * dt
    ud = np.interp(t - T, t, u, left=u[0])
    _, y, _ = signal.lsim(([wn * wn], [1., 2 * zeta * wn, wn * wn]), U=ud, T=t)
    return y


def vaf(y, ym):
    return 100. * (1 - np.var(y - ym) / np.var(y))


def refit(u, y, dt):
    r = least_squares(lambda p: sim2(u, dt, *p) - y, (2 * np.pi, 1.0, 0.02),
                      bounds=([2 * np.pi * .05, .05, 0.], [2 * np.pi * 30, 8., .15]), max_nfev=4000)
    return r.x


def load_step(joint, subdir):
    short = joint.split("_")[-1]                     # "L_arm_j1" -> "j1"  (dirs are arm_step_j1)
    p = os.path.join(ROOT, "SystemID", subdir, f"arm_step_{short}", f"{joint}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    return dict(cmd=np.asarray(d["q_cmd"], float), real=clean(np.asarray(d["q_state"], float)),
                base=float(d["probe_center"]), dt=1.0 / float(d["rate"]))


def main():
    fit = json.load(open(FITJSON))
    joints = [f"L_arm_j{i}" for i in range(1, 7)]  # j7 has no chirp fit

    for subdir, fname, zoom in [("1 step", "step_fit_single.png", True),
                                ("", "step_fit_multi.png", False)]:
        panels = [(j, load_step(j, subdir)) for j in joints]
        panels = [(j, s) for j, s in panels if s is not None]
        n = len(panels)
        if n == 0:
            continue
        fig, ax = plt.subplots(n, 1, figsize=(12, 2.0 * n), squeeze=False)
        print(f"\n=== {subdir or 'multi-step'} ===\n{'joint':10s} {'cur-fit VAF':>11s} {'refit VAF':>10s}"
              f"   refit(wn,zeta,delay)")
        for i, (j, s) in enumerate(panels):
            dt = s["dt"]; u = s["cmd"] - s["base"]; y = s["real"] - s["base"]
            t = np.arange(len(u)) * dt
            fp = fit[j]
            ycur = sim2(u, dt, fp["wn_rad"], fp["zeta"], fp["delay_s"])
            vcur = vaf(y, ycur)
            wn2, z2, T2 = refit(u, y, dt); yref = sim2(u, dt, wn2, z2, T2); vref = vaf(y, yref)
            print(f"{j:10s} {vcur:11.2f} {vref:10.2f}   wn={wn2/(2*np.pi):.2f}Hz zeta={z2:.2f} "
                  f"delay={1e3*T2:.0f}ms  (chirp: wn={fp['wn_hz']:.2f} zeta={fp['zeta']:.2f} "
                  f"delay={1e3*fp['delay_s']:.0f}ms)")
            sub = slice(None, None, 3 if zoom else 15)
            ax[i, 0].plot(t, u + s["base"], color="0.75", lw=1.0, label="q_cmd (step)")
            ax[i, 0].plot(t[sub], y[sub] + s["base"], "o", ms=2.6, color="tab:orange", label="real q_state", zorder=4)
            ax[i, 0].plot(t, ycur + s["base"], "-", lw=1.5, color="tab:blue",
                          label=f"current fit (chirp)  VAF={vcur:.1f}%", zorder=3)
            ax[i, 0].plot(t, yref + s["base"], "--", lw=1.2, color="tab:green",
                          label=f"refit to this step  VAF={vref:.1f}%", zorder=2)
            ax[i, 0].set_ylabel(f"{j}\n[rad]"); ax[i, 0].grid(alpha=0.3)
            if zoom:
                ax[i, 0].set_xlim(0.3, min(t[-1], 4.5))
            if i == 0:
                ax[i, 0].legend(fontsize=7, loc="lower right", ncol=2)
        ax[-1, 0].set_xlabel("time [s]")
        fig.suptitle(f"current chirp-fit vs NEW step data ({subdir or '30 s multi-step'}) — "
                     f"amplitude 0.1 rad, 500 Hz", y=1.0)
        fig.tight_layout(); fig.savefig(f"{ROOT}/fit/{fname}", dpi=130)
        print(f"saved -> {ROOT}/fit/{fname}")


if __name__ == "__main__":
    main()
