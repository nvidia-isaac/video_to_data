#!/usr/bin/env python3
"""Fit IsaacSim implicit-PD-equivalent dynamics to the real arm sysid chirp data.

The real harmonic-drive joint runs a cascaded (position->velocity->torque) PID. IsaacSim's
ImplicitActuator is a single torque-level PD: tau = Kp(q_des - q) - Kd*qdot, which on a rigid
joint of effective inertia I is EXACTLY a 2nd-order low-pass from q_des -> q:

    H(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2) * exp(-s*T)     (DC gain 1)
    wn^2     = Kp / I        2*zeta*wn = Kd / I

So fitting (wn, zeta, T) to each joint's measured q_cmd -> q_state tells us:
  (a) how well ANY implicit-PD setting can match the real joint (the fit residual is the floor), and
  (b) the TARGET wn/zeta/delay we then convert to stiffness/damping given I (in fit step 2/sim).

Per joint we fit (wn, zeta, T) by simulating the model on the command and minimizing the time-domain
error, and we also estimate the empirical frequency response (Welch CSD) for an honest Bode overlay.

Pure numpy/scipy/matplotlib -- runs in the isaaclab venv, no Isaac needed.
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import signal
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_joint(path):
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["name"] = str(d["joint_name"])
    return out


def clean(y):
    """Linear-interpolate NaNs over their indices; return (y_clean, valid_mask)."""
    y = np.asarray(y, float).copy()
    nan = np.isnan(y)
    if nan.any():
        idx = np.arange(len(y))
        y[nan] = np.interp(idx[nan], idx[~nan], y[~nan])
    return y, ~nan


def simulate(u, dt, wn, zeta, T):
    """Simulate H(s)=wn^2/(s^2+2 zeta wn s+wn^2) on input u (delayed by T seconds, ZOH/linear)."""
    n = len(u)
    t = np.arange(n) * dt
    # fractional input delay via linear interpolation (hold first sample for t<T)
    td = t - T
    u_del = np.interp(td, t, u, left=u[0])
    num = [wn * wn]
    den = [1.0, 2.0 * zeta * wn, wn * wn]
    _, y, _ = signal.lsim((num, den), U=u_del, T=t)
    return y


def fit_joint(u, y, valid, dt):
    """Least-squares fit of (wn, zeta, T) minimizing y_model - y over valid samples."""
    w = valid.astype(float)

    def resid(p):
        wn, zeta, T = p
        ym = simulate(u, dt, wn, zeta, T)
        return (ym - y) * w

    # init: wn ~ 2*pi*1 Hz, slightly underdamped, ~1 control step delay
    p0 = [2 * np.pi * 1.0, 0.7, 0.01]
    lb = [2 * np.pi * 0.05, 0.05, 0.0]
    ub = [2 * np.pi * 20.0, 3.0, 0.1]
    res = least_squares(resid, p0, bounds=(lb, ub), method="trf", max_nfev=2000)
    wn, zeta, T = res.x
    ym = simulate(u, dt, wn, zeta, T)
    e = (ym - y)[valid]
    yv = y[valid]
    rms = float(np.sqrt(np.mean(e ** 2)))
    vaf = float(100.0 * (1.0 - np.var(e) / np.var(yv)))  # variance accounted for
    return dict(wn=float(wn), zeta=float(zeta), T=float(T), rms=rms, vaf=vaf, y_model=ym)


def empirical_frf(u, y, dt, nperseg=2048):
    """Empirical FRF H = Puy/Puu via Welch cross-spectrum; returns f[Hz], gain, phase[deg]."""
    fs = 1.0 / dt
    f, Puu = signal.welch(u, fs=fs, nperseg=nperseg)
    _, Puy = signal.csd(u, y, fs=fs, nperseg=nperseg)
    H = Puy / Puu
    return f, np.abs(H), np.degrees(np.angle(H))


def model_frf(f, wn, zeta, T):
    s = 1j * 2 * np.pi * f
    H = (wn * wn) / (s * s + 2 * zeta * wn * s + wn * wn) * np.exp(-s * T)
    return np.abs(H), np.degrees(np.angle(H))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", nargs="?", default="logs/sysid/arm_sysid_amp_0.35_dur_60")
    ap.add_argument("--out", default="logs/sysid/fit")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.data_dir, "L_arm_j*.npz")))
    if not paths:
        raise SystemExit(f"no joint files in {args.data_dir}")

    results = {}
    n = len(paths)
    fig_t, ax_t = plt.subplots(n, 1, figsize=(11, 2.1 * n), sharex=True, squeeze=False)
    fig_b, ax_b = plt.subplots(n, 2, figsize=(12, 2.1 * n), squeeze=False)

    print(f"{'joint':10s} {'wn[Hz]':>8s} {'zeta':>6s} {'delay[ms]':>10s} "
          f"{'-3dB[Hz]':>9s} {'VAF%':>7s} {'RMS[rad]':>9s}")
    for i, p in enumerate(paths):
        d = load_joint(p)
        dt = 1.0 / float(d["rate"])
        base = float(d["probe_center"])
        u = np.asarray(d["q_cmd"], float) - base
        y_raw = np.asarray(d["q_state"], float) - base
        y, valid = clean(y_raw)

        # skip joints with no real response (e.g. j7 reads flat 0)
        out_amp = np.nanstd(y[valid])
        cmd_amp = np.std(u)
        if out_amp < 0.05 * cmd_amp:
            print(f"{d['name']:10s}  --- no joint response (out std {out_amp:.4f}) -> SKIP")
            results[d["name"]] = {"skipped": True, "reason": "no response"}
            ax_t[i, 0].plot(u, color="tab:blue", lw=0.7, label="q_cmd")
            ax_t[i, 0].plot(y, color="tab:orange", lw=0.7, label="q_state(flat)")
            ax_t[i, 0].set_title(f"{d['name']} -- NO RESPONSE"); ax_t[i, 0].legend(fontsize=7)
            continue

        fit = fit_joint(u, y, valid, dt)
        wn_hz = fit["wn"] / (2 * np.pi)
        # -3dB bandwidth of the 2nd-order low-pass
        zeta = fit["zeta"]
        f3 = wn_hz * np.sqrt(1 - 2 * zeta ** 2 + np.sqrt(4 * zeta ** 4 - 4 * zeta ** 2 + 2))
        results[d["name"]] = dict(wn_rad=fit["wn"], wn_hz=wn_hz, zeta=zeta,
                                  delay_s=fit["T"], f3db_hz=float(f3),
                                  vaf=fit["vaf"], rms=fit["rms"])
        print(f"{d['name']:10s} {wn_hz:8.3f} {zeta:6.3f} {1e3*fit['T']:10.2f} "
              f"{f3:9.3f} {fit['vaf']:7.1f} {fit['rms']:9.4f}")

        # time overlay
        ax_t[i, 0].plot(u + base, color="tab:blue", lw=0.7, label="q_cmd")
        ax_t[i, 0].plot(y + base, color="tab:orange", lw=0.9, label="q_state (real)")
        ax_t[i, 0].plot(fit["y_model"] + base, color="tab:green", lw=0.9, ls="--", label="fit model")
        ax_t[i, 0].set_title(f"{d['name']}  wn={wn_hz:.2f}Hz zeta={zeta:.2f} "
                             f"delay={1e3*fit['T']:.0f}ms VAF={fit['vaf']:.0f}%", fontsize=9)
        ax_t[i, 0].set_ylabel("rad"); ax_t[i, 0].grid(alpha=0.3)
        if i == 0:
            ax_t[i, 0].legend(fontsize=7, loc="upper right", ncol=3)

        # Bode overlay (empirical vs fitted), only over the excited band 0.1-2 Hz
        f, g_e, ph_e = empirical_frf(u, y, dt)
        g_m, ph_m = model_frf(f, fit["wn"], fit["zeta"], fit["T"])
        band = (f >= 0.08) & (f <= 3.0)
        ax_b[i, 0].semilogx(f[band], 20 * np.log10(g_e[band]), color="tab:orange", label="real")
        ax_b[i, 0].semilogx(f[band], 20 * np.log10(g_m[band]), color="tab:green", ls="--", label="fit")
        ax_b[i, 0].set_ylabel(f"{d['name']}\n|H| dB"); ax_b[i, 0].grid(alpha=0.3, which="both")
        ax_b[i, 1].semilogx(f[band], ph_e[band], color="tab:orange")
        ax_b[i, 1].semilogx(f[band], ph_m[band], color="tab:green", ls="--")
        ax_b[i, 1].set_ylabel("phase deg"); ax_b[i, 1].grid(alpha=0.3, which="both")
        if i == 0:
            ax_b[i, 0].legend(fontsize=7); ax_b[i, 0].set_title("gain"); ax_b[i, 1].set_title("phase")

    ax_t[-1, 0].set_xlabel("time step (100 Hz)")
    ax_b[-1, 0].set_xlabel("Hz"); ax_b[-1, 1].set_xlabel("Hz")
    fig_t.suptitle("real q_state vs fitted implicit-PD model"); fig_t.tight_layout()
    fig_b.suptitle("frequency response: real vs fitted model"); fig_b.tight_layout()
    fig_t.savefig(os.path.join(args.out, "fit_time.png"), dpi=130)
    fig_b.savefig(os.path.join(args.out, "fit_bode.png"), dpi=130)
    with open(os.path.join(args.out, "fit_params.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved -> {args.out}/fit_time.png, fit_bode.png, fit_params.json")


if __name__ == "__main__":
    main()
