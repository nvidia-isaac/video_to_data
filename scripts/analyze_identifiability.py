#!/usr/bin/env python3
"""Identifiability analysis of the arm-sysid fit: is (wn, zeta, delay) uniquely determined?

For each joint we probe four things:
  A. Jacobian conditioning at the optimum (scaled SVD) -> "sloppy" (near-flat) directions + the
     parameter combination that is under-constrained.
  B. Pole locations (slow/fast) vs the 2 Hz top of the chirp -> is the fast pole even excited?
  C. Reduced model: 1st-order + delay (one pole) VAF vs the 2nd-order + delay VAF -> is the 2nd
     pole necessary, or is the extra DOF redundant (=> wn,zeta not separately identifiable)?
  D. zeta-profile: FIX zeta at several values, re-fit (wn, T), report VAF -> an explicit family of
     'other solutions' that fit ~equally well.

Pure numpy/scipy (isaaclab venv).
"""

import glob
import os

import numpy as np
from scipy import signal
from scipy.optimize import least_squares

DATA = "/home/cning/simtoolreal_isaaclab/logs/sysid/arm_sysid_amp_0.35_dur_60"


def clean(y):
    y = np.asarray(y, float).copy()
    nan = np.isnan(y)
    if nan.any():
        idx = np.arange(len(y)); y[nan] = np.interp(idx[nan], idx[~nan], y[~nan])
    return y


def _delayed(u, dt, T):
    n = len(u); t = np.arange(n) * dt
    return np.interp(t - T, t, u, left=u[0]), t


def sim2(u, dt, wn, zeta, T):
    ud, t = _delayed(u, dt, T)
    _, y, _ = signal.lsim(([wn * wn], [1.0, 2 * zeta * wn, wn * wn]), U=ud, T=t)
    return y


def sim1(u, dt, tau, T):
    ud, t = _delayed(u, dt, T)
    _, y, _ = signal.lsim(([1.0], [tau, 1.0]), U=ud, T=t)
    return y


def vaf(y, ym):
    return 100.0 * (1.0 - np.var(y - ym) / np.var(y))


def fit2(u, y, dt, p0=(2 * np.pi, 0.7, 0.01)):
    r = least_squares(lambda p: sim2(u, dt, *p) - y, p0,
                      bounds=([2 * np.pi * .05, .05, 0.], [2 * np.pi * 20, 8., .1]), max_nfev=3000)
    return r


def fit1(u, y, dt):
    r = least_squares(lambda p: sim1(u, dt, *p) - y, (0.8, 0.01),
                      bounds=([1e-3, 0.], [20., .1]), max_nfev=3000)
    return r


def fit2_fixed_zeta(u, y, dt, zeta):
    r = least_squares(lambda p: sim2(u, dt, p[0], zeta, p[1]) - y, (2 * np.pi, 0.01),
                      bounds=([2 * np.pi * .05, 0.], [2 * np.pi * 20, .1]), max_nfev=3000)
    wn, T = r.x
    return wn, T, vaf(y, sim2(u, dt, wn, zeta, T))


def main():
    for p in sorted(glob.glob(os.path.join(DATA, "L_arm_j*.npz"))):
        d = np.load(p, allow_pickle=True)
        name = str(d["joint_name"]); dt = 1.0 / float(d["rate"]); base = float(d["probe_center"])
        u = np.asarray(d["q_cmd"], float) - base
        y = clean(np.asarray(d["q_state"], float)) - base
        if np.std(y) < 0.05 * np.std(u):
            print(f"\n=== {name}: no response, skip ==="); continue

        r = fit2(u, y, dt)
        wn, zeta, T = r.x
        v2 = vaf(y, sim2(u, dt, wn, zeta, T))

        # A. scaled Jacobian conditioning
        J = r.jac                                  # (N,3) d resid / d param
        scale = np.array([wn, zeta, max(T, 1e-3)])  # column-scale by param magnitude
        Js = J * scale
        sv = np.linalg.svd(Js, compute_uv=False)
        cond = sv[0] / sv[-1]
        _, _, VT = np.linalg.svd(Js, full_matrices=False)
        sloppy = VT[-1]                            # under-constrained direction (scaled params)

        # parameter correlations (caveat: residuals autocorrelated -> CIs optimistic)
        JtJ = J.T @ J
        try:
            C = np.linalg.inv(JtJ) * (2 * r.cost / (len(y) - 3))
            sd = np.sqrt(np.diag(C)); corr = C / np.outer(sd, sd)
        except np.linalg.LinAlgError:
            sd = np.full(3, np.nan); corr = np.full((3, 3), np.nan)

        # B. poles
        wn_hz = wn / (2 * np.pi)
        disc = np.sqrt(max(zeta * zeta - 1, 0))
        p_slow = wn * (zeta - disc); p_fast = wn * (zeta + disc)   # rad/s (magnitudes)

        # C. 1st-order + delay
        r1 = fit1(u, y, dt); tau, T1 = r1.x
        v1 = vaf(y, sim1(u, dt, tau, T1))

        print(f"\n=== {name} ===")
        print(f"  2nd+delay : wn={wn_hz:.3f} Hz  zeta={zeta:.3f}  delay={1e3*T:.1f} ms   VAF={v2:.3f}%")
        print(f"  poles     : slow={p_slow/(2*np.pi):.3f} Hz   fast={p_fast/(2*np.pi):.3f} Hz   "
              f"(chirp tops out at 2 Hz -> fast pole {'UNEXCITED' if p_fast/(2*np.pi)>2 else 'in band'})")
        print(f"  Jac cond  : {cond:.1f}  (singular values {sv[0]:.2e}/{sv[1]:.2e}/{sv[2]:.2e})")
        print(f"  sloppy dir (d wn, d zeta, d T)_scaled = ({sloppy[0]:+.2f},{sloppy[1]:+.2f},{sloppy[2]:+.2f})")
        print(f"  std err   : wn±{sd[0]/(2*np.pi):.3f}Hz zeta±{sd[1]:.3f} T±{1e3*sd[2]:.1f}ms  "
              f"corr(wn,zeta)={corr[0,1]:+.3f} corr(zeta,T)={corr[1,2]:+.3f}")
        print(f"  1st+delay : tau={tau:.3f}s (pole {1/(2*np.pi*tau):.3f} Hz) delay={1e3*T1:.1f}ms  "
              f"VAF={v1:.3f}%   (Δ vs 2nd-order = {v2-v1:+.3f} pts)")
        # D. zeta-profile: other solutions
        zs = [1.0, 1.5, 2.0, zeta, 4.0, 6.0]
        prof = []
        for zz in sorted(set(round(z, 3) for z in zs)):
            wnz, Tz, vz = fit2_fixed_zeta(u, y, dt, zz)
            prof.append(f"zeta={zz:.2f}->wn={wnz/(2*np.pi):.3f}Hz,delay={1e3*Tz:.0f}ms,VAF={vz:.2f}%")
        print("  zeta-profile (fix zeta, refit wn&delay): " + " | ".join(prof))


if __name__ == "__main__":
    main()
