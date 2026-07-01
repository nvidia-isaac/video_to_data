#!/usr/bin/env python3
"""Controlled test: is the (wn,zeta) ambiguity caused by the chirp only exciting <= 2 Hz?

Take a KNOWN ground-truth 2nd-order+delay (slow pole 0.2 Hz, fast pole ~6 Hz, like the real fit).
Drive it with (A) the real 0.1->2 Hz chirp and (B) a 0.1->10 Hz chirp (which DOES sweep through the
6 Hz fast pole), with and without realistic measurement noise. Fit each + run a zeta-profile.

If the 2 Hz case gives a FLAT zeta-valley (can't recover true zeta) but the 10 Hz case recovers the
true zeta with a clear minimum, the cause is the excitation band, not the method. The noisy 10 Hz
case shows the SNR caveat (the plant attenuates the fast pole, so output up there is tiny).
"""
import numpy as np
from scipy import signal
from scipy.optimize import least_squares

FS, DUR, AMP = 100.0, 60.0, 0.35
DT = 1.0 / FS
WN_T, ZETA_T, T_T = 2 * np.pi * 1.09, 2.85, 0.041     # ground truth (j1-like)
RNG = np.random.default_rng(0)


def chirp(f1):
    t = np.arange(int(DUR * FS)) * DT
    return t, AMP * signal.chirp(t, f0=0.1, f1=f1, t1=DUR, method="linear")


def sim2(u, wn, zeta, T):
    t = np.arange(len(u)) * DT
    ud = np.interp(t - T, t, u, left=u[0])
    _, y, _ = signal.lsim(([wn * wn], [1., 2 * zeta * wn, wn * wn]), U=ud, T=t)
    return y


def vaf(y, ym):
    return 100. * (1 - np.var(y - ym) / np.var(y))


def fit_free(u, y):
    r = least_squares(lambda p: sim2(u, *p) - y, (2 * np.pi, 0.7, 0.01),
                      bounds=([2 * np.pi * .05, .05, 0.], [2 * np.pi * 20, 8., .15]), max_nfev=4000)
    return r.x


def fit_fixed_zeta(u, y, zeta):
    r = least_squares(lambda p: sim2(u, p[0], zeta, p[1]) - y, (2 * np.pi, .01),
                      bounds=([2 * np.pi * .05, 0.], [2 * np.pi * 20, .15]), max_nfev=3000)
    wn, T = r.x
    return wn, T, vaf(y, sim2(u, wn, zeta, T))


fast_hz = WN_T * (ZETA_T + np.sqrt(ZETA_T**2 - 1)) / (2 * np.pi)
print(f"GROUND TRUTH: wn={WN_T/(2*np.pi):.3f} Hz  zeta={ZETA_T}  delay={1e3*T_T:.0f} ms  "
      f"(slow pole {WN_T*(ZETA_T-np.sqrt(ZETA_T**2-1))/(2*np.pi):.3f} Hz, fast pole {fast_hz:.2f} Hz)\n")

for f1 in (2.0, 10.0):
    t, u = chirp(f1)
    y0 = sim2(u, WN_T, ZETA_T, T_T)
    for noise in (0.0, 0.002):
        y = y0 + (RNG.normal(0, noise, len(y0)) if noise else 0.0)
        wn, zeta, T = fit_free(u, y)
        # zeta-profile
        zs = [1.5, 2.0, 2.85, 4.0, 6.0]
        prof = [(z, *fit_fixed_zeta(u, y, z)) for z in zs]
        vafs = np.array([p[3] for p in prof])
        spread = vafs.max() - vafs.min()
        # output amplitude near the fast pole (last 10% of the 10Hz sweep ~ 9-10 Hz)
        hf = np.std(y0[int(0.9 * len(y0)):])
        print(f"chirp 0.1->{f1:>4} Hz | noise={noise*1e3:.0f} mrad | hf-out {1e3*hf:5.1f} mrad")
        print(f"   free fit:  wn={wn/(2*np.pi):.3f} Hz  zeta={zeta:.3f}  delay={1e3*T:.0f} ms"
              f"   {'<-- recovers truth' if abs(zeta-ZETA_T)<0.4 else '<-- WRONG zeta (ambiguous)'}")
        print("   zeta-profile VAF: " + "  ".join(f"z={z}:{v:.3f}%" for z, _, _, v in prof)
              + f"   spread={spread:.3f} pts {'(FLAT->unidentifiable)' if spread<0.1 else '(peaked->identifiable)'}")
    print()
