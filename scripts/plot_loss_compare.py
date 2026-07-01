"""Overlay the flow-matching loss curves of two state-specialist runs (e.g. +goal vs no-goal)."""
import argparse, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="label:csv_path pairs")
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="flow-matching loss: +goal vs no-goal")
args = ap.parse_args()

fig, ax = plt.subplots(figsize=(9, 5.5))
colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
for i, r in enumerate(args.runs):
    label, path = r.split(":", 1)
    df = pd.read_csv(path)
    c = colors[i % len(colors)]
    ax.plot(df["step"], df["loss_ema"], color=c, lw=2.0, label=f"{label} train EMA (final {df['loss_ema'].iloc[-1]:.3f})")
    v = df[df["val_loss"].astype(str).str.len() > 0].copy()
    if len(v):
        v["val_loss"] = pd.to_numeric(v["val_loss"], errors="coerce")
        v = v.dropna(subset=["val_loss"])
        ax.plot(v["step"], v["val_loss"], "o--", color=c, ms=4, lw=1.0, alpha=0.7,
                label=f"{label} val (final {v['val_loss'].iloc[-1]:.3f})")
ax.set_xlabel("step", fontsize=13); ax.set_ylabel("flow-matching loss", fontsize=13)
ax.set_title(args.title, fontsize=14); ax.legend(fontsize=10); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(args.out, dpi=120)
print(f"[plot] wrote {args.out}")
