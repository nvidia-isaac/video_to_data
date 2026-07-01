"""Plot the strike-training data-collection success-rate curves (nail_driven over 100 episodes) vs
training step: one line WITHOUT perturbation, one WITH all perturbations. Re-run after each eval to
refresh the chart. Reads logs/strike_eval/strike_eval.csv, writes logs/strike_eval/strike_eval.png."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/cning/simtoolreal_isaaclab"
CSV = f"{REPO}/logs/strike_eval/strike_eval.csv"
OUT = f"{REPO}/logs/strike_eval/strike_eval.png"

rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["step"]), int(r["perturbation"]), float(r["rate"]), int(r["attempts"])))

def series(pert):
    pts = sorted((s, rate) for s, p, rate, _ in rows if p == pert)
    return [s for s, _ in pts], [100 * rate for _, rate in pts]

plt.figure(figsize=(9, 5.5))
for pert, color, lab in [(0, "#2a9d8f", "no perturbation"), (1, "#e76f51", "all perturbations")]:
    xs, ys = series(pert)
    if xs:
        plt.plot(xs, ys, "-o", color=color, label=lab)
        for x, y in zip(xs, ys):
            plt.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=7, color=color)
plt.xlabel("training epoch (strike fine-tune, warm-started from v4)")
plt.ylabel("nail-driven success rate over 100 eps (%)")
plt.title("Vega-right strike policy — data-collection success rate vs training")
plt.ylim(-2, 100); plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(OUT, dpi=110)
print(f"WROTE {OUT} ({len(rows)} rows)")
