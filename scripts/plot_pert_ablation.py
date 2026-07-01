"""Vertical bar chart of the perturbation ablation (no-jv state specialist @ table_dist 0.15).
Pass --data as JSON list of [label, value_or_null, color_key]; null value -> shown as 'pending' (hatched)."""
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help='JSON list: [["label", value|null, "group"], ...]')
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="Perturbation ablation: no-jv state specialist @ table_dist 0.15 (400 ep)")
args = ap.parse_args()

rows = json.loads(args.data)
labels = [r[0] for r in rows]
vals = [(r[1] if r[1] is not None else 0.0) for r in rows]
pending = [r[1] is None for r in rows]
groups = [r[2] for r in rows]
cmap = {"baseline": "0.6", "single": "tab:blue", "combo": "tab:orange", "all": "tab:green"}
colors = [cmap.get(g, "tab:blue") for g in groups]

x = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(10, 5.8))
bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
for i, b in enumerate(bars):
    if pending[i]:
        b.set_hatch("//"); b.set_alpha(0.35)
        ax.text(b.get_x()+b.get_width()/2, 2, "pending", ha="center", va="bottom", fontsize=10, rotation=90)
    else:
        ax.text(b.get_x()+b.get_width()/2, vals[i]+0.6, f"{vals[i]:.1f}%", ha="center", va="bottom", fontsize=12)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, rotation=18, ha="right")
ax.set_ylabel("success rate (%)", fontsize=12)
ax.set_title(args.title, fontsize=12)
ax.set_ylim(0, max([v for v in vals if v] + [50]) * 1.25)
ax.grid(axis="y", alpha=0.3)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=cmap["baseline"], label="no perturbation"),
                   Patch(color=cmap["single"], label="single perturbation"),
                   Patch(color=cmap["combo"], label="combination"),
                   Patch(color=cmap["all"], label="all perturbations")], fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig(args.out, dpi=120)
print(f"[plot] wrote {args.out}")
