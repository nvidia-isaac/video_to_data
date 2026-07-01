"""Grouped bar chart: no-goal SimToolReal specialist success @ table_dist 0.15,
joint_vel vs no-joint_vel, across datasets. Pass --data as JSON: {group_label: {"jv": pct, "nojv": pct}}."""
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help='JSON e.g. {"diverse_goal":{"jv":26,"nojv":38}}')
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="No-goal state specialist @ table_dist 0.15: joint_vel ablation")
args = ap.parse_args()

data = json.loads(args.data)
groups = list(data.keys())
x = np.arange(len(groups))
w = 0.36
jv = [data[g]["jv"] for g in groups]
nojv = [data[g]["nojv"] for g in groups]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
b1 = ax.bar(x - w/2, jv, w, label="with joint_vel (proprio 109)", color="tab:blue")
b2 = ax.bar(x + w/2, nojv, w, label="without joint_vel (proprio 80, 'state-based')", color="tab:green")
for b in (b1, b2):
    ax.bar_label(b, fmt="%.1f%%", fontsize=13, padding=2)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=13)
ax.set_ylabel("success rate (%)", fontsize=13)
ax.set_title(args.title, fontsize=13)
ax.set_ylim(0, max(jv + nojv) * 1.25 + 1)
ax.legend(fontsize=11, loc="upper left"); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(args.out, dpi=120)
print(f"[plot] wrote {args.out}")
