"""Bar chart of model success rates (current metric). Reads logs/all_eval_results.json
({label: {"success": float_pct, "group": str, "detail": str}}) and writes a PNG.

  python scripts/plot_success_bar.py --json logs/all_eval_results.json --out logs/model_success_bar.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="/home/cning/simtoolreal_isaaclab/logs/all_eval_results.json")
    ap.add_argument("--out", default="/home/cning/simtoolreal_isaaclab/logs/model_success_bar.png")
    ap.add_argument("--title", default="Hammer success rate by model")
    ap.add_argument("--fs", type=float, default=27.0, help="base (large) font size ~3x default")
    args = ap.parse_args()
    fs = args.fs
    data = json.load(open(args.json))
    items = sorted(data.items(), key=lambda kv: kv[1]["success"])  # ascending ramp
    labels = [k for k, _ in items]
    vals = [v["success"] for _, v in items]
    groups = [v.get("group", "bc") for _, v in items]
    cmap = {"expert": "#2ca02c", "image": "#1f77b4", "keypoint": "#ff7f0e",
            "simtoolreal": "#9467bd", "teleport": "#d62728",
            "diverse": "#17becf", "diverse_tele": "#1f77b4"}
    colors = [cmap.get(g, "#7f7f7f") for g in groups]
    fig, ax = plt.subplots(figsize=(18, 11))
    bars = ax.bar(range(len(labels)), vals, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=fs)
    ax.set_ylabel("nail-driven success rate (%)", fontsize=fs)
    ax.set_title(args.title, fontsize=fs * 1.05)
    ax.set_ylim(0, max(100, max(vals) * 1.15))
    ax.tick_params(axis="y", labelsize=fs * 0.85)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=fs)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"wrote {args.out} with {len(labels)} bars (fontsize {fs:.0f})")


if __name__ == "__main__":
    main()
