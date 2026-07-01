"""DAgger success-rate vs iteration (best run: linear/slow beta, lr=1e-5, chunk relabel).
Clean per-iter eval @ no-jv / table_dist 0.15 / 1200-step / 400 ep. Pass evaluated points via --data JSON."""
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help='JSON list [[iter, beta, success_or_null], ...]')
ap.add_argument("--bc", type=float, default=60.8, help="BC warm-start baseline %")
ap.add_argument("--expert", type=float, default=86.0, help="SAPG expert reference %")
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="DAgger success vs iteration (best run: linear beta 1.0->0.1, lr=1e-5, chunk relabel)")
args = ap.parse_args()

rows = [r for r in json.loads(args.data) if r[2] is not None]
its = [r[0] for r in rows]; betas = [r[1] for r in rows]; succ = [r[2] for r in rows]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.axhline(args.expert, ls=":", color="0.4", lw=1.3, label=f"SAPG expert ({args.expert:.0f}%)")
ax.axhline(args.bc, ls="--", color="tab:green", lw=1.6, label=f"BC warm-start ({args.bc:.1f}%)")
ax.plot(its, succ, "o-", color="tab:blue", lw=2.0, ms=7, label="DAgger (best run)")
for it, b, s in zip(its, betas, succ):
    ax.annotate(f"{s:.0f}", (it, s), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
ax.set_xlabel("DAgger iteration  (β shown below)", fontsize=12); ax.set_ylabel("success rate (%)", fontsize=12)
ax.set_title(args.title, fontsize=11)
ax.set_xticks(its); ax.set_xticklabels([f"{it}\nβ{b:.1f}" for it, b in zip(its, betas)], fontsize=9)
ax.set_xlim(min(its) - 0.5, max(its) + 0.5); ax.set_ylim(0, max(args.expert + 6, max(succ) + 12))
ax.legend(fontsize=10, loc="lower left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(args.out, dpi=120)
print(f"[plot] wrote {args.out}")
