"""Flow-matching trainer for the STATE-BASED keypoint policy (Gr00tN1d7KeypointPolicy).

Reads a collect_bc_data.py HDF5 and trains the keypoint policy directly on the object-centric
keypoints (tool + dynamic screw) + proprio state -> 40-step delta-joint action chunk. No images,
no video/memmap (keypoints are tiny -> everything fits in RAM):

  keypoints -> normalize (shared per-coord mean/std, geometry-preserving)        -> [B, 8, 3]
  state     -> z-score (stats), pad 29->max_state_dim, history=1                 -> [B, 1, 132]
  action    -> chunk actions[t:t+horizon] (no cross-episode), z-score, pad+mask  -> [B, 40, 132]
  embodiment_id = 0

Run (GR00T venv):
  cd Isaac-GR00T && source .venv/bin/activate
  python /home/cning/simtoolreal_isaaclab/scripts/train_keypoint_policy.py \
      --hdf5 /home/cning/simtoolreal_isaaclab/datasets/hammer_kp_1000.hdf5 --name hammer_kp --steps 20000
"""

import argparse
import os
import time

import h5py
import numpy as np
import torch

from gr00t.model.gr00t_n1d7.gr00t_n1d7_keypoint import Gr00tN1d7KeypointPolicy, make_keypoint_config

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True, help="collect_bc_data.py HDF5 with obs/keypoints")
    p.add_argument("--no_teleport_mask", action="store_true", help="disable masking the action-chunk loss past a teleport (default: mask, when the dataset has the per-step 'teleport' flag). Ablation only")
    p.add_argument("--name", default="hammer_kp")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--out", default="/home/cning/simtoolreal_isaaclab/logs/gr00t_specialist")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def lr_at(step, total, warmup):
    if step < warmup:
        return step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)

    # ---- load all keypoints / state / actions into RAM ----
    f = h5py.File(args.hdf5, "r")
    data = f["data"]
    demos = sorted([k for k in data if k.startswith("demo_")], key=lambda s: int(s.split("_")[1]))
    kps, states, actions, tps, bounds = [], [], [], [], []
    has_tp = "teleport" in data[demos[0]]   # per-step teleport flag present? (tool_displacement dataset)
    off = 0
    for dk in demos:
        g = data[dk]
        kp = g["obs/keypoints"][:].astype(np.float32)      # (T,8,3)
        st = g["obs/joint_pos"][:].astype(np.float32)      # (T,29)
        ac = g["actions"][:].astype(np.float32)            # (T,29)
        tp = g["teleport"][:].astype(np.float32) if "teleport" in g else np.zeros(len(ac), np.float32)  # (T,)
        T = min(len(kp), len(st), len(ac))
        kps.append(kp[:T]); states.append(st[:T]); actions.append(ac[:T]); tps.append(tp[:T])
        bounds.append((off, off + T)); off += T
    f.close()
    kp = np.concatenate(kps); st = np.concatenate(states); ac = np.concatenate(actions); tp_all = np.concatenate(tps)
    N, n_kp = kp.shape[0], kp.shape[1]
    data_dim = st.shape[1]
    print(f"[train] {len(demos)} demos | N={N} frames | {n_kp} keypoints | state/action dim={data_dim}", flush=True)

    # ---- train/val split by EPISODE FIRST (fit normalization stats on TRAIN frames only) ----
    frame_end = torch.zeros(N, dtype=torch.long, device=dev)
    n_ep = len(bounds)
    rng = np.random.default_rng(args.seed)
    n_val = min(n_ep - 1, max(1, int(round(args.val_frac * n_ep)))) if args.val_frac > 0 and n_ep >= 5 else 0
    val_eps = set(rng.choice(n_ep, n_val, replace=False).tolist()) if n_val else set()
    tr_list, va_list = [], []
    for e, (lo, hi) in enumerate(bounds):
        frame_end[lo:hi] = hi
        (va_list if e in val_eps else tr_list).extend(range(lo, hi))
    tr_np = np.array(tr_list, dtype=np.int64)
    print(f"[train] {n_ep} eps ({n_ep - len(val_eps)} train / {len(val_eps)} val)", flush=True)

    # ---- normalize: fit stats on TRAIN frames ONLY (no val leakage), apply to all ----
    kp_mean = kp[tr_np].reshape(-1, 3).mean(0)                  # shared per-coord (geometry-preserving)
    kp_std = kp[tr_np].reshape(-1, 3).std(0) + 1e-6
    s_mean, s_std = st[tr_np].mean(0), st[tr_np].std(0) + 1e-6
    a_mean, a_std = ac[tr_np].mean(0), ac[tr_np].std(0) + 1e-6
    KP = torch.tensor((kp - kp_mean) / kp_std, device=dev)         # [N,8,3]
    ST = torch.tensor((st - s_mean) / s_std, device=dev)          # [N,29]
    AC = torch.tensor((ac - a_mean) / a_std, device=dev)          # [N,29]
    TP = torch.tensor(tp_all, device=dev)                          # [N] per-step teleport flag (0/1)
    mask_teleport = has_tp and (not args.no_teleport_mask)         # mask chunk-loss past a teleport?
    if has_tp:
        print(f"[train] teleport flag present | chunk-loss masking: {'ON' if mask_teleport else 'OFF'} "
              f"({float(tp_all.mean())*100:.2f}% of frames are teleports)", flush=True)
    tr_idx = torch.tensor(tr_list); va_idx = torch.tensor(va_list)

    # ---- model ----
    cfg = make_keypoint_config(n_tool_keypoints=4, n_screw_keypoints=n_kp - 4)
    H, Smax, Amax = cfg.action_horizon, cfg.max_state_dim, cfg.max_action_dim
    model = Gr00tN1d7KeypointPolicy(cfg).to(dev).train()
    print(f"[train] keypoint policy: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params "
          f"({sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable)", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    def make_batch(pool):
        sel = pool[torch.randint(0, len(pool), (args.batch_size,))]
        idx = sel.tolist()
        kpb = KP[sel]                                            # [B,8,3]
        state = torch.zeros(args.batch_size, 1, Smax, device=dev)
        action = torch.zeros(args.batch_size, H, Amax, device=dev)
        mask = torch.zeros(args.batch_size, H, Amax, device=dev)
        for b, t in enumerate(idx):
            ep_end = int(frame_end[t]); valid = min(H, ep_end - t)
            state[b, 0, :data_dim] = ST[t]
            action[b, :valid, :data_dim] = AC[t:t + valid]
            mask[b, :valid, :data_dim] = 1.0
            if mask_teleport and valid > 0:   # zero chunk-loss past the first teleport (unpredictable recovery)
                tpc = TP[t:t + valid]
                prior = torch.cat([tpc.new_zeros(1), torch.cumsum(tpc, 0)[:-1]])
                mask[b, :valid, :data_dim] *= (prior == 0).float()[:, None]
        return {"keypoints": kpb, "state": state, "action": action, "action_mask": mask,
                "embodiment_id": torch.zeros(args.batch_size, dtype=torch.long, device=dev)}

    @torch.no_grad()
    def val_loss(nb=16):
        model.eval(); ls = [model(make_batch(va_idx))["loss"].item() for _ in range(nb)]; model.train()
        return float(np.mean(ls))

    out_dir = f"{args.out}/{args.name}"; os.makedirs(out_dir, exist_ok=True)
    csv = open(f"{out_dir}/loss_curve.csv", "w"); csv.write("step,loss,loss_ema,val_loss,lr\n")
    ema, hist, t0 = None, [], time.time()
    stats = {"kp_mean": kp_mean.tolist(), "kp_std": kp_std.tolist(),
             "state_mean": s_mean.tolist(), "state_std": s_std.tolist(),
             "action_mean": a_mean.tolist(), "action_std": a_std.tolist()}
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps, args.warmup) * args.lr
        for pg in opt.param_groups:
            pg["lr"] = lr
        loss = model(make_batch(tr_idx))["loss"]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        lv = loss.item(); ema = lv if ema is None else 0.98 * ema + 0.02 * lv; hist.append((step, lv, ema))
        vl = ""
        if len(va_idx) and (step % args.eval_every == 0 or step == 1):
            vl = val_loss()
            print(f"[train] step {step:5d}/{args.steps} loss={lv:.4f} ema={ema:.4f} val={vl:.4f} "
                  f"lr={lr:.2e} ({step/(time.time()-t0):.1f} it/s)", flush=True)
        csv.write(f"{step},{lv:.5f},{ema:.5f},{vl},{lr:.6e}\n"); csv.flush()
        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save({"model_state_dict": model.state_dict(), "config": cfg.to_dict(),
                        "stats": stats, "data_dim": data_dim, "n_keypoints": n_kp, "step": step},
                       f"{out_dir}/{args.name}.pt")
    csv.close()
    _plot(f"{out_dir}/loss_curve.csv", f"{out_dir}/loss_curve.png", args.name)
    print(f"\n[train] DONE: {args.steps} steps | ema {hist[0][2]:.4f} -> {hist[-1][2]:.4f} "
          f"(min raw {min(h[1] for h in hist):.4f}) -> {out_dir}/{args.name}.pt", flush=True)


def _plot(csv_path, png, name):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["step"], df["loss"], color="tab:blue", alpha=0.25, lw=0.6, label="train (raw)")
    ax.plot(df["step"], df["loss_ema"], color="tab:blue", lw=1.8, label="train (EMA)")
    v = df[df["val_loss"].astype(str).str.len() > 0].copy()
    if len(v):
        v["val_loss"] = pd.to_numeric(v["val_loss"], errors="coerce")
        ax.plot(v["step"], v["val_loss"], "o-", color="tab:red", ms=3, lw=1.2, label="val")
    ax.set_xlabel("step"); ax.set_ylabel("flow-matching loss"); ax.set_title(f"keypoint policy '{name}'")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
