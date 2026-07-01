"""Flow-matching trainer for the SimToolReal SPECIALIST (state-based, expert-faithful obs).

Same keypoint-token backbone + ~50M flow head as the keypoint policy (train_keypoint_policy.py),
but fed the richer observation that mirrors the SimToolReal RL expert's actor input (minus the goal):

  keypoints_rel_palm -> normalize (shared per-coord, geometry-preserving)        -> [B, 8, 3]
  proprio (109)      -> z-score per-dim, pad 109->max_state_dim, history=1        -> [B, 1, 132]
                        proprio = joint_pos29 + joint_vel29 + prev_targets29 + palm_pos3
                                  + palm_rot4 + fingertip_pos_rel_palm15
  action             -> chunk actions[t:t+horizon] (no cross-episode), z-score, pad+mask -> [B, 40, 132]
  embodiment_id = 0

Needs an HDF5 collected with `collect_bc_data.py --simtoolreal` (has obs/keypoints_rel_palm + obs/proprio).

Run (GR00T venv):
  cd Isaac-GR00T && source .venv/bin/activate
  python /home/cning/simtoolreal_isaaclab/scripts/train_simtoolreal_specialist.py \
      --hdf5 /home/cning/simtoolreal_isaaclab/datasets/hammer_str_1000.hdf5 --name hammer_str --steps 20000
"""

import argparse
import os
import time

import h5py
import numpy as np
import torch

from gr00t.model.gr00t_n1d7.gr00t_n1d7_keypoint import Gr00tN1d7KeypointPolicy, make_simtoolreal_config

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True, help="collect_bc_data.py --simtoolreal HDF5 (obs/keypoints_rel_palm + obs/proprio)")
    p.add_argument("--with_goal", action="store_true", help="append obs/keypoints_rel_goal(12) to proprio (109->121): the goal signal the SAPG teacher actor has (needs a dataset recorded with the updated collector)")
    p.add_argument("--no_joint_vel", action="store_true", help="DROP joint_vel from proprio (dims [29:58]) -> 109->80. The 'state-based' variant without velocity (sim-to-real: joint_vel is often noisy/unavailable on real hardware). Eval with the matching eval_simtoolreal_client.py --no_joint_vel")
    p.add_argument("--no_teleport_mask", action="store_true", help="disable masking the action-chunk loss past a teleport (default: mask, when the dataset has the per-step 'teleport' flag). Ablation only")
    p.add_argument("--name", default="hammer_str")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--dit_hidden", type=int, default=512, help="DiT hidden/inner dim (512=~50M, 768=~150M)")
    p.add_argument("--dit_layers", type=int, default=12, help="DiT layers (12=~50M, 18=~150M)")
    p.add_argument("--dit_heads", type=int, default=8, help="DiT attention heads (head_dim=dit_hidden/dit_heads)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--val_batches", type=int, default=64, help="minibatches averaged per val_loss eval (higher = cleaner convergence signal)")
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--out", default="/home/cning/simtoolreal_isaaclab/logs/gr00t_specialist")
    p.add_argument("--init_from", default="", help="warm-start the model weights from this .pt (resume/continue training; optimizer restarts fresh)")
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

    # ---- load all keypoints_rel_palm / proprio / actions into RAM ----
    f = h5py.File(args.hdf5, "r")
    data = f["data"]
    demos = sorted([k for k in data if k.startswith("demo_")], key=lambda s: int(s.split("_")[1]))
    if "obs/proprio" not in data[demos[0]]:
        raise SystemExit("[train] HDF5 has no obs/proprio -- collect with `collect_bc_data.py --simtoolreal`")
    if args.with_goal and "obs/keypoints_rel_goal" not in data[demos[0]]:
        raise SystemExit("[train] --with_goal needs obs/keypoints_rel_goal -- recollect with the updated collector")
    kps, props, actions, tps, bounds = [], [], [], [], []
    has_tp = "teleport" in data[demos[0]]   # per-step teleport flag present? (collected with tool_displacement)
    off = 0
    for dk in demos:
        g = data[dk]
        kp = g["obs/keypoints_rel_palm"][:].astype(np.float32)   # (T,8,3)
        pr = g["obs/proprio"][:].astype(np.float32)              # (T,109)
        if args.no_joint_vel:  # drop joint_vel [29:58] -> 80 = joint_pos(29)+prev_targets(29)+palm(3)+palm_quat(4)+ft_rel(15)
            pr = np.concatenate([pr[:, :29], pr[:, 58:]], axis=1)
        if args.with_goal:   # append keypoints_rel_goal(12) -> proprio 109->121 (the teacher's goal signal)
            pr = np.concatenate([pr, g["obs/keypoints_rel_goal"][:].astype(np.float32)], axis=1)
        ac = g["actions"][:].astype(np.float32)                  # (T,29)
        tp = g["teleport"][:].astype(np.float32) if "teleport" in g else np.zeros(len(ac), np.float32)  # (T,) 1=teleport this step
        T = min(len(kp), len(pr), len(ac))
        kps.append(kp[:T]); props.append(pr[:T]); actions.append(ac[:T]); tps.append(tp[:T])
        bounds.append((off, off + T)); off += T
    f.close()
    kp = np.concatenate(kps); pr = np.concatenate(props); ac = np.concatenate(actions); tp_all = np.concatenate(tps)
    N, n_kp = kp.shape[0], kp.shape[1]
    data_dim = pr.shape[1]        # 109 proprio
    act_dim = ac.shape[1]         # 29
    print(f"[train] {len(demos)} demos | N={N} frames | {n_kp} keypoints | proprio dim={data_dim} action dim={act_dim}", flush=True)

    # ---- train/val split by EPISODE FIRST (so normalization stats are fit on TRAIN frames only,
    #      no validation leakage into the normalizer) ----
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

    # ---- normalize: fit stats on TRAIN frames ONLY, then apply to all (train + val + deploy) ----
    kp_mean = kp[tr_np].reshape(-1, 3).mean(0)                   # shared per-coord (geometry-preserving)
    kp_std = kp[tr_np].reshape(-1, 3).std(0) + 1e-6
    p_mean, p_std = pr[tr_np].mean(0), pr[tr_np].std(0) + 1e-6   # proprio z-score per-dim
    a_mean, a_std = ac[tr_np].mean(0), ac[tr_np].std(0) + 1e-6
    KP = torch.tensor((kp - kp_mean) / kp_std, device=dev)         # [N,8,3]
    PR = torch.tensor((pr - p_mean) / p_std, device=dev)          # [N,109]
    AC = torch.tensor((ac - a_mean) / a_std, device=dev)          # [N,29]
    TP = torch.tensor(tp_all, device=dev)                          # [N] per-step teleport flag (0/1)
    mask_teleport = has_tp and (not args.no_teleport_mask)         # mask chunk-loss past a teleport?
    if has_tp:
        print(f"[train] teleport flag present | chunk-loss masking past teleports: {'ON' if mask_teleport else 'OFF'} "
              f"({float(tp_all.mean())*100:.2f}% of frames are teleports)", flush=True)
    tr_idx = torch.tensor(tr_list); va_idx = torch.tensor(va_list)

    # ---- model ----
    cfg = make_simtoolreal_config(n_tool_keypoints=4, n_screw_keypoints=n_kp - 4,
                                  hidden_size=args.dit_hidden, input_embedding_dim=args.dit_hidden,
                                  dit_num_layers=args.dit_layers, dit_num_heads=args.dit_heads,
                                  dit_head_dim=args.dit_hidden // args.dit_heads)
    H, Smax, Amax = cfg.action_horizon, cfg.max_state_dim, cfg.max_action_dim
    if data_dim > Smax:
        raise SystemExit(f"[train] proprio dim {data_dim} > max_state_dim {Smax}")
    model = Gr00tN1d7KeypointPolicy(cfg).to(dev).train()
    if args.init_from:
        ck0 = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ck0["model_state_dict"])
        print(f"[train] warm-started weights from {args.init_from} (was step {ck0.get('step')})", flush=True)
    print(f"[train] simtoolreal specialist: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params "
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
            state[b, 0, :data_dim] = PR[t]
            action[b, :valid, :act_dim] = AC[t:t + valid]
            mask[b, :valid, :act_dim] = 1.0
            if mask_teleport and valid > 0:
                # zero the loss for chunk steps AT/AFTER the first teleport in the window: those actions
                # are recovery the chunk's input obs couldn't predict (teleport is an ad-hoc state jump).
                tpc = TP[t:t + valid]                                    # teleport flag per chunk frame
                prior = torch.cat([tpc.new_zeros(1), torch.cumsum(tpc, 0)[:-1]])  # teleports BEFORE this step
                keep = (prior == 0).float()                             # 1 until (and incl.) the teleport step, 0 after
                mask[b, :valid, :act_dim] *= keep[:, None]
        return {"keypoints": kpb, "state": state, "action": action, "action_mask": mask,
                "embodiment_id": torch.zeros(args.batch_size, dtype=torch.long, device=dev)}

    @torch.no_grad()
    def val_loss(nb=None):
        nb = nb or args.val_batches
        model.eval(); ls = [model(make_batch(va_idx))["loss"].item() for _ in range(nb)]; model.train()
        return float(np.mean(ls))

    out_dir = f"{args.out}/{args.name}"; os.makedirs(out_dir, exist_ok=True)
    csv = open(f"{out_dir}/loss_curve.csv", "w"); csv.write("step,loss,loss_ema,val_loss,lr\n")
    ema, hist, val_hist, t0 = None, [], [], time.time()
    stats = {"kp_mean": kp_mean.tolist(), "kp_std": kp_std.tolist(),
             "proprio_mean": p_mean.tolist(), "proprio_std": p_std.tolist(),
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
            val_hist.append((step, vl))
            print(f"[train] step {step:5d}/{args.steps} loss={lv:.4f} ema={ema:.4f} val={vl:.4f} "
                  f"lr={lr:.2e} ({step/(time.time()-t0):.1f} it/s)", flush=True)
        csv.write(f"{step},{lv:.5f},{ema:.5f},{vl},{lr:.6e}\n"); csv.flush()
        if step % args.ckpt_every == 0 or step == args.steps:
            payload = {"model_state_dict": model.state_dict(), "config": cfg.to_dict(),
                       "stats": stats, "data_dim": data_dim, "action_dim": act_dim,
                       "n_keypoints": n_kp, "step": step,
                       "no_joint_vel": args.no_joint_vel, "with_goal": args.with_goal}
            torch.save(payload, f"{out_dir}/{args.name}.pt")              # latest (for eval/resume)
            torch.save(payload, f"{out_dir}/{args.name}_step{step}.pt")   # step-named (eval picks best by SUCCESS)
    csv.close()
    _plot(f"{out_dir}/loss_curve.csv", f"{out_dir}/loss_curve.png", args.name)
    # ---- convergence verdict: compare mean val over the last window vs the prior window ----
    verdict = "n/a (no val set)"
    if len(val_hist) >= 6:
        k = max(1, len(val_hist) // 5)
        late = float(np.mean([v for _, v in val_hist[-k:]]))
        prev = float(np.mean([v for _, v in val_hist[-2 * k:-k]]))
        rel = (prev - late) / (abs(prev) + 1e-9)
        verdict = (f"CONVERGED (val plateau {prev:.4f}->{late:.4f}, {rel:+.1%} over last window)"
                   if rel < 0.02 else
                   f"NOT CONVERGED (val still falling {prev:.4f}->{late:.4f}, {rel:+.1%}); train more steps")
    print(f"\n[train] DONE: {args.steps} steps | ema {hist[0][2]:.4f} -> {hist[-1][2]:.4f} "
          f"(min raw {min(h[1] for h in hist):.4f}) | CONVERGENCE: {verdict} -> {out_dir}/{args.name}.pt", flush=True)


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
    ax.set_xlabel("step"); ax.set_ylabel("flow-matching loss"); ax.set_title(f"simtoolreal specialist '{name}'")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
