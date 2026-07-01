"""Flow-matching trainer for the GR00T DINOv3 specialist on a converted LeRobot dataset.

Loads a GR00T LeRobot dataset (from convert_bc_to_gr00t.py), builds the Gr00tN1d7Specialist
(DINOv3 encoder + ~50M action expert), and trains the real flow-matching loss on real
(image, state, action-chunk) samples. Implements the model's training contract directly
(it does NOT use finetune.sh, which needs torchcodec + a registered embodiment DataConfig):

  state  -> normalize (stats.json), pad 29->max_state_dim, history=1   -> [B, 1, 132]
  action -> chunk actions[t:t+horizon] (no cross-episode), normalize,
            pad dim+horizon, + action_mask                              -> [B, 40, 132]
  image  -> resize 256 + ImageNet-normalize                            -> pixel_values [B,3,256,256]
  embodiment_id = 0

SCALABILITY: instead of holding every frame in RAM (~400GB for 1000 demos), all frames are
decoded ONCE into a disk memmap (uint8 [N,3,256,256]); training reads single frames lazily.

Outputs under <out>/<name>/: <name>.pt (latest), <name>_step{N}.pt, loss_curve.csv, loss_curve.png.

Run (GR00T venv):
  cd Isaac-GR00T && source .venv/bin/activate
  python /home/cning/simtoolreal_isaaclab/scripts/train_specialist.py \
      --dataset .../hammer_gr00t_lerobot_full --name hammer --steps 5000 --batch_size 32
"""

import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import gr00t.model as _gr00t_model
from gr00t.model.gr00t_n1d7.gr00t_n1d7_specialist import (
    Gr00tN1d7Specialist,
    make_specialist_config,
)

# The pretrained DINOv3 encoder weights ship alongside the model package
# (gr00t/model/pretrained/), so the default resolves wherever gr00t is installed.
DEFAULT_DINOV3_WEIGHTS = os.path.join(
    os.path.dirname(_gr00t_model.__file__), "pretrained", "dinov3_vits16plus.pth"
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="GR00T LeRobot dataset dir (from convert_bc_to_gr00t.py)")
    p.add_argument("--max_episodes", type=int, default=0, help="cap episodes used (0=all) -> bounds the framecache disk size")
    p.add_argument("--no_teleport_mask", action="store_true", help="disable masking the action-chunk loss past a teleport (default: mask, when the dataset has the per-frame 'teleport' column). Ablation only")
    p.add_argument("--name", default="hammer")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--val_frac", type=float, default=0.05, help="fraction of episodes held out for val loss")
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--tune_visual", action=argparse.BooleanOptionalAction, default=True,
                   help="train the DINOv3 trunk too (default ON; pass --no-tune_visual to freeze it)")
    p.add_argument("--backbone_lr_mult", type=float, default=0.1,
                   help="LR multiplier for the DINOv3 trunk vs the action head (protects pretrained features)")
    p.add_argument("--pretrained_backbone", action="store_true", help="load DINOv3 weights from timm hub (download)")
    p.add_argument("--dinov3_weights", default=DEFAULT_DINOV3_WEIGHTS,
                   help="local official DINOv3 .pth to init the encoder ('' to skip). "
                        "Default: gr00t/model/pretrained/dinov3_vits16plus.pth. Tuned during training (--tune_visual, default on)")
    p.add_argument("--cache_dir", default="", help="frame-cache dir (default <dataset>/.framecache_<img>)")
    p.add_argument("--out", default="/home/cning/simtoolreal_isaaclab/logs/gr00t_specialist")
    p.add_argument("--resume", default="", help="checkpoint .pt to resume from")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _read_teleport_flags(pqs):
    """Read the per-frame 'teleport' column from each parquet (cheap). Returns (tp_all[N] f32, has_tp)."""
    import pandas as pd
    tps, has_tp = [], False
    for pq in pqs:
        df = pd.read_parquet(pq)
        if "teleport" in df.columns:
            has_tp = True; tps.append(df["teleport"].values.astype(np.float32))
        else:
            tps.append(np.zeros(len(df), np.float32))
    return np.concatenate(tps).astype(np.float32), has_tp


def build_or_load_cache(dataset, cache_dir, img_size, max_episodes=0):
    """Decode every episode video ONCE -> uint8 memmap [N,V,3,img,img]; gather state/action/bounds.

    Returns (frames_memmap, states[N,D], actions[N,D], ep_bounds list[(lo,hi)], info dict).
    max_episodes>0 caps the number of episodes (e.g. to fit the framecache on disk).
    """
    import av
    import cv2
    import pandas as pd

    os.makedirs(cache_dir, exist_ok=True)
    frames_path = os.path.join(cache_dir, "frames_u8.dat")
    meta_path = os.path.join(cache_dir, "meta.npz")

    pqs = sorted(glob.glob(f"{dataset}/data/chunk-*/episode_*.parquet"))
    if max_episodes and max_episodes > 0:
        pqs = pqs[:max_episodes]
    assert pqs, f"no parquet episodes under {dataset}/data"
    # video views from modality.json (e.g. front, wrist), in declared order -> [N, V, 3, img, img]
    mod = json.load(open(f"{dataset}/meta/modality.json"))
    view_orig = [mod["video"][k]["original_key"] for k in mod["video"]]
    V = len(view_orig)
    fp = hashlib.md5(("|".join(f"{p}:{os.path.getsize(p)}" for p in pqs) + f"|{img_size}|V{V}").encode()).hexdigest()

    if os.path.exists(meta_path) and os.path.exists(frames_path):
        meta = np.load(meta_path, allow_pickle=True)
        if str(meta["fingerprint"]) == fp:
            N, Vc = int(meta["N"]), int(meta["V"])
            frames = np.memmap(frames_path, dtype=np.uint8, mode="r", shape=(N, Vc, 3, img_size, img_size))
            if "tp_all" in meta.files:
                tp_all = meta["tp_all"].astype(np.float32); has_tp = bool(meta["has_tp"])
            else:  # old cache built before teleport flags were stored -> read from parquets (cheap)
                tp_all, has_tp = _read_teleport_flags(pqs)
            print(f"[cache] reuse {frames_path}: N={N} frames x V={Vc} views"
                  + (f" | teleport {tp_all.mean()*100:.2f}%" if has_tp else " | no teleport column"), flush=True)
            return (frames, torch.tensor(meta["states"]), torch.tensor(meta["actions"]),
                    [tuple(b) for b in meta["bounds"]], {"N": N, "V": Vc}, tp_all, has_tp)

    # pass 1: lengths + state/action + per-view video paths (parquet is small)
    states, actions, teleports, lengths, vidsets = [], [], [], [], []
    has_tp = False
    for pq in pqs:
        ep = int(pq.split("episode_")[1].split(".")[0])
        df = pd.read_parquet(pq)
        paths = [glob.glob(f"{dataset}/videos/chunk-*/{ok}/episode_{ep:06d}.mp4")[0] for ok in view_orig]
        states.append(np.stack(df["observation.state"].values).astype(np.float32))
        actions.append(np.stack(df["action"].values).astype(np.float32))
        if "teleport" in df:
            has_tp = True
            teleports.append(df["teleport"].values.astype(np.float32))
        else:
            teleports.append(np.zeros(len(df), np.float32))
        lengths.append(len(df)); vidsets.append(paths)
    bounds, off = [], 0
    for L in lengths:
        bounds.append((off, off + L)); off += L
    N = off
    states = np.concatenate(states); actions = np.concatenate(actions); tp_all = np.concatenate(teleports)
    print(f"[cache] building {frames_path}: {len(pqs)} eps, N={N} frames x V={V} views @ {img_size}px "
          f"(~{N*V*3*img_size*img_size/1e9:.1f} GB)", flush=True)

    frames = np.memmap(frames_path, dtype=np.uint8, mode="w+", shape=(N, V, 3, img_size, img_size))
    t0 = time.time()
    for i, (paths, (lo, hi)) in enumerate(zip(vidsets, bounds)):
        T = hi - lo
        for v, vid in enumerate(paths):
            c = av.open(vid); j = 0; last = None
            for fr in c.decode(video=0):
                if j >= T:
                    break
                im = cv2.resize(fr.to_ndarray(format="rgb24"), (img_size, img_size), interpolation=cv2.INTER_AREA)
                chw = np.transpose(im, (2, 0, 1))
                frames[lo + j, v] = chw; last = chw; j += 1
            c.close()
            while j < T and last is not None:  # guard: video short -> repeat last frame
                frames[lo + j, v] = last; j += 1
        if (i + 1) % 50 == 0 or i + 1 == len(vidsets):
            print(f"[cache]  decoded {i+1}/{len(vidsets)} eps ({lo+T}/{N} frames, {time.time()-t0:.0f}s)", flush=True)
    frames.flush()
    np.savez(meta_path, fingerprint=fp, N=N, V=V, states=states, actions=actions,
             bounds=np.array(bounds), img_size=img_size, tp_all=tp_all, has_tp=has_tp)
    return frames, torch.tensor(states), torch.tensor(actions), bounds, {"N": N, "V": V}, tp_all, has_tp


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base * (1 + np.cos(np.pi * min(1.0, prog)))


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    cache_dir = args.cache_dir or os.path.join(args.dataset, f".framecache_{args.img_size}")

    # ---- data ----
    frames, states, actions, bounds, info, tp_all, has_tp = build_or_load_cache(args.dataset, cache_dir, args.img_size, args.max_episodes)
    N, V = info["N"], info["V"]  # V camera views -> pixel_values [B,V,3,H,W] (backbone concats their tokens)
    stats = json.load(open(f"{args.dataset}/meta/stats.json"))
    s_mean = torch.tensor(stats["observation.state"]["mean"], dtype=torch.float32, device=dev)
    s_std = torch.tensor(stats["observation.state"]["std"], dtype=torch.float32, device=dev)
    a_mean = torch.tensor(stats["action"]["mean"], dtype=torch.float32, device=dev)
    a_std = torch.tensor(stats["action"]["std"], dtype=torch.float32, device=dev)
    data_dim = s_mean.numel()
    states = states.to(dev); actions = actions.to(dev)
    states_n = (states - s_mean) / s_std
    actions_n = (actions - a_mean) / a_std
    TP = torch.as_tensor(tp_all, dtype=torch.float32, device=dev)   # [N] per-step teleport flag (0/1)
    mask_teleport = has_tp and (not args.no_teleport_mask)          # mask chunk-loss past a teleport?
    if has_tp:
        print(f"[train] teleport flag present | chunk-loss masking: {'ON' if mask_teleport else 'OFF'} "
              f"({float(tp_all.mean())*100:.2f}% of frames are teleports)", flush=True)
    img_mean = IMAGENET_MEAN.to(dev); img_std = IMAGENET_STD.to(dev)

    # per-frame: which episode end (for action-chunk clipping) + train/val membership
    frame_ep_end = torch.zeros(N, dtype=torch.long, device=dev)
    n_ep = len(bounds)
    rng = np.random.default_rng(args.seed)
    n_val = int(round(args.val_frac * n_ep))
    if args.val_frac > 0 and n_ep >= 5:
        n_val = max(1, n_val)
    n_val = min(n_val, n_ep - 1)  # always keep >=1 train episode
    val_eps = set(rng.choice(n_ep, n_val, replace=False).tolist()) if n_val > 0 else set()
    train_idx, val_idx = [], []
    for e, (lo, hi) in enumerate(bounds):
        frame_ep_end[lo:hi] = hi
        (val_idx if e in val_eps else train_idx).extend(range(lo, hi))
    train_idx = torch.tensor(train_idx); val_idx = torch.tensor(val_idx)
    print(f"[train] N={N} frames | {n_ep} eps ({n_ep-len(val_eps)} train / {len(val_eps)} val) "
          f"| train_frames={len(train_idx)} val_frames={len(val_idx)} | data_dim={data_dim}", flush=True)

    # ---- model ----
    pretrained_ok = args.pretrained_backbone
    weights = args.dinov3_weights if args.dinov3_weights and os.path.exists(args.dinov3_weights) else None
    if args.dinov3_weights and weights is None:
        print(f"[train] WARNING: --dinov3_weights {args.dinov3_weights} not found; falling back", flush=True)
    try:
        cfg = make_specialist_config(image_size=args.img_size, pretrained_backbone=pretrained_ok,
                                     dinov3_weights=weights)
        model = Gr00tN1d7Specialist(cfg)
        if weights:
            print(f"[train] DINOv3 encoder initialized from {weights}", flush=True)
    except Exception as e:
        print(f"[train] DINOv3 weight load failed ({type(e).__name__}: {str(e)[:120]}); "
              f"falling back to random init + tune_visual", flush=True)
        pretrained_ok = False; args.tune_visual = True
        cfg = make_specialist_config(image_size=args.img_size, pretrained_backbone=False, dinov3_weights=None)
        model = Gr00tN1d7Specialist(cfg)
    if args.tune_visual:
        model.backbone.set_trainable_parameters(True)
    model = model.to(dev).train()
    H, Smax, Amax = cfg.action_horizon, cfg.max_state_dim, cfg.max_action_dim
    n_tot = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] specialist {n_tot/1e6:.1f}M params, {n_tr/1e6:.1f}M trainable | "
          f"DINOv3 pretrained={pretrained_ok} tune_visual={args.tune_visual}", flush=True)
    # discriminative LR: the pretrained DINOv3 trunk trains at backbone_lr_mult x the head LR
    bb = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    hd = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    groups = [{"params": hd, "base_lr": args.lr}]
    if bb:
        groups.append({"params": bb, "base_lr": args.lr * args.backbone_lr_mult})
    for g in groups:
        g["lr"] = g["base_lr"]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    print(f"[train] opt: head {sum(p.numel() for p in hd)/1e6:.1f}M @ lr={args.lr:.1e}"
          + (f" + backbone {sum(p.numel() for p in bb)/1e6:.1f}M @ lr={args.lr*args.backbone_lr_mult:.1e}" if bb else " (backbone frozen)"),
          flush=True)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model_state_dict"]); start_step = ck.get("step", 0)
        if "opt_state_dict" in ck:
            opt.load_state_dict(ck["opt_state_dict"])
        print(f"[train] resumed from {args.resume} @ step {start_step}", flush=True)

    def make_batch(pool):
        sel = pool[torch.randint(0, len(pool), (args.batch_size,))]
        idx = sel.cpu().numpy()
        px = torch.from_numpy(np.ascontiguousarray(frames[idx])).to(dev).float() / 255.0
        px = (px - img_mean) / img_std
        state = torch.zeros(args.batch_size, 1, Smax, device=dev)
        action = torch.zeros(args.batch_size, H, Amax, device=dev)
        mask = torch.zeros(args.batch_size, H, Amax, device=dev)
        for b, t in enumerate(sel.tolist()):
            ep_end = int(frame_ep_end[t]); valid = min(H, ep_end - t)
            state[b, 0, :data_dim] = states_n[t]
            action[b, :valid, :data_dim] = actions_n[t:t + valid]
            mask[b, :valid, :data_dim] = 1.0
            if mask_teleport and valid > 0:   # zero chunk-loss past the first teleport (unpredictable recovery)
                tpc = TP[t:t + valid]
                prior = torch.cat([tpc.new_zeros(1), torch.cumsum(tpc, 0)[:-1]])
                mask[b, :valid, :data_dim] *= (prior == 0).float()[:, None]
        return {"pixel_values": px, "state": state, "action": action,
                "action_mask": mask, "embodiment_id": torch.zeros(args.batch_size, dtype=torch.long, device=dev)}

    @torch.no_grad()
    def val_loss(nb=12):
        model.eval()
        ls = [model(make_batch(val_idx))["loss"].item() for _ in range(nb)]
        model.train()
        return float(np.mean(ls))

    # ---- train ----
    out_dir = f"{args.out}/{args.name}"; os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/loss_curve.csv"
    csv = open(csv_path, "a" if start_step else "w")
    if not start_step:
        csv.write("step,loss,loss_ema,val_loss,lr\n")
    ema, hist, t0 = None, [], time.time()
    for step in range(start_step + 1, args.steps + 1):
        frac = lr_at(step, args.steps, 1.0, args.warmup)  # cosine schedule as a [0,1] fraction
        for pg in opt.param_groups:
            pg["lr"] = frac * pg["base_lr"]
        lr = frac * args.lr  # head LR (for logging)
        out = model(make_batch(train_idx))
        loss = out["loss"]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        lv = loss.item()
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        hist.append((step, lv, ema))
        vloss = ""
        if len(val_idx) and (step % args.eval_every == 0 or step == 1):
            vloss = val_loss()
            sps = step / (time.time() - t0)
            print(f"[train] step {step:5d}/{args.steps} loss={lv:.4f} ema={ema:.4f} "
                  f"val={vloss:.4f} lr={lr:.2e} ({sps:.1f} it/s)", flush=True)
        csv.write(f"{step},{lv:.5f},{ema:.5f},{vloss},{lr:.6e}\n"); csv.flush()
        if step % args.ckpt_every == 0 or step == args.steps:
            save_ckpt(f"{out_dir}/{args.name}.pt", model, opt, cfg, stats, data_dim, step, hist)
    csv.close()
    save_ckpt(f"{out_dir}/{args.name}.pt", model, opt, cfg, stats, data_dim, args.steps, hist)
    plot_curve(csv_path, f"{out_dir}/loss_curve.png", args.name)
    print(f"\n[train] DONE: {args.steps} steps | first ema {hist[0][2]:.4f} -> last ema {hist[-1][2]:.4f} "
          f"(min raw {min(h[1] for h in hist):.4f}) | curve -> {out_dir}/loss_curve.png", flush=True)


def save_ckpt(path, model, opt, cfg, stats, data_dim, step, hist):
    torch.save({
        "model_state_dict": model.state_dict(),
        "opt_state_dict": opt.state_dict(),
        "config": cfg.to_dict(),
        "stats": {k: stats[k] for k in ["observation.state", "action"]},
        "data_dim": data_dim, "step": step,
        "loss_ema_last": hist[-1][2] if hist else None,
    }, path)


def plot_curve(csv_path, png_path, name):
    import matplotlib
    matplotlib.use("Agg")
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
    ax.set_xlabel("step"); ax.set_ylabel("flow-matching loss"); ax.set_title(f"specialist '{name}' training")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(png_path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
