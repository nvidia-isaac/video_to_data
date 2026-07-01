"""Offline diagnostic: does a trained state-specialist reproduce the EXPERT's recorded actions
on its OWN training frames? Replicates eval_simtoolreal_server.infer() exactly, but feeds known
(obs, action) pairs from the HDF5. If the model reproduces training actions well -> no fitting/
inference bug (0% eval is covariate-shift/dataset). If it fails on training frames -> real bug.

Run (GR00T venv, CPU to avoid touching the GPU):
  python scripts/replay_check.py --checkpoint logs/.../hammer_goal.pt --hdf5 datasets/...new.hdf5 --with_goal
"""
import argparse, h5py, numpy as np, torch
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7_keypoint import Gr00tN1d7KeypointPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--hdf5", required=True)
ap.add_argument("--with_goal", action="store_true", help="append obs/keypoints_rel_goal(12) -> 121 (match a +goal checkpoint)")
ap.add_argument("--n_demos", type=int, default=30)
ap.add_argument("--per_demo", type=int, default=16, help="frames sampled per demo")
ap.add_argument("--device", default="cpu")
args = ap.parse_args()
dev = torch.device(args.device)

ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
cfg = Gr00tN1d7Config(**ck["config"])
model = Gr00tN1d7KeypointPolicy(cfg).to(dev).eval()
model.load_state_dict(ck["model_state_dict"])
st = ck["stats"]
kp_mean = torch.tensor(st["kp_mean"], device=dev).view(1, 1, 3)
kp_std = torch.tensor(st["kp_std"], device=dev).view(1, 1, 3)
p_mean = torch.tensor(st["proprio_mean"], device=dev)
p_std = torch.tensor(st["proprio_std"], device=dev)
a_mean = torch.tensor(st["action_mean"], device=dev)
a_std = torch.tensor(st["action_std"], device=dev)
D = ck["data_dim"]; A = ck.get("action_dim", a_mean.numel()); Smax = cfg.max_state_dim; H = cfg.action_horizon
print(f"[replay] {args.checkpoint} | step={ck.get('step')} data_dim={D} action_dim={A} horizon={H} with_goal={args.with_goal}")
print(f"[replay] checkpoint expects proprio_dim={p_mean.numel()} (D={D}) | a_std mean={float(a_std.mean()):.4f}")

h = h5py.File(args.hdf5, "r")
demos = list(h["data"].keys())[:args.n_demos]
rng = np.random.default_rng(0)

# collect frames: input (kp, proprio[+goal]) at t, target action at t, and chunk targets t..t+H
KP, PR, A0, AK, KOK = [], [], [], [], []   # AK: [n,H,29] chunk targets; KOK: valid chunk length
for d in demos:
    g = h["data"][d]
    kpr = g["obs/keypoints_rel_palm"][:].astype(np.float32)   # (T,8,3)
    pro = g["obs/proprio"][:].astype(np.float32)              # (T,109)
    act = g["actions"][:].astype(np.float32)                  # (T,29)
    if args.with_goal:
        goal = g["obs/keypoints_rel_goal"][:].astype(np.float32)  # (T,12)
        pro = np.concatenate([pro, goal], axis=1)             # (T,121)
    T = len(act)
    ts = rng.choice(T, size=min(args.per_demo, T), replace=False)
    for t in ts:
        KP.append(kpr[t]); PR.append(pro[t]); A0.append(act[t])
        L = min(H, T - t)
        ch = np.zeros((H, 29), np.float32); ch[:L] = act[t:t + L]
        AK.append(ch); KOK.append(L)
h.close()
KP = torch.tensor(np.stack(KP), device=dev); PR = torch.tensor(np.stack(PR), device=dev)
A0 = torch.tensor(np.stack(A0), device=dev); AK = torch.tensor(np.stack(AK), device=dev)
KOK = np.array(KOK); B = KP.shape[0]
assert PR.shape[1] == p_mean.numel(), f"proprio dim {PR.shape[1]} != checkpoint stats {p_mean.numel()} (--with_goal mismatch?)"
print(f"[replay] {B} frames from {len(demos)} demos")

# replicate server.infer() exactly, batched
@torch.no_grad()
def infer(kp_raw, pr_raw):
    kp = (kp_raw - kp_mean) / kp_std
    p = (pr_raw - p_mean) / p_std
    state_in = torch.zeros(kp.shape[0], 1, Smax, device=dev)
    state_in[:, 0, :D] = p
    pred = model.get_action({"keypoints": kp, "state": state_in,
                             "embodiment_id": torch.zeros(kp.shape[0], dtype=torch.long, device=dev)})["action_pred"]
    return pred[:, :, :A] * a_std + a_mean   # [B,H,29] denormalized

preds = []
for i in range(0, B, 64):
    preds.append(infer(KP[i:i+64], PR[i:i+64]))
pred = torch.cat(preds, 0)   # [B,H,29]

# --- first-action (k=0) reproduction: the closed-loop-relevant prediction ---
err0 = (pred[:, 0, :] - A0).abs()                       # [B,29]
mae0 = err0.mean().item()
rel0 = (err0.mean(0) / a_std).mean().item()             # MAE as a fraction of per-dim action std
# correlation per dim
pc = pred[:, 0, :].cpu().numpy(); tc = A0.cpu().numpy()
corr = np.mean([np.corrcoef(pc[:, j], tc[:, j])[0, 1] for j in range(29) if tc[:, j].std() > 1e-6])
print(f"\n[replay] k=0 (first action) reproduction on TRAINING frames:")
print(f"   MAE = {mae0:.4f} | MAE/std = {rel0:.3f}  (well-fit BC: <~0.3; ~1.0 = no better than predicting the mean)")
print(f"   mean per-dim corr(pred,true) = {corr:.3f}  (well-fit: >~0.8)")

# --- chunk reproduction across the horizon (open-loop) ---
print(f"[replay] chunk MAE/std by horizon step k (only frames with valid k):")
for k in [0, 4, 8, 16, 32]:
    m = KOK > k
    if m.sum() == 0: continue
    e = (pred[m, k, :] - AK[torch.tensor(m), k, :]).abs().mean(0)
    print(f"   k={k:2d}: MAE/std={float((e/a_std).mean()):.3f} (n={int(m.sum())})")
