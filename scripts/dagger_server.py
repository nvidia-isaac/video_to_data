"""DAgger SERVER for the SimToolReal specialist (run in the GR00T venv).

Holds the learner (Gr00tN1d7KeypointPolicy, the SimToolReal specialist) + its optimizer + a GROWING
replay buffer of (state, EXPERT-relabelled action) episodes, and serves the DAgger loop over TCP:

  ping
  act(keypoints[B,8,3], proprio[B,109])         -> action_chunk[B,40,29]   (drive the online rollout)
  add(episodes=[{kp[T,8,3],proprio[T,109],label[T,29]}, ...])              (aggregate relabelled data)
  train(steps, batch_size, lr)                  -> mean_loss               (retrain on the buffer)
  save(path)                                                              (write an eval-loadable ckpt)

Normalization stats are FIXED (loaded from the warm-start BC checkpoint) so the warm-started weights
stay valid as the buffer's distribution shifts toward the learner's. Action chunks are built exactly
like the BC/keypoint trainers: label[t:t+H] along each (learner-visited) episode, pad+mask.

Run:
  cd Isaac-GR00T && source .venv/bin/activate
  python /home/cning/simtoolreal_isaaclab/scripts/dagger_server.py \
      --init_from /home/cning/simtoolreal_isaaclab/logs/gr00t_specialist/hammer_str/hammer_str.pt \
      --seed_hdf5 /home/cning/simtoolreal_isaaclab/datasets/hammer_str_1000.hdf5 --port 5603
"""

import argparse
import os
import pickle
import socket
import struct

import h5py
import numpy as np
import torch

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7_keypoint import Gr00tN1d7KeypointPolicy, make_simtoolreal_config

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def recvall(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


def recv_msg(sock):
    hdr = recvall(sock, 4)
    return None if hdr is None else pickle.loads(recvall(sock, struct.unpack(">I", hdr)[0]))


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack(">I", len(data)) + data)


class Buffer:
    """Growing buffer of NORMALIZED episodes (list-based: O(1) append, index-based chunk sampling)."""

    def __init__(self, dev, H, kp_mean, kp_std, p_mean, p_std, a_mean, a_std, rh=0):
        self.dev, self.H = dev, H
        self.rh = rh if (rh and rh > 0) else H   # chunk positions SUPERVISED per state (1 = single-action DAgger; 0/default = full H)
        self.kp_mean, self.kp_std = kp_mean, kp_std
        self.p_mean, self.p_std = p_mean, p_std
        self.a_mean, self.a_std = a_mean, a_std
        self.eps = []                                        # list of (kp[T,8,3], pr[T,D], ac[T,A]) normalized
        self.lengths = []
        self.cum = torch.zeros(1, dtype=torch.long, device=dev)   # cumulative frame counts (len = n_eps+1)
        self.n_eps = 0

    def add(self, kp, proprio, label):
        """Append one episode (raw np arrays); normalize with the FIXED stats. O(1)."""
        kp = torch.as_tensor(kp, device=self.dev, dtype=torch.float32)
        pr = torch.as_tensor(proprio, device=self.dev, dtype=torch.float32)
        ac = torch.as_tensor(label, device=self.dev, dtype=torch.float32)
        T = min(len(kp), len(pr), len(ac))
        if T < 1:
            return
        kp = (kp[:T] - self.kp_mean.view(1, 1, 3)) / self.kp_std.view(1, 1, 3)
        pr = (pr[:T] - self.p_mean) / self.p_std
        ac = (ac[:T] - self.a_mean) / self.a_std
        self.eps.append((kp, pr, ac)); self.lengths.append(T)
        self.cum = torch.cat([self.cum, self.cum[-1:] + T])
        self.n_eps += 1

    def __len__(self):
        return int(self.cum[-1].item())

    def batch(self, B, Smax, Amax, data_dim, act_dim):
        total = len(self)
        fidx = torch.randint(0, total, (B,), device=self.dev)         # uniform over ALL frames
        ep = (torch.searchsorted(self.cum, fidx, right=True) - 1).tolist()
        off = (fidx - self.cum[ep]).tolist()
        kpb = torch.empty(B, 8, 3, device=self.dev)
        state = torch.zeros(B, 1, Smax, device=self.dev)
        action = torch.zeros(B, self.H, Amax, device=self.dev)
        mask = torch.zeros(B, self.H, Amax, device=self.dev)
        for b in range(B):
            kp_e, pr_e, ac_e = self.eps[ep[b]]; t = off[b]; T = self.lengths[ep[b]]
            kpb[b] = kp_e[t]
            state[b, 0, :data_dim] = pr_e[t]
            valid = min(self.rh, T - t)   # supervise only the first `rh` chunk positions (rh=1 -> single-action DAgger)
            action[b, :valid, :act_dim] = ac_e[t:t + valid]
            mask[b, :valid, :act_dim] = 1.0
        return {"keypoints": kpb, "state": state, "action": action, "action_mask": mask,
                "embodiment_id": torch.zeros(B, dtype=torch.long, device=self.dev)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", required=True, help="BC checkpoint to warm-start weights + FIXED stats")
    ap.add_argument("--seed_hdf5", default="", help="optional BC dataset to seed the buffer (DAgger keeps D_BC)")
    ap.add_argument("--no_joint_vel", action="store_true", help="drop joint_vel [29:58] from proprio (109->80) to match a --no_joint_vel learner; slices the seed HDF5 (the client sends pre-sliced proprio)")
    ap.add_argument("--relabel_horizon", type=int, default=0, help="chunk positions to SUPERVISE per state: 0=full H=40 (original chunk relabel), 1=single-action textbook DAgger (mask chunk loss past pos 0). The H=40 model is unchanged")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5603)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
    cfg = Gr00tN1d7Config(**ck["config"]) if "config" in ck else make_simtoolreal_config()
    model = Gr00tN1d7KeypointPolicy(cfg).to(dev).train()
    model.load_state_dict(ck["model_state_dict"])
    st = ck["stats"]; dd = ck["data_dim"]; ad = ck.get("action_dim", len(st["action_mean"]))
    nkp = ck.get("n_keypoints", 8)
    H, Smax, Amax = cfg.action_horizon, cfg.max_state_dim, cfg.max_action_dim
    tt = lambda x: torch.tensor(x, device=dev, dtype=torch.float32)
    buf = Buffer(dev, H, tt(st["kp_mean"]), tt(st["kp_std"]), tt(st["proprio_mean"]), tt(st["proprio_std"]),
                 tt(st["action_mean"]), tt(st["action_std"]), rh=args.relabel_horizon)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    print(f"[server] warm-started from {args.init_from} (step {ck.get('step')}) | proprio={dd} act={ad} H={H} relabel_h={buf.rh} dev={dev}", flush=True)

    if args.seed_hdf5:
        f = h5py.File(args.seed_hdf5, "r"); data = f["data"]
        demos = sorted([k for k in data if k.startswith("demo_")], key=lambda s: int(s.split("_")[1]))
        for dk in demos:
            g = data[dk]
            pr = g["obs/proprio"][:]
            if args.no_joint_vel:   # drop joint_vel [29:58] -> 80, matching the learner's proprio
                pr = np.concatenate([pr[:, :29], pr[:, 58:]], axis=1)
            buf.add(g["obs/keypoints_rel_palm"][:], pr, g["actions"][:])
        f.close()
        print(f"[server] seeded buffer from {args.seed_hdf5}: {buf.n_eps} eps / {len(buf)} frames", flush=True)

    @torch.no_grad()
    def act(kp, proprio):
        model.eval()
        k = (torch.from_numpy(kp).to(dev).float() - buf.kp_mean.view(1, 1, 3)) / buf.kp_std.view(1, 1, 3)
        p = (torch.from_numpy(proprio).to(dev).float() - buf.p_mean) / buf.p_std
        B = p.shape[0]
        state = torch.zeros(B, 1, Smax, device=dev); state[:, 0, :dd] = p
        pred = model.get_action({"keypoints": k, "state": state,
                                 "embodiment_id": torch.zeros(B, dtype=torch.long, device=dev)})["action_pred"]
        model.train()
        return (pred[:, :, :ad] * buf.a_std + buf.a_mean).float().cpu().numpy()

    def train(steps, B, lr):
        for pg in opt.param_groups:
            pg["lr"] = lr
        losses = []
        for _ in range(steps):
            loss = model(buf.batch(B, Smax, Amax, dd, ad))["loss"]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")

    def save(path, step):
        torch.save({"model_state_dict": model.state_dict(), "config": cfg.to_dict(), "stats": st,
                    "data_dim": dd, "action_dim": ad, "n_keypoints": nkp, "step": step}, path)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port)); srv.listen(1)
    print(f"[server] listening on {args.host}:{args.port}", flush=True)
    while True:
        conn, addr = srv.accept()
        print(f"[server] client {addr} connected", flush=True)
        try:
            while True:
                req = recv_msg(conn)
                if req is None or req.get("cmd") == "close":
                    break
                c = req["cmd"]
                if c == "ping":
                    send_msg(conn, {"ok": True})
                elif c == "act":
                    send_msg(conn, {"action": act(req["keypoints"], req["proprio"])})
                elif c == "add":
                    for ep in req["episodes"]:
                        buf.add(ep["kp"], ep["proprio"], ep["label"])
                    send_msg(conn, {"n_eps": buf.n_eps, "n_frames": len(buf)})
                elif c == "train":
                    ml = train(int(req["steps"]), int(req["batch_size"]), float(req.get("lr", 1e-4)))
                    send_msg(conn, {"mean_loss": ml})
                elif c == "save":
                    save(req["path"], int(req.get("step", 0)))
                    send_msg(conn, {"saved": req["path"]})
                else:
                    send_msg(conn, {"error": f"unknown cmd {c}"})
        except (ConnectionResetError, BrokenPipeError):
            print("[server] client disconnected", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
