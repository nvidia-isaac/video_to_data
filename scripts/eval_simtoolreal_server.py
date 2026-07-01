"""Policy SERVER for the SimToolReal SPECIALIST (state-based, expert-faithful obs). Run in GR00T venv.

Loads a checkpoint from train_simtoolreal_specialist.py (a Gr00tN1d7KeypointPolicy with a 109-dim
proprio state) and serves action inference over TCP (length-prefixed pickle). The env client sends
raw env-local PALM-RELATIVE keypoints + the 109-dim proprio; the server normalizes with the
checkpoint stats, runs the flow-matching head, and returns de-normalized delta-joint action chunks.

  request : {"cmd":"act", "keypoints": float32[B,8,3], "proprio": float32[B,109]} | {"cmd":"ping"|"close"}
  response: {"action": float32[B, horizon, 29]}

Run:
  cd Isaac-GR00T && source .venv/bin/activate
  python /home/cning/simtoolreal_isaaclab/scripts/eval_simtoolreal_server.py \
      --checkpoint /home/cning/simtoolreal_isaaclab/logs/gr00t_specialist/hammer_str/hammer_str.pt --port 5602
"""

import argparse
import pickle
import socket
import struct

import torch

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7_keypoint import Gr00tN1d7KeypointPolicy


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5602)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Gr00tN1d7Config(**ck["config"])
    model = Gr00tN1d7KeypointPolicy(cfg)
    model.load_state_dict(ck["model_state_dict"])
    model = model.to(dev).eval()
    st = ck["stats"]
    kp_mean = torch.tensor(st["kp_mean"], device=dev).view(1, 1, 3)
    kp_std = torch.tensor(st["kp_std"], device=dev).view(1, 1, 3)
    p_mean = torch.tensor(st["proprio_mean"], device=dev)
    p_std = torch.tensor(st["proprio_std"], device=dev)
    a_mean = torch.tensor(st["action_mean"], device=dev)
    a_std = torch.tensor(st["action_std"], device=dev)
    D = ck["data_dim"]                                   # proprio dim (109)
    A = ck.get("action_dim", a_mean.numel())             # action dim (29)
    Smax = cfg.max_state_dim
    print(f"[server] loaded {args.checkpoint} | step={ck.get('step')} keypoints={ck.get('n_keypoints')} "
          f"proprio_dim={D} action_dim={A} horizon={cfg.action_horizon} dev={dev}", flush=True)

    @torch.no_grad()
    def infer(keypoints, proprio):
        kp = (torch.from_numpy(keypoints).to(dev).float() - kp_mean) / kp_std   # [B,8,3]
        p = (torch.from_numpy(proprio).to(dev).float() - p_mean) / p_std        # [B,109]
        B = p.shape[0]
        state_in = torch.zeros(B, 1, Smax, device=dev)
        state_in[:, 0, :D] = p
        inputs = {"keypoints": kp, "state": state_in,
                  "embodiment_id": torch.zeros(B, dtype=torch.long, device=dev)}
        pred = model.get_action(inputs)["action_pred"]            # [B, horizon, max_action_dim]
        return (pred[:, :, :A] * a_std + a_mean).float().cpu().numpy()

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
                if req.get("cmd") == "ping":
                    send_msg(conn, {"ok": True}); continue
                send_msg(conn, {"action": infer(req["keypoints"], req["proprio"])})
        except (ConnectionResetError, BrokenPipeError):
            print("[server] client disconnected", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
