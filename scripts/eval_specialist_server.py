"""Policy SERVER for evaluating the GR00T DINOv3 specialist (run in the GR00T venv).

Loads a specialist checkpoint (from train_specialist.py) and serves action inference over a
plain TCP socket (length-prefixed pickle, numpy arrays) so the Isaac Lab env client can run in
a SEPARATE process/venv (the model needs the GR00T venv; the sim needs the isaaclab venv).

Protocol:
  request : {"cmd": "act", "image": uint8[B,H,W,3], "state": float32[B,29]}
            {"cmd": "ping"} | {"cmd": "close"}
  response: {"action": float32[B, horizon, 29]}   # DENORMALIZED delta joint targets
            {"ok": True}

Preprocessing mirrors training exactly: image -> resize 256 (cv2.INTER_AREA) -> /255 -> ImageNet
normalize; state -> z-score (checkpoint stats) -> pad to max_state_dim. Output action is the
model's predicted action chunk, sliced to 29 dims and de-normalized (x*std + mean), i.e. raw
delta joint targets the client applies as cur_targets = joint_pos + delta.

Run:
  cd Isaac-GR00T && source .venv/bin/activate
  python /home/cning/simtoolreal_isaaclab/scripts/eval_specialist_server.py \
      --checkpoint /home/cning/simtoolreal_isaaclab/logs/gr00t_specialist/hammer/hammer.pt \
      --host 127.0.0.1 --port 5599
"""

import argparse
import pickle
import socket
import struct

import cv2
import numpy as np
import torch

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7_specialist import Gr00tN1d7Specialist

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_msg(sock):
    hdr = recvall(sock, 4)
    if hdr is None:
        return None
    n = struct.unpack(">I", hdr)[0]
    return pickle.loads(recvall(sock, n))


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack(">I", len(data)) + data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Gr00tN1d7Config(**ck["config"])
    cfg.backbone_pretrained = False  # weights come from the checkpoint; skip the timm download
    cfg.dinov3_weights = None        # ditto: the trained encoder is in the checkpoint, don't re-init from .pth
    model = Gr00tN1d7Specialist(cfg)
    model.load_state_dict(ck["model_state_dict"])
    model = model.to(dev).eval()
    img_size = (cfg.image_target_size or [args.img_size])[0]

    st = ck["stats"]
    s_mean = torch.tensor(st["observation.state"]["mean"], device=dev)
    s_std = torch.tensor(st["observation.state"]["std"], device=dev)
    a_mean = torch.tensor(st["action"]["mean"], device=dev)
    a_std = torch.tensor(st["action"]["std"], device=dev)
    D = s_mean.numel()
    Smax = cfg.max_state_dim
    imean, istd = IMAGENET_MEAN.to(dev), IMAGENET_STD.to(dev)
    print(f"[server] loaded {args.checkpoint} | step={ck.get('step')} data_dim={D} horizon={cfg.action_horizon} "
          f"img={img_size} dev={dev}", flush=True)

    @torch.no_grad()
    def infer(images, state):
        # images: list of [B,H,W,3] uint8, one per view (front[, wrist]) in modality.json order
        views = []
        for image in images:
            # resize EXACTLY like training (train_specialist.py): cv2.INTER_AREA on the uint8 HWC
            # frame FIRST, THEN /255 + ImageNet-normalize. (Bilinear in float space did not match.)
            resized = np.stack([cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_AREA)
                                for im in image])              # [B,img_size,img_size,3] uint8
            px = torch.from_numpy(resized).to(dev).permute(0, 3, 1, 2).float() / 255.0
            views.append((px - imean) / istd)
        px = torch.stack(views, dim=1) if len(views) > 1 else views[0]  # [B,V,3,H,W] or [B,3,H,W]
        s = (torch.from_numpy(state).to(dev).float() - s_mean) / s_std
        B = s.shape[0]
        state_in = torch.zeros(B, 1, Smax, device=dev)
        state_in[:, 0, :D] = s
        inputs = {"pixel_values": px, "state": state_in,
                  "embodiment_id": torch.zeros(B, dtype=torch.long, device=dev)}
        pred = model.get_action(inputs)["action_pred"]            # [B, horizon, max_action_dim]
        delta = pred[:, :, :D] * a_std + a_mean                   # de-normalize -> raw deltas
        return delta.float().cpu().numpy()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
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
                    send_msg(conn, {"ok": True})
                    continue
                imgs = [req["image"]] + ([req["image_wrist"]] if "image_wrist" in req else [])
                send_msg(conn, {"action": infer(imgs, req["state"])})
        except (ConnectionResetError, BrokenPipeError):
            print("[server] client disconnected", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
