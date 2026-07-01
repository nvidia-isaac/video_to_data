"""Validate a trained native Vega-right SAPG policy on the DEPLOY / data-collection task (nail-driving,
no perturbation): run N episodes, report the nail_driven success rate, and save per-episode SUCCESS
(green) + FAILURE (red) videos. Same env the retarget collection used, but native (compat=False) + the
native policy driving the right arm directly.

Run (GPU free after training):
  source ~/isaaclab/env_isaaclab/bin/activate && cd ~/isaaclab/IsaacLab && \
  OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/validate_vega_policy.py --headless \
     --checkpoint ~/simtoolreal_isaaclab/logs/simtoolreal/00_vega_right_v4/nn/00_vega_right_v4.pth \
     --episodes 100 --num_envs 25
"""
import argparse, math
from isaaclab.app import AppLauncher

REPO = "/home/cning/simtoolreal_isaaclab"
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=f"{REPO}/logs/simtoolreal/00_vega_right_v4/nn/00_vega_right_v4.pth")
parser.add_argument("--agent_cfg", default="rl_games_sapg_cfg.yaml")
parser.add_argument("--num_envs", type=int, default=25)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--max_steps", type=int, default=60000)
parser.add_argument("--video_envs", type=int, default=8, help="buffer per-episode frames for the first N envs (memory)")
parser.add_argument("--max_success_vids", type=int, default=12)
parser.add_argument("--max_fail_vids", type=int, default=12)
parser.add_argument("--video_max_frames", type=int, default=1300)
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cam_eye", type=str, default="0.95,-1.75,1.45")
parser.add_argument("--cam_lookat", type=str, default="0.0,-0.50,1.00")
parser.add_argument("--cam_focal", type=float, default=22.0)
parser.add_argument("--markers", action="store_true", default=True, help="draw object(green)/goal(blue/red) keypoints")
parser.add_argument("--out_dir", default=f"{REPO}/videos/vega_right_validate")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import os, sys  # noqa: E402
sys.path.insert(0, REPO)
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
import importlib  # noqa: E402

import simtoolreal_lab.tasks  # noqa: E402,F401
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

TASK = "Isaac-SimToolReal-Vega-Hammer-Right-Direct-v0"   # gym id (env class = HammerEnv); cfg overridden below
AGENTS_DIR = f"{REPO}/simtoolreal_lab/tasks/simtoolreal/agents"
from simtoolreal_lab.tasks.vega_hammer_right.vega_hammer_right_deploy_env_cfg import VegaHammerRightDeployEnvCfg  # noqa: E402


def sphere(path, color, radius=0.014):
    return VisualizationMarkers(VisualizationMarkersCfg(
        prim_path=path, markers={"s": sim_utils.SphereCfg(
            radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))}))


_FONT = [None]
def label(fr, text, color):
    if _FONT[0] is None:
        try: _FONT[0] = ImageFont.truetype("DejaVuSans-Bold.ttf", max(14, fr.shape[0] // 18))
        except Exception: _FONT[0] = ImageFont.load_default()
    im = Image.fromarray(np.ascontiguousarray(fr)); d = ImageDraw.Draw(im)
    H, W = fr.shape[:2]; bw = max(3, H // 80)
    d.rectangle([0, 0, W - 1, H - 1], outline=color, width=bw)
    x, y = max(2, W // 40), max(2, H // 30)
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        d.text((x + dx, y + dy), text, font=_FONT[0], fill=(0, 0, 0))
    d.text((x, y), text, font=_FONT[0], fill=color)
    return np.asarray(im)


def main():
    torch.manual_seed(args.seed)
    cfg = VegaHammerRightDeployEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.cam_eye = tuple(float(v) for v in args.cam_eye.split(","))
    cfg.cam_lookat = tuple(float(v) for v in args.cam_lookat.split(","))
    cfg.cam_focal = args.cam_focal

    with open(os.path.join(AGENTS_DIR, args.agent_cfg)) as f:
        agent_cfg = yaml.safe_load(f)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(TASK, cfg=cfg, render_mode=None)
    base = env.unwrapped
    cam = base.scene.sensors["per_env_cam"]
    N = base.num_envs
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
    agent_cfg["params"]["config"]["num_actors"] = N
    agent_cfg["params"]["config"].setdefault("player", {})
    agent_cfg["params"]["config"]["player"]["games_num"] = 100000
    agent_cfg["params"]["config"]["player"]["deterministic"] = True
    runner = Runner(IsaacAlgoObserver()); runner.load(agent_cfg); runner.reset()
    player = runner.create_player(); player.restore(args.checkpoint); player.has_batch_dimension = True
    print(f"[val] loaded {args.checkpoint}", flush=True)

    mk = None
    if args.markers:
        mk = (sphere("/Visuals/obj_kp", (0.1, 0.95, 0.1)),
              sphere("/Visuals/goal_kp", (0.2, 0.45, 1.0)),
              sphere("/Visuals/goal_ctr", (1.0, 0.25, 0.2), 0.02))

    def draw():
        if mk is None: return
        eo = base.scene.env_origins
        mk[0].visualize(translations=(base.object_keypoints + eo.unsqueeze(1)).reshape(-1, 3))
        mk[1].visualize(translations=(base.goal_keypoints + eo.unsqueeze(1)).reshape(-1, 3))
        mk[2].visualize(translations=(base.goal_pos + eo))

    os.makedirs(args.out_dir, exist_ok=True)
    nve = min(args.video_envs, N)
    bufs = [[] for _ in range(nve)]
    sv, fv = 0, 0                    # saved success / fail videos
    successes = attempts = steps = 0

    def save_ep(frames, ok, idx):
        color = (60, 220, 60) if ok else (235, 60, 60)
        tag = "SUCCESS" if ok else "FAILURE"
        path = f"{args.out_dir}/{'success' if ok else 'fail'}_{idx:02d}.mp4"
        w = imageio.get_writer(path, fps=args.fps, macro_block_size=None, codec="libx264")
        for fr in frames:
            w.append_data(label(fr, tag, color))
        w.close()
        print(f"[val]  saved {tag} video ({len(frames)}f) -> {path}", flush=True)

    obs = env.reset(); o = obs["obs"]
    if player.is_rnn:
        player.init_rnn()
    draw()
    print(f"[val] rollout: N={N}, target {args.episodes} eps, NO perturbation, tol {cfg.success_tolerance}", flush=True)
    while attempts < args.episodes and steps < args.max_steps:
        a = player.get_action(o, is_deterministic=True)
        obs, _, done, _ = env.step(a); o = obs["obs"]
        nd = base.nail_driven
        rgb = cam.data.output["rgb"][:nve, ..., :3].to(torch.uint8).cpu().numpy()
        for i in range(nve):
            if len(bufs[i]) < args.video_max_frames:
                bufs[i].append(rgb[i])
        d = done.bool()
        if player.is_rnn and bool(d.any()):
            for s in player.states:
                s[:, d, :] = 0.0
        ndone = torch.nonzero(done).flatten().tolist()
        if ndone:
            attempts += len(ndone)
            successes += int(nd[done].sum().item())
            for i in ndone:
                if i < nve:
                    ok = bool(nd[i].item())
                    if ok and sv < args.max_success_vids:
                        save_ep(bufs[i], True, sv); sv += 1
                    elif (not ok) and fv < args.max_fail_vids:
                        save_ep(bufs[i], False, fv); fv += 1
                    bufs[i] = []
        draw(); base.sim.render()
        steps += 1
        if steps % 200 == 0:
            print(f"[val] step={steps} eps={attempts} success={successes} rate={successes/max(1,attempts):.0%} "
                  f"vids[S={sv} F={fv}]", flush=True)
    print(f"\n[val] DONE: {successes}/{attempts} episodes nail-driven "
          f"(success_rate {successes/max(1,attempts):.1%}) | videos -> {args.out_dir} (S={sv} F={fv})", flush=True)
    env.close(); app.close()


if __name__ == "__main__":
    main()
