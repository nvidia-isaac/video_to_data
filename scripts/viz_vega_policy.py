"""Render a 10 s mp4 of the TRAINED Vega-right SimToolReal policy doing the task, with the object
keypoints (GREEN) + goal keypoints (BLUE) + goal center (RED) drawn. Tiles a few envs so we likely
catch a grasp-lift-to-goal (the policy is ~30% per episode). Mirrors play.py's SAPG player setup +
viz_vega_env.py's per-env-camera markers.

Run (training done -> GPU free):
  source ~/isaaclab/env_isaaclab/bin/activate && cd ~/isaaclab/IsaacLab && \
  OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/viz_vega_policy.py --headless \
     --checkpoint ~/simtoolreal_isaaclab/logs/simtoolreal/00_vega_right_v3/nn/00_vega_right_v3.pth
"""
import argparse, math
from isaaclab.app import AppLauncher

REPO = "/home/cning/simtoolreal_isaaclab"
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=f"{REPO}/logs/simtoolreal/00_vega_right_v3/nn/00_vega_right_v3.pth")
parser.add_argument("--agent_cfg", default="rl_games_sapg_cfg.yaml")
parser.add_argument("--num_envs", type=int, default=6)
parser.add_argument("--steps", type=int, default=600, help="control steps (60 Hz); 600 = 10 s")
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--cam_eye", type=str, default="0.95,-1.75,1.45")
parser.add_argument("--cam_lookat", type=str, default="0.0,-0.50,1.00")
parser.add_argument("--cam_focal", type=float, default=22.0)
parser.add_argument("--out", default=f"{REPO}/videos/vega_right_policy.mp4")
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
import importlib  # noqa: E402

import simtoolreal_lab.tasks  # noqa: E402,F401
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

TASK = "Isaac-SimToolReal-Vega-Hammer-Right-Direct-v0"
AGENTS_DIR = f"{REPO}/simtoolreal_lab/tasks/simtoolreal/agents"


def resolve_env_cfg(task):
    entry = gym.spec(task).kwargs["env_cfg_entry_point"]
    m, c = entry.split(":")
    return getattr(importlib.import_module(m), c)


def sphere(path, color, radius=0.014):
    return VisualizationMarkers(VisualizationMarkersCfg(
        prim_path=path, markers={"s": sim_utils.SphereCfg(
            radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))}))


def main():
    torch.manual_seed(args.seed)
    env_cfg = resolve_env_cfg(TASK)()
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.per_env_camera = True
    env_cfg.domain_randomization = False           # clean deterministic rollout
    env_cfg.use_tolerance_curriculum = False
    env_cfg.eval_append_expl_coef = True           # SAPG coef_cond: feed the exploit coef at play
    env_cfg.cam_eye = tuple(float(v) for v in args.cam_eye.split(","))
    env_cfg.cam_lookat = tuple(float(v) for v in args.cam_lookat.split(","))
    env_cfg.cam_focal = args.cam_focal

    with open(os.path.join(AGENTS_DIR, args.agent_cfg)) as f:
        agent_cfg = yaml.safe_load(f)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    cam = base.scene.sensors["per_env_cam"]
    N = base.num_envs

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
    agent_cfg["params"]["config"]["num_actors"] = N
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args.checkpoint
    agent_cfg["params"]["config"].setdefault("player", {})
    agent_cfg["params"]["config"]["player"]["games_num"] = 1024
    agent_cfg["params"]["config"]["player"]["deterministic"] = True
    runner = Runner(IsaacAlgoObserver()); runner.load(agent_cfg); runner.reset()
    player = runner.create_player(); player.restore(args.checkpoint); player.has_batch_dimension = True
    print(f"[viz] loaded {args.checkpoint}", flush=True)

    obj_kp = sphere("/Visuals/object_kp", (0.10, 0.95, 0.10))
    goal_kp = sphere("/Visuals/goal_kp", (0.20, 0.45, 1.00))
    goal_ctr = sphere("/Visuals/goal_ctr", (1.00, 0.25, 0.20), radius=0.020)

    def draw():
        eo = base.scene.env_origins
        obj_kp.visualize(translations=(base.object_keypoints + eo.unsqueeze(1)).reshape(-1, 3))
        goal_kp.visualize(translations=(base.goal_keypoints + eo.unsqueeze(1)).reshape(-1, 3))
        goal_ctr.visualize(translations=(base.goal_pos + eo))

    cols = 3 if N >= 6 else N
    rows = (N + cols - 1) // cols

    def grab():
        rgb = cam.data.output["rgb"][..., :3]
        tiles = [rgb[i].cpu().numpy().astype(np.uint8) for i in range(N)]
        while len(tiles) < rows * cols:
            tiles.append(np.zeros_like(tiles[0]))
        rowimgs = [np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1) for r in range(rows)]
        return np.concatenate(rowimgs, axis=0)

    obs = env.reset(); o = obs["obs"]
    if player.is_rnn:
        player.init_rnn()
    draw()
    frames = []
    lift_frames = torch.zeros(N, device=base.device)
    succ_max = 0
    for t in range(args.steps):
        a = player.get_action(o, is_deterministic=True)
        obs, _, done, _ = env.step(a); o = obs["obs"]
        if player.is_rnn and bool(done.any()):
            d = done.bool()
            for s in player.states:
                s[:, d, :] = 0.0
        lift_frames += base.lifted_object.float()
        succ_max = max(succ_max, int(base.successes.max().item()))
        draw(); base.sim.render()
        frames.append(grab())
    print(f"[viz] lift_rate={base.lifted_object.float().mean().item():.2f} "
          f"frac_envs_lifted_some={(lift_frames>0).float().mean().item():.2f} succ_max_in_clip={succ_max}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=args.fps, quality=8)
    imageio.imwrite(args.out.replace(".mp4", "_frameMID.png"), frames[len(frames) // 2])
    print(f"WROTE {args.out}  frames={len(frames)} size={frames[0].shape} fps={args.fps}", flush=True)
    print("VIZ_OK", flush=True)
    env.close(); app.close()


if __name__ == "__main__":
    main()
