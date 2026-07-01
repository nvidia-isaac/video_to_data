"""Deploy the ORIGINAL pretrained SimToolReal SAPG policy on the Vega RIGHT hand via the shadow-IIWA
retarget env (Isaac-SimToolReal-VegaHammerRetarget-Direct-v0), and render a per-env mp4 to see whether
the right hand performs the hammer task.

The policy was trained on the IIWA + LEFT Sharpa hand. This env drives a virtual ("shadow") IIWA arm
in the policy's trained joint space, mirrors the shadow palm EE pose across the robot's sagittal plane
onto the Vega RIGHT arm (differential IK), passes the 22 hand DOFs through (mirrored), and feeds the
policy a mirrored shadow observation. The LEFT arm is parked. Net build + checkpoint restore mirror
deploy_pretrained.py (so model.pth loads bit-for-bit).

Run: source ~/isaaclab/env_isaaclab/bin/activate && cd ~/isaaclab/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES \
     ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/deploy_vega_retarget.py --headless --num_envs 6 --steps 400
"""

import argparse

from isaaclab.app import AppLauncher

ORIG_REPO = "/home/cning/simtoolreal"
DEFAULT_CONFIG = f"{ORIG_REPO}/pretrained_policy/config.yaml"
DEFAULT_CKPT = f"{ORIG_REPO}/pretrained_policy/model.pth"
LOG_DIR = "/home/cning/simtoolreal_isaaclab/logs/simtoolreal"

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=6)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--warmup", type=int, default=8, help="settle steps before recording (zero action)")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--orig_config", type=str, default=DEFAULT_CONFIG)
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
parser.add_argument("--out", type=str, default="")
parser.add_argument("--success_tolerance", type=float, default=-1.0, help="override goal success_tolerance (kp_tol=1.5x); >0.01 loosens so the goal advances through the lift->reorient->strike phases despite the retarget pose mismatch")
parser.add_argument("--viz_kp", action="store_true", help="draw debug markers: GREEN=left keypoints (empty-half box), RED=screw head, YELLOW=left palm center")
parser.add_argument("--log_ee", action="store_true", help="log + visualize the RIGHT-arm target EE pose (frame marker @ target, orange sphere @ achieved) and dump a per-step npz for jump analysis")
# per-env camera framing (env-local). Defaults = the proven Vega-scene framing from render_vega_random.py.
parser.add_argument("--cam_eye", type=str, default="0.0,-2.5,1.60")     # approved FRONT camera (perfect view)
parser.add_argument("--cam_lookat", type=str, default="0.0,-0.40,1.02")
parser.add_argument("--cam_z_far", type=float, default=7.0)
parser.add_argument("--cam_focal", type=float, default=34.56)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")

import simtoolreal_lab.tasks  # noqa: E402,F401
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

from simtoolreal_lab.tasks.vega_hammer_retarget.vega_hammer_retarget_env_cfg import VegaHammerRetargetEnvCfg  # noqa: E402


def build_agent_cfg(num_envs: int) -> dict:
    """Load the ORIGINAL config.yaml -> a self-contained rl_games cfg (net section UNCHANGED so the
    checkpoint state_dict loads bit-for-bit). Same as deploy_pretrained.build_agent_cfg."""
    with open(args_cli.orig_config) as f:
        params = yaml.safe_load(f)["train"]["params"]
    params["seed"] = args_cli.seed
    params["load_checkpoint"] = True
    params["load_path"] = args_cli.checkpoint
    c = params["config"]
    c["name"] = "00_vega_retarget"
    c["device_name"] = c["device"] = "cuda:0"
    c["multi_gpu"] = False
    c["num_actors"] = num_envs
    c["clip_actions"] = False
    c["max_epochs"] = 1
    c["max_frames"] = 100
    c["expl_coef_block_size"] = max(1, num_envs // 6)
    c["minibatch_size"] = num_envs
    if "central_value_config" in c:
        c["central_value_config"]["minibatch_size"] = num_envs
    c["train_dir"] = LOG_DIR
    c["defer_summaries_sec"] = 5
    c["summaries_interval_sec_min"] = 5
    c["summaries_interval_sec_max"] = 300
    c.setdefault("player", {})
    c["player"]["games_num"] = 128
    c["player"]["deterministic"] = True
    return {"params": params}


def main():
    import math

    torch.manual_seed(args_cli.seed)
    env_cfg = VegaHammerRetargetEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.success_tolerance > 0.0:
        env_cfg.success_tolerance = args_cli.success_tolerance
    env_cfg.per_env_camera = True
    env_cfg.cam_eye = tuple(float(v) for v in args_cli.cam_eye.split(","))
    env_cfg.cam_lookat = tuple(float(v) for v in args_cli.cam_lookat.split(","))
    env_cfg.cam_z_far = args_cli.cam_z_far
    env_cfg.cam_focal = args_cli.cam_focal

    agent_cfg = build_agent_cfg(args_cli.num_envs)
    env = gym.make("Isaac-SimToolReal-VegaHammerRetarget-Direct-v0", cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(env, "cuda:0", math.inf, math.inf)
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(args_cli.checkpoint)
    player.has_batch_dimension = True

    base = env.unwrapped
    base._dbg_retarget = True
    base._viz_left_kp = args_cli.viz_kp
    base._log_ee = args_cli.log_ee
    base._ee_log = []
    # 2nd policy instance (own rnn state) drives the LEFT arm/hand to hold the thread_tester
    if getattr(base.cfg, "left_hold", False):
        player_left = runner.create_player()
        player_left.restore(args_cli.checkpoint)
        player_left.has_batch_dimension = True
        if player_left.is_rnn:
            player_left.init_rnn()
        base.attach_left_policy(player_left)
        print("[deploy] LEFT-hand thread_tester hold: attached 2nd policy instance", flush=True)
    cam = base.scene.sensors["per_env_cam"]
    N = base.num_envs

    def grab():
        rgb = cam.data.output["rgb"][..., :3]
        tiles = [rgb[i].cpu().numpy().astype(np.uint8) for i in range(N)]
        sep = np.full((tiles[0].shape[0], 4, 3), 30, np.uint8)
        out = tiles[0]
        for t in tiles[1:]:
            out = np.concatenate([out, sep, t], axis=1)
        return out

    obs = env.reset()
    o = obs["obs"]
    if player.is_rnn:
        player.init_rnn()
    zero = torch.zeros((N, base.cfg.action_space), device=base.device)
    for _ in range(args_cli.warmup):
        env.step(zero)

    # reset-time geometry: where is the Vega RIGHT palm vs the object/goal? (env-local, env 0;
    # base.palm_center/object_pos/goal_pos are ALREADY env-local in the env)
    base._compute_intermediate_values()
    i = 0
    pc, op, gp = base.palm_center[i], base.object_pos[i], base.goal_pos[i]
    print(f"[geom] right palm_center(env-local)={[round(float(v),3) for v in pc]}", flush=True)
    print(f"[geom] object_pos={[round(float(v),3) for v in op]}  goal_pos={[round(float(v),3) for v in gp]}", flush=True)
    print(f"[geom] palm->object={[round(float(v),3) for v in (op-pc)]}  obj->goal={[round(float(v),3) for v in (gp-op)]}", flush=True)
    print(f"[geom] obs0 finite={bool(torch.isfinite(o).all())} absmax={o.abs().max().item():.2f} "
          f"kp_rel_palm_absmax={o[i,113:125].abs().max().item():.2f} kp_rel_goal_absmax={o[i,125:137].abs().max().item():.2f}", flush=True)
    if getattr(base.cfg, "left_hold", False):
        from isaaclab.utils.math import quat_apply as _qa
        ttp, ttq = base._thread_tester_pose()
        screw = base.screw_head_world[i] - base.scene.env_origins[i]
        ttc = (ttp + _qa(ttq, base._left_grasp_off))[i]
        lp = base._left_palm_center()[0][i]
        print(f"[geom-L] thread_tester={[round(float(v),3) for v in ttp[i]]} screw_head={[round(float(v),3) for v in screw]}", flush=True)
        print(f"[geom-L] screw-rel-tt(world)={[round(float(v),3) for v in (screw-ttp[i])]} "
              f"left_grasp_center={[round(float(v),3) for v in ttc]} left_palm={[round(float(v),3) for v in lp]}", flush=True)

    frames = []
    peak_succ = torch.zeros(N, device=base.device)
    for t in range(args_cli.steps):
        a = player.get_action(o, is_deterministic=True)
        obs, rew, done, info = env.step(a)
        o = obs["obs"]
        peak_succ = torch.maximum(peak_succ, base.successes)
        frames.append(grab())
        if t % 50 == 0:
            kpd = base.keypoints_max_dist                      # max keypoint->goal distance (m); <0.015 advances
            print(f"[t={t}] goal_idx(succ) per-env={[int(v) for v in base.successes.tolist()]} "
                  f"kp->goal[min={kpd.min()*1000:.0f} med={kpd.median()*1000:.0f}]mm "
                  f"lifted={int(base.lifted_object.sum().item())}/{N} objz[0]={base.object_pos[0,2].item():.3f}", flush=True)

    out = args_cli.out or f"/home/cning/simtoolreal_isaaclab/videos/vega_retarget_{N}env.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimwrite(out, frames, fps=args_cli.fps, quality=8)
    imageio.imwrite(out.replace(".mp4", "_frame000.png"), frames[0])
    imageio.imwrite(out.replace(".mp4", "_frameMID.png"), frames[len(frames) // 2])
    print(f"WROTE {out}  frames={len(frames)} size={frames[0].shape}", flush=True)
    print(f"PEAK_SUCCESS per env: {[int(v) for v in peak_succ.tolist()]}", flush=True)
    if args_cli.log_ee and base._ee_log:
        import numpy as _np
        log = base._ee_log
        npz = {k: _np.stack([_np.asarray(r[k]) for r in log]) for k in log[0]}
        path = out.replace(".mp4", "_eelog.npz")
        _np.savez(path, **npz)
        print(f"EE_LOG: {len(log)} steps -> {path}", flush=True)
    print("DEPLOY_OK", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
