"""Kinematically play the tighten GOAL poses (no policy / no physics control).

Teleports the screwdriver through its per-env goal sequence (`per_env_goals` — lift -> reorient
tip-down + blade-to-slot -> over screw -> lower to contact -> rotate) so you can SEE whether the
goal poses make sense relative to the screw + thread_test. The robot is parked out of the way and
the screwdriver is made kinematic (set-pose, no fall). Records an RTX mp4.

Run:
  cd IsaacLab && (venv) OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
    ~/simtoolreal_isaaclab/scripts/visualize_goals.py --headless --demo_task tighten_screw
"""

import argparse
from isaaclab.app import AppLauncher

ORIG_REPO = "/home/cning/simtoolreal"

parser = argparse.ArgumentParser()
parser.add_argument("--goal_generator", type=str, default="", help="override the env's goal_generator_module (e.g. ...outer_contact_traj043 for the WRONG-way rim-contact replay; empty = the cfg default)")
parser.add_argument("--object_scale", type=float, default=0.0, help="override the screwdriver (Object) uniform spawn scale AND the goal generator's TIP (=0.134*scale) in lockstep. e.g. 2.16 keeps the base-043 tip/slot ratio against the +80% screw. 0 = cfg default")
parser.add_argument("--contact_clearance", type=float, default=999.0, help="override screw_contact_clearance (m): tip seat rel. to head_offset along +z. NEGATIVE seats INTO the slot; small (shallow) = light engagement in the wide opening (avoids wedging where the slot narrows). 999 = cfg default")
parser.add_argument("--wide_slot", action="store_true", help="(screwdriver043) swap in the 60%-wider cross-slot screw + assembly (screw_new_wideslot_sdf / screw_assembly043_wideslot)")
parser.add_argument("--rim_r", type=float, default=0.0, help="(outer_contact_traj only) radial offset of the rim contact from the screw axis (m). Set from the head profile so the tip lands on the cone OUTSIDE the slot. 0 = module default")
parser.add_argument("--rim_axial", type=float, default=999.0, help="(outer_contact_traj only) tip height rel. to the slot ref (m); set NEGATIVE to press the tip into the domed cone surface so it actually contacts. 999 = module default")
parser.add_argument("--env", type=str, default="screwdriver", choices=["screwdriver", "screwdriver_aligned", "screwdriver043", "screwdriver043big"],
                    help="which screwdriver env's goals to visualize (044 flat slot, or 043 cross slot)")
parser.add_argument("--demo_task", type=str, default="tighten_screw")
parser.add_argument("--hold", type=int, default=5, help="render frames held at each goal pose")
parser.add_argument("--cam_eye", type=str, default="-0.45,-0.50,0.78")
parser.add_argument("--cam_lookat", type=str, default="0.04,0.0,0.64")
parser.add_argument("--randomize_layout", action="store_true", help="randomize the screw/screwdriver layout (goals adapt)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--physical_screw", action="store_true", help="use the PHYSICAL revolute-jointed screw (articulation) so the teleported screwdriver drives it via real contact -- shows how a PERFECT goal trajectory tightens the screw")
parser.add_argument("--screw_friction", type=float, default=0.05, help="(physical screw) revolute-joint friction")
parser.add_argument("--screw_damping", type=float, default=0.08, help="(physical screw) revolute-joint damping")
parser.add_argument("--num_envs", type=int, default=1, help=">1: multi-env mode -- random layout per env, one per-env camera clip each, reports per-env screw rotation (verify the goal trajectory is correct across scenes)")
parser.add_argument("--cam_width", type=int, default=900)
parser.add_argument("--cam_height", type=int, default=600)
parser.add_argument("--no_clips", action="store_true", help="(multi-env) skip writing per-env mp4s -- fast diagnostic: just teleport + report per-env screw rotation vs layout features")
parser.add_argument("--seat_frames", type=int, default=0, help="hold the tip at the contact pose for N frames BEFORE the rotate phase, so the physical cross seats into the slot (robust engagement)")
parser.add_argument("--rotate_hold", type=int, default=0, help="override --hold ONLY for the rotate phase (>0 -> slower in-slot rotation -> less cam-out/slip)")
parser.add_argument("--record_envs", type=str, default="", help="(multi-env) comma list of env indices to WRITE clips for (default empty = all). All cameras still render; only these are encoded -- avoids OOM from 100 simultaneous writers.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # rendering

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")
import simtoolreal_lab.tasks  # noqa: E402, F401
from simtoolreal_lab.tasks.screwdriver.screwdriver_env_cfg import ScrewdriverEnvCfg  # noqa: E402
from simtoolreal_lab.tasks.screwdriver043.screwdriver043_env_cfg import Screwdriver043EnvCfg  # noqa: E402
from simtoolreal_lab.tasks.screwdriver043.screwdriver043big_env_cfg import Screwdriver043BigEnvCfg  # noqa: E402
from simtoolreal_lab.tasks.screwdriver.screwdriver_aligned_env_cfg import ScrewdriverAlignedEnvCfg  # noqa: E402

VIDEO_DIR = "/home/cning/simtoolreal_isaaclab/videos"

# *big / *aligned reuse their parent task_id (same ScrewdriverEnv class); only the cfg differs (cfg= is
# passed to gym.make), so no separate gym registration is needed.
ENV_MAP = {
    "screwdriver":         (ScrewdriverEnvCfg,        "Isaac-SimToolReal-Screwdriver-Direct-v0"),
    "screwdriver_aligned": (ScrewdriverAlignedEnvCfg, "Isaac-SimToolReal-Screwdriver-Direct-v0"),
    "screwdriver043":      (Screwdriver043EnvCfg,     "Isaac-SimToolReal-Screwdriver043-Direct-v0"),
    "screwdriver043big":   (Screwdriver043BigEnvCfg,  "Isaac-SimToolReal-Screwdriver043-Direct-v0"),
}


def run_multi(cfg, task_id):
    """Multi-env: each env a random layout; teleport every screwdriver through ITS perfect goals;
    record one per-env clip + report per-env screw rotation -> verify the target across scenes."""
    import imageio.v2 as imageio
    N = args_cli.num_envs
    # moderate layout diversity (tighter than the wide policy-eval ranges) so the screw stays in a
    # ~10cm region -> a real zoom can frame it in EVERY scene while still varying pos + yaw.
    cfg.layout_threadtest_center_x_range = (0.14, 0.22)
    cfg.layout_threadtest_center_y_range = (-0.05, 0.05)
    cfg.layout_yaw_range = (-0.35, 0.35)     # +/-20 deg
    cfg.per_env_camera = not args_cli.no_clips   # diagnostic mode skips cameras entirely (fast)
    cfg.cam_width = args_cli.cam_width
    cfg.cam_height = args_cli.cam_height
    cfg.cam_eye = tuple(float(v) for v in args_cli.cam_eye.split(","))
    cfg.cam_lookat = tuple(float(v) for v in args_cli.cam_lookat.split(","))
    env = gym.make(task_id, cfg=cfg, render_mode=None)
    env.reset()
    base = env.unwrapped
    goals = base.per_env_goals.clone()           # (N,T,7) xyz + xyzw, env-local
    T = goals.shape[1]
    pe_cam = None if args_cli.no_clips else base.scene.sensors["per_env_cam"]
    PE_DIR = f"{VIDEO_DIR}/goalviz_per_env"
    os.makedirs(PE_DIR, exist_ok=True)
    for f in os.listdir(PE_DIR):
        if f.endswith(".mp4"):
            os.remove(os.path.join(PE_DIR, f))
    rec_ids = range(N) if not args_cli.record_envs else [int(x) for x in args_cli.record_envs.split(",")]
    writers = {} if args_cli.no_clips else {i: imageio.get_writer(f"{PE_DIR}/env_{i:03d}.mp4", fps=30, macro_block_size=1) for i in rec_ids}

    eo = base.scene.env_origins                  # (N,3)
    all_ids = torch.arange(N, device=base.device)
    zero_act = torch.zeros((N, base.cfg.action_space), device=base.device)
    z6 = torch.zeros((N, 6), device=base.device)
    gp = goals[:, :, 0:3] + eo.unsqueeze(1)      # (N,T,3) world
    gq = goals[:, :, [6, 3, 4, 5]]               # (N,T,4) wxyz
    SUB = args_cli.hold
    phys = args_cli.physical_screw and getattr(base, "screw_asm", None) is not None

    def step_all(pos, quat):
        quat = quat / quat.norm(dim=-1, keepdim=True)
        base.object.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), all_ids)
        base.object.write_root_velocity_to_sim(z6, all_ids)
        env.step(zero_act)
        if pe_cam is not None:
            rgb = pe_cam.data.output["rgb"]
            for i in writers:
                fr = rgb[i].cpu().numpy()
                if fr.shape[-1] == 4:
                    fr = fr[..., :3]
                writers[i].append_data(fr.astype("uint8"))

    ROTATE_START = T - 24            # the last 24 goals are the in-slot rotation phase
    print(f"[viz] MULTI-ENV: {N} scenes, {T} goals x {SUB} frames/seg{' | PHYSICAL screw' if phys else ''}", flush=True)
    for _ in range(SUB):
        step_all(gp[:, 0], gq[:, 0])
    for i in range(T - 1):
        if i == ROTATE_START - 1 and args_cli.seat_frames > 0:
            for _ in range(args_cli.seat_frames):     # SEAT the cross into the slot before rotating
                step_all(gp[:, i], gq[:, i])
        sub = args_cli.rotate_hold if (args_cli.rotate_hold > 0 and i >= ROTATE_START - 1) else SUB
        q0 = gq[:, i]; q1 = gq[:, i + 1].clone()
        flip = (q0 * q1).sum(-1) < 0
        q1[flip] = -q1[flip]
        for k in range(sub):
            t = (k + 1) / sub
            step_all((1 - t) * gp[:, i] + t * gp[:, i + 1], (1 - t) * q0 + t * q1)
        if i % 12 == 0:
            print(f"  [viz] goal {i + 1}/{T}", flush=True)
    for _ in range(SUB * 2):
        step_all(gp[:, -1], gq[:, -1])
    for w in writers.values():
        w.close()

    if phys:
        PI = 3.14159265
        deg = (base.screw_asm.data.joint_pos[:, 0] * 180.0 / PI).cpu()
        ad = deg.abs()
        # per-env layout features (to correlate failures)
        slot = base._nominal_slot
        lyaw = (torch.atan2(slot[:, 1], slot[:, 0]) * 180.0 / PI).cpu()
        sq = gq[:, 0]                                          # (N,4) wxyz start orientation
        w, x, y, z = sq[:, 0], sq[:, 1], sq[:, 2], sq[:, 3]
        tx, ty, tz = 1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)  # tool dir (start)
        t_az = (torch.atan2(ty, tx) * 180.0 / PI).cpu()
        tz = tz.cpu()
        dist = torch.norm(gp[:, 0, :2] - base.screw_head_world[:, :2], dim=1).cpu()
        print("[viz] PER-ENV screw rotation under the PERFECT trajectory (deg):", flush=True)
        print(f"  mean={ad.mean():.0f}  median={ad.median():.0f}  min={ad.min():.0f}  max={ad.max():.0f}", flush=True)
        for thr in [90, 150, 170]:
            print(f"  scenes with |rotation| >= {thr} deg: {int((ad >= thr).sum().item())}/{N}", flush=True)
        order = ad.argsort()
        fail = ad < 90
        print(f"  FAIL group (|rot|<90, n={int(fail.sum())}): lyaw mean={lyaw[fail].abs().mean():.0f} | t_az mean={t_az[fail].abs().mean():.0f} | start_dist mean={dist[fail].mean():.3f}", flush=True)
        ok = ad >= 150
        print(f"  OK   group (|rot|>=150, n={int(ok.sum())}): lyaw mean={lyaw[ok].abs().mean():.0f} | t_az mean={t_az[ok].abs().mean():.0f} | start_dist mean={dist[ok].mean():.3f}", flush=True)
        print("  PER-ENV (sorted by |rot|):  env | rot | layout_yaw | start_tool_az | start_tool_z | start_dist", flush=True)
        for idx in order.tolist():
            print(f"    env_{idx:03d}: rot={deg[idx]:6.0f}  lyaw={lyaw[idx]:5.0f}  t_az={t_az[idx]:6.0f}  t_z={tz[idx]:+.2f}  dist={dist[idx]:.3f}", flush=True)
    print(f"[viz] clips -> {PE_DIR}/env_*.mp4 ({'NONE (--no_clips)' if args_cli.no_clips else N})", flush=True)
    env.close()
    simulation_app.close()


def main():
    torch.manual_seed(args_cli.seed)
    cfg_cls, task_id = ENV_MAP[args_cli.env]
    cfg = cfg_cls()
    if args_cli.goal_generator:               # WRONG-way rim-contact replay vs the cfg-default correct one
        cfg.goal_generator_module = args_cli.goal_generator
    if args_cli.object_scale > 0:             # enlarge the screwdriver to keep the tip/slot ratio
        import importlib as _il
        import numpy as _np
        s = args_cli.object_scale
        cfg.object_cfg.spawn.scale = (s, s, s)
        # TIP = body-origin->tip = 0.134 * scale; patch BEFORE gym.make so the env's _tip_local AND
        # the generator's compute_goals_batch both use it (same cached module object).
        _il.import_module(cfg.goal_generator_module).TIP = _np.array([0.134 * s, 0.0, 0.0])
    if args_cli.contact_clearance != 999.0:   # override tip seat depth (shallow = light engagement)
        cfg.screw_contact_clearance = args_cli.contact_clearance
    if args_cli.rim_r > 0 or args_cli.rim_axial != 999.0:   # tune rim contact to the head profile
        import importlib as _il2
        _gg = _il2.import_module(cfg.goal_generator_module)
        if args_cli.rim_r > 0:
            _gg.RIM_R = args_cli.rim_r
        if args_cli.rim_axial != 999.0:
            _gg.RIM_AXIAL = args_cli.rim_axial
    if args_cli.wide_slot and args_cli.env.startswith("screwdriver043"):
        _A = "/home/cning/simtoolreal_isaaclab/assets/usd"
        cfg.screw_cfg.spawn.usd_path = f"{_A}/screw_new_wideslot_sdf/screw_new_wideslot_sdf.usd"
        cfg.screw_asm_cfg.spawn.usd_path = f"{_A}/screw_assembly043_wideslot/screw_assembly043_wideslot.usd"
        print("[viz] WIDE SLOT: 60%-wider cross slot", flush=True)
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.demo_mode = True                     # makes the env build per_env_goals at reset
    cfg.use_fixed_goal_trajectory = True     # REQUIRED: loads start_pose -> demo_start_pose so the
                                             # env actually generates per_env_goals (else they're 0)
    cfg.randomize_layout = args_cli.randomize_layout or args_cli.num_envs > 1  # multi-env -> diverse scenes
    cfg.pretrained_object_scale = (2.5, 0.75, 0.75)
    cfg.trajectory_path = f"{ORIG_REPO}/dextoolbench/trajectories/screwdriver/044_screwdriver/{args_cli.demo_task}.json"
    cfg.max_consecutive_successes = 0        # no success-based reset
    cfg.episode_length_s = 1.0e6             # never time out
    # kinematic screwdriver: hold whatever pose we write (no gravity/physics drift)
    cfg.object_cfg.spawn.rigid_props.kinematic_enabled = True
    # PHYSICAL screw: the (kinematic, perfectly-teleported) screwdriver drives a revolute-jointed
    # screw purely by contact -> visualizes how an IDEAL goal trajectory tightens the screw.
    if args_cli.physical_screw:
        cfg.physical_screw = True
        cfg.screw_joint_friction = args_cli.screw_friction
        cfg.screw_joint_damping = args_cli.screw_damping
        cfg.sim.physx.gpu_collision_stack_size = 2 ** 29
    # park the arm straight up so it doesn't sit on top of the teleported screwdriver
    for j in [f"iiwa14_joint_{i}" for i in range(1, 8)]:
        cfg.robot_cfg.init_state.joint_pos[j] = 0.0
    N = args_cli.num_envs
    # physical SDF screw needs a bigger contact stack (esp. the +80% screwdriver043big); dropping
    # contacts in the CORRECT run would falsely show no spin. multi-env -> 2^30, single -> 2^29.
    cfg.sim.physx.gpu_collision_stack_size = (2 ** 30 if N > 1 else 2 ** 29) if args_cli.physical_screw else 2 ** 28
    if N > 1:
        run_multi(cfg, task_id)
        return

    # ---- single-env: viewer cam + RecordVideo ----
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.viewer.resolution = (1280, 720)
    cfg.viewer.eye = tuple(float(v) for v in args_cli.cam_eye.split(","))
    cfg.viewer.lookat = tuple(float(v) for v in args_cli.cam_lookat.split(","))
    env = gym.make(task_id, cfg=cfg, render_mode="rgb_array")
    obs, _ = env.reset()
    base = env.unwrapped
    goals = base.per_env_goals[0].clone()    # (T,7) xyz + xyzw, env-local
    T = goals.shape[0]

    os.makedirs(VIDEO_DIR, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env, video_folder=VIDEO_DIR, step_trigger=lambda s: s == 0,
        video_length=(T + 4) * args_cli.hold,
        name_prefix=f"goalposes_{args_cli.env}_{args_cli.demo_task}{'_physical' if args_cli.physical_screw else ''}", disable_logger=True)
    video_env = env
    env.reset()

    eo = base.scene.env_origins[0]
    eidx = torch.tensor([0], device=base.device)
    zero_act = torch.zeros((1, base.cfg.action_space), device=base.device)
    z6 = torch.zeros((1, 6), device=base.device)
    gp = goals[:, 0:3] + eo               # (T,3) world positions
    gq = goals[:, [6, 3, 4, 5]]           # (T,4) wxyz
    SUB = args_cli.hold                   # interpolated frames between consecutive goals (smooth)

    phys = args_cli.physical_screw and getattr(base, "screw_asm", None) is not None
    _n = [0]

    def screw_deg():
        return base.screw_asm.data.joint_pos[0, 0].item() * 180.0 / 3.14159265 if phys else 0.0

    def step_pose(pos, quat):
        pose = torch.cat([pos, quat / quat.norm()]).unsqueeze(0)
        base.object.write_root_pose_to_sim(pose, eidx)
        base.object.write_root_velocity_to_sim(z6, eidx)
        env.step(zero_act)
        _n[0] += 1
        if phys and _n[0] % 40 == 0:
            print(f"  [viz] frame {_n[0]:4d}  screw_angle={screw_deg():7.1f} deg", flush=True)

    print(f"[viz] smoothly playing {T} goals, {SUB} interp frames/segment (~{(T + 3) * SUB} frames)"
          f"{' | PHYSICAL screw' if phys else ''}", flush=True)
    for _ in range(SUB):                  # settle on the first goal
        step_pose(gp[0], gq[0])
    for i in range(T - 1):                # continuous LERP(pos) + nlerp(quat) between goals
        q0, q1 = gq[i], gq[i + 1]
        if torch.dot(q0, q1) < 0:
            q1 = -q1                       # shortest-path quaternion blend
        for k in range(SUB):
            t = (k + 1) / SUB
            step_pose((1 - t) * gp[i] + t * gp[i + 1], (1 - t) * q0 + t * q1)
    for _ in range(SUB * 3):              # hold on the final (rotated) pose
        step_pose(gp[-1], gq[-1])
    if phys:
        print(f"[viz] FINAL screw rotation under the perfect trajectory = {screw_deg():.1f} deg", flush=True)

    try:
        video_env.render()
    except Exception:
        pass
    video_env.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
