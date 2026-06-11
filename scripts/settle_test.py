"""Settle test: spawn the screwdriver resting on the table (no robot control) and log its
linear/angular speed over time. Isolates resting-contact jitter (SDF vs convex, prop overrides)."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", default="sdf", choices=["sdf", "convex"])
parser.add_argument("--max_depen", type=float, default=-1.0, help=">=0 overrides max_depenetration_velocity")
parser.add_argument("--vel_iters", type=int, default=-1, help=">=0 overrides solver_velocity_iteration_count")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--contact_offset", type=float, default=-1.0, help=">=0 sets contact_offset")
parser.add_argument("--rest_offset", type=float, default=-99.0, help=">-1 sets rest_offset")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import sys  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")
import simtoolreal_lab.tasks  # noqa: E402, F401
from simtoolreal_lab.tasks.screwdriver.screwdriver_env_cfg import ScrewdriverEnvCfg  # noqa: E402

ASSETS = "/home/cning/simtoolreal_isaaclab/assets/usd"
USD = {"sdf": f"{ASSETS}/044_screwdriver_sdf/044_screwdriver_sdf.usd",
       "convex": f"{ASSETS}/044_screwdriver/044_screwdriver.usd"}[args.usd]

cfg = ScrewdriverEnvCfg()
cfg.scene.num_envs = 6
cfg.randomize_layout = False
cfg.demo_mode = False
cfg.use_fixed_goal_trajectory = False
cfg.domain_randomization = False
cfg.sim.physx.gpu_collision_stack_size = 2 ** 28
cfg.object_cfg.spawn.usd_path = USD
# no reset noise so it settles cleanly in place
cfg.reset_position_noise_x = 0.0
cfg.reset_position_noise_y = 0.0
cfg.reset_position_noise_z = 0.0
rp = cfg.object_cfg.spawn.rigid_props
if args.max_depen >= 0:
    rp.max_depenetration_velocity = args.max_depen
if args.vel_iters >= 0:
    rp.solver_velocity_iteration_count = args.vel_iters
co = ro = None
if args.contact_offset >= 0 or args.rest_offset > -1.0:
    import isaaclab.sim as sim_utils
    cp = cfg.object_cfg.spawn.collision_props or sim_utils.CollisionPropertiesCfg()
    if args.contact_offset >= 0:
        cp.contact_offset = co = args.contact_offset
    if args.rest_offset > -1.0:
        cp.rest_offset = ro = args.rest_offset
    cfg.object_cfg.spawn.collision_props = cp

env = gym.make("Isaac-SimToolReal-Screwdriver-Direct-v0", cfg=cfg)
base = env.unwrapped
zero = torch.zeros((base.num_envs, base.cfg.action_space), device=base.device)
env.reset()
print(f"SETTLE usd={args.usd} max_depen={rp.max_depenetration_velocity} vel_iters={rp.solver_velocity_iteration_count} "
      f"contact_offset={co} rest_offset={ro}")
for t in range(args.steps):
    env.step(zero)
    if t % 30 == 0 or t >= args.steps - 2:
        v = base.object.data.root_lin_vel_w[:, :3].norm(dim=-1)   # m/s
        w = base.object.data.root_ang_vel_w[:, :3].norm(dim=-1)   # rad/s
        z = base.object.data.root_pos_w[:, 2]
        print(f"t={t:3d} z_mean={z.mean().item():.4f} |v|_mean={v.mean().item()*1000:6.2f} mm/s "
              f"|v|_max={v.max().item()*1000:6.2f} mm/s |w|_mean={w.mean().item():.3f} rad/s |w|_max={w.max().item():.3f}", flush=True)
env.close()
app.close()
