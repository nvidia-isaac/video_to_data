"""Does the 043 PHILLIPS cross tip REALLY tighten the cross-slot screw (via contact, not scripting)?

Spawns the cross-slot screw as a DYNAMIC SDF body pinned by a world-anchored REVOLUTE joint (axis =
its vertical shaft axis), so it can ONLY spin -- like a screw in threads. Then drives the kinematic
SDF screwdriver's cross tip down into the slot and rotates it. The screw is NOT coupled to the driver
in any way; if it turns, it's purely from the cross tip pushing the cross-slot walls (real contact).

Prints driver angle vs measured screw angle over time. If the screw angle follows the driver, the tip
physically tightens it. After --hold_frac of the run the driver lifts out -> tests coasting/holding.

Run: cd IsaacLab && (venv) OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
       ~/simtoolreal_isaaclab/scripts/screw_tighten_test043.py --headless
"""
import argparse
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--omega", type=float, default=2.0, help="driver spin rate (rad/s)")
parser.add_argument("--tip_in", type=float, default=0.003, help="how far the tip target sits below the slot top (m)")
parser.add_argument("--steps", type=int, default=480)
parser.add_argument("--joint_friction", type=float, default=0.0, help="revolute-joint friction (thread resistance)")
parser.add_argument("--hold_frac", type=float, default=0.7, help="spin for this fraction of steps, then lift out")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
from pxr import UsdPhysics, PhysxSchema, Gf  # noqa: E402

ASSETS = "/home/cning/simtoolreal_isaaclab/assets/usd"
SCREW_SDF = f"{ASSETS}/screw_new_sdf/screw_new_sdf.usd"
SD_SDF = f"{ASSETS}/043_screwdriver_aligned_sdf/043_screwdriver_aligned_sdf.usd"

# 043 env geometry
SCREW_POS = np.array([0.0623, 0.0, 0.5545])       # screw root (= shaft axis), stood head-up
SCREW_ROT_WXYZ = np.array([0.70710678, 0.70710678, 0.0, 0.0])  # +90deg about x
HEAD = np.array([0.0623, 0.0, 0.559])             # cross slot (world)
SCREW_SCALE = 0.00664
TIP_LOCAL = np.array([0.134, 0.0, 0.0])           # aligned screwdriver: origin -> tip


def unit(v):
    v = np.asarray(v, float); return v / (np.linalg.norm(v) + 1e-12)


def align_R(la, wa, lb, wb):
    la, wa = unit(la), unit(wa)
    lb = unit(np.asarray(lb, float) - np.dot(lb, la) * la)
    wb = unit(np.asarray(wb, float) - np.dot(wb, wa) * wa)
    L = np.column_stack([la, lb, np.cross(la, lb)]); W = np.column_stack([wa, wb, np.cross(wa, wb)])
    return W @ L.T


def R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2; w = .25 * s; x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / s; x = .25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = .25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2]) * 2; w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = .25 * s
    return np.array([w, x, y, z])


R_CONTACT = align_R([1, 0, 0], [0, 0, -1], [0, 0, 1], [1, 0, 0])  # tip-down, cross arm -> world +x (a slot arm)
TIP_TARGET = HEAD - np.array([0.0, 0.0, args.tip_in])

sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 120.0, device="cpu",
                                                physx=sim_utils.PhysxCfg(gpu_collision_stack_size=2 ** 28)))
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

# DYNAMIC cross-slot screw (SDF). NOT kinematic -> it can be turned by contact.
screw_cfg = RigidObjectCfg(
    prim_path="/World/screw",
    spawn=sim_utils.UsdFileCfg(usd_path=SCREW_SDF, scale=(SCREW_SCALE,) * 3,
                               rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                   max_angular_velocity=1000.0, solver_position_iteration_count=16)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(SCREW_POS), rot=tuple(SCREW_ROT_WXYZ)))
screw = RigidObject(screw_cfg)

# kinematic SDF screwdriver
sd_cfg = RigidObjectCfg(
    prim_path="/World/sd",
    spawn=sim_utils.UsdFileCfg(usd_path=SD_SDF,
                               rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(TIP_TARGET - R_CONTACT @ TIP_LOCAL),
                                              rot=tuple(R_to_quat(R_CONTACT))))
sd = RigidObject(sd_cfg)

# world-anchored REVOLUTE joint pinning the screw to spin about world +Z at its shaft axis.
# frame0 (world): pos = screw world pos, rot = identity, axis Z. frame1 (screw): pos 0,
# rot = inverse(screw init rot) so the two frames coincide at bind.
stage = stage_utils.get_current_stage()
joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/screw_spin")
joint.CreateBody1Rel().SetTargets(["/World/screw"])          # body0 omitted -> world
joint.CreateAxisAttr("Z")
joint.CreateLocalPos0Attr(Gf.Vec3f(*SCREW_POS.tolist()))
joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
joint.CreateLocalRot1Attr(Gf.Quatf(0.70710678, -0.70710678, 0.0, 0.0))  # inverse(+90deg x)
if args.joint_friction > 0:
    pj = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
    pj.CreateJointFrictionAttr(args.joint_friction)

sim.reset()
eidx = torch.tensor([0])
dt = 1 / 60.0
ang = 0.0
hold_step = int(args.steps * args.hold_frac)


def screw_yaw():
    q = screw.data.root_quat_w[0].cpu().numpy()              # wxyz
    w, x, y, z = q
    Rm = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    xax = Rm @ np.array([1.0, 0.0, 0.0])                     # screw mesh-x -> world (lies in xy plane)
    return np.degrees(np.arctan2(xax[1], xax[0]))


yaw0 = None
print(f"TIGHTEN_TEST043 omega={args.omega} tip_in={args.tip_in} joint_friction={args.joint_friction}", flush=True)
for t in range(args.steps):
    lift = np.array([0.0, 0.0, 0.10]) if t >= hold_step else np.zeros(3)
    if t < hold_step:
        ang += args.omega * dt
    Rc = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]) @ R_CONTACT
    pos = TIP_TARGET + lift - Rc @ TIP_LOCAL
    sd.write_root_pose_to_sim(torch.tensor([[*pos.tolist(), *R_to_quat(Rc).tolist()]], dtype=torch.float32), eidx)
    sim.step()
    screw.update(dt)
    sy = screw_yaw()
    if yaw0 is None:
        yaw0 = sy
    if t % 40 == 0 or t == args.steps - 1 or t == hold_step:
        net = ((sy - yaw0 + 180) % 360) - 180
        tag = " <-LIFTED" if t >= hold_step else ""
        print(f"t={t:3d} driver={ang*180/np.pi:7.1f}deg | screw_yaw={net:7.1f}deg{tag}", flush=True)
app.close()
