"""Physics prototype: can blade contact turn a REVOLUTE-jointed screw?

Dynamic SDF screw + a revolute joint about its shaft axis (so it can ONLY spin, no sink/eject/
off-center). The SDF screwdriver is KINEMATICALLY scripted to sit tip-in-slot and rotate about the
screw axis (forcing firm engagement). We log whether the screw follows via contact friction torque.
If it spins here, full physics is viable (and the only gap is the policy seating the blade); if not,
contact-torque from the thin blade is insufficient.
"""
import argparse
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--omega", type=float, default=2.0, help="screwdriver spin rate (rad/s)")
parser.add_argument("--tip_in", type=float, default=0.004, help="how far the tip is pushed below the head into the slot (m)")
parser.add_argument("--steps", type=int, default=360)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from pxr import UsdPhysics, Gf, UsdGeom, Usd  # noqa: E402

ASSETS = "/home/cning/simtoolreal_isaaclab/assets/usd"
SCREW_SDF = f"{ASSETS}/flat_screw_sdf/flat_screw.usd"
SD_SDF = f"{ASSETS}/044_screwdriver_sdf/044_screwdriver_sdf.usd"
SCALE = 0.013


def unit(v):
    v = np.asarray(v, float); return v / (np.linalg.norm(v) + 1e-12)


def align_R(la, wa, lb, wb):
    la, wa = unit(la), unit(wa)
    lb = unit(np.asarray(lb, float) - np.dot(lb, la) * la)
    wb = unit(np.asarray(wb, float) - np.dot(wb, wa) * wa)
    L = np.column_stack([la, lb, np.cross(la, lb)]); W = np.column_stack([wa, wb, np.cross(wa, wb)])
    return W @ L.T


def R_to_quat(R):  # -> wxyz
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


def quat_to_R(q):  # wxyz -> 3x3
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---- geometry (matches the env's nominal screw + aligned-044 local axes) ----
SCREW_ROOT = np.array([0.0, 0.0, 0.10])
SCREW_QUAT = np.array([0.3852, 0.9223, -0.0306, 0.0])      # wxyz, shaft down/head up
HEAD_OFF = np.array([0.0072, -0.0238, 0.0436])             # world @ nominal
HEAD = SCREW_ROOT + HEAD_OFF
TIP_LOCAL = np.array([0.134, 0.0, 0.0])
R_CONTACT = align_R([1, 0, 0], [0, 0, -1], [0, 0, 1], [1, 0, 0])  # tool->down, blade->slot(world x)
TIP_TARGET = HEAD - np.array([0.0, 0.0, args.tip_in])      # push tip into the slot

sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 120.0, device="cpu",
                                                physx=sim_utils.PhysxCfg(gpu_collision_stack_size=2 ** 28)))
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

# dynamic SDF screw
screw_cfg = RigidObjectCfg(
    prim_path="/World/screw",
    spawn=sim_utils.UsdFileCfg(usd_path=SCREW_SDF, scale=(SCALE,) * 3,
                               rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False,
                                                                            disable_gravity=False),
                               mass_props=sim_utils.MassPropertiesCfg(mass=0.02)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(SCREW_ROOT), rot=tuple(SCREW_QUAT)))
screw = RigidObject(screw_cfg)

# kinematic SDF screwdriver (scripted)
sd_cfg = RigidObjectCfg(
    prim_path="/World/sd",
    spawn=sim_utils.UsdFileCfg(usd_path=SD_SDF,
                               rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(TIP_TARGET - R_CONTACT @ TIP_LOCAL),
                                              rot=tuple(R_to_quat(R_CONTACT))))
sd = RigidObject(sd_cfg)

# ---- revolute joint: world -> screw, axis = world +Z through the head ----
stage = sim.stage
j = UsdPhysics.RevoluteJoint.Define(stage, "/World/screw_joint")
j.CreateBody1Rel().SetTargets(["/World/screw"])
j.CreateAxisAttr("Z")
Rs = quat_to_R(SCREW_QUAT)
local1 = (Rs.T @ HEAD_OFF) / SCALE          # head in screw mesh-local (account for prim scale)
q1 = R_to_quat(Rs.T)                          # screw^-1 so joint Z == world Z
j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in HEAD]))
j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
j.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in local1]))
j.CreateLocalRot1Attr(Gf.Quatf(float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3])))

sim.reset()
eidx = torch.tensor([0])
print(f"PHYS_TEST omega={args.omega} tip_in={args.tip_in} | screw=SDF dynamic + revolute(Z@head)")
ang = 0.0
dt = 1 / 60.0
prev_screw_yaw = None
for t in range(args.steps):
    ang += args.omega * dt
    Rc = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]) @ R_CONTACT
    pos = TIP_TARGET - Rc @ TIP_LOCAL
    pose = torch.tensor([[*pos.tolist(), *R_to_quat(Rc).tolist()]], dtype=torch.float32)
    sd.write_root_pose_to_sim(pose, eidx)
    sd.write_root_velocity_to_sim(torch.zeros((1, 6)), eidx)
    sim.step()
    screw.update(dt)
    if t % 30 == 0 or t == args.steps - 1:
        w = screw.data.root_ang_vel_w[0]
        v = screw.data.root_lin_vel_w[0]
        sq = screw.data.root_quat_w[0].numpy()        # wxyz
        # screw spin about z relative to nominal (yaw of the blade/x-axis through the body)
        print(f"t={t:3d} driver_ang={ang*180/np.pi:6.1f}deg | screw |wz|={w[2].item():.3f} rad/s "
              f"|w|={w.norm().item():.3f} |v|={v.norm().item()*1000:.2f}mm/s "
              f"screw_z={screw.data.root_pos_w[0,2].item():.4f}", flush=True)
app.close()
