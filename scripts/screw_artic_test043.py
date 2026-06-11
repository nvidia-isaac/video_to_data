"""Test the screw_assembly043 ARTICULATION (043 phillips + cross-slot screw, 50% thread_test).
Script the SDF cross tip to spin in the cross slot and read the screw_spin joint -> does it follow?"""
import argparse
import numpy as np
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--omega", type=float, default=2.0)
parser.add_argument("--tip_in", type=float, default=0.0006)
parser.add_argument("--steps", type=int, default=360)
parser.add_argument("--damping", type=float, default=0.0)
parser.add_argument("--friction", type=float, default=0.0)
parser.add_argument("--armature", type=float, default=0.0)
parser.add_argument("--hold_frac", type=float, default=0.6, help="spin the blade for this fraction of steps, then HOLD it still (tests coasting)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg, Articulation, ArticulationCfg  # noqa: E402

ASSETS = "/home/cning/simtoolreal_isaaclab/assets/usd"
ART = f"{ASSETS}/screw_assembly043/screw_assembly043.usd"
SD_SDF = f"{ASSETS}/043_screwdriver_aligned_sdf/043_screwdriver_aligned_sdf.usd"


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


TT_WORLD = np.array([0.0475, 0.0, 0.53])
HEAD = np.array([0.0623, 0.0, 0.559])
TIP_LOCAL = np.array([0.134, 0.0, 0.0])
R_CONTACT = align_R([1, 0, 0], [0, 0, -1], [0, 0, 1], [1, 0, 0])
TIP_TARGET = HEAD - np.array([0.0, 0.0, args.tip_in])

sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 120.0, device="cpu",
                                                physx=sim_utils.PhysxCfg(gpu_collision_stack_size=2 ** 28)))
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

art_cfg = ArticulationCfg(
    prim_path="/World/screw_asm",
    spawn=sim_utils.UsdFileCfg(usd_path=ART),
    init_state=ArticulationCfg.InitialStateCfg(pos=tuple(TT_WORLD)),
    actuators={})
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
art_cfg.actuators = {"spin": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0,
                                                 damping=args.damping, friction=args.friction,
                                                 armature=args.armature)}
art = Articulation(art_cfg)

sd_cfg = RigidObjectCfg(
    prim_path="/World/sd",
    spawn=sim_utils.UsdFileCfg(usd_path=SD_SDF,
                               rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(TIP_TARGET - R_CONTACT @ TIP_LOCAL),
                                              rot=tuple(R_to_quat(R_CONTACT))))
sd = RigidObject(sd_cfg)

sim.reset()
eidx = torch.tensor([0])
print(f"ARTIC_TEST omega={args.omega} damping={args.damping} friction={args.friction} armature={args.armature} | dof={art.joint_names}")
ang = 0.0
dt = 1 / 60.0
hold_step = int(args.steps * args.hold_frac)
for t in range(args.steps):
    lift = np.array([0.0, 0.0, 0.0])
    if t < hold_step:           # spin the blade in the slot ...
        ang += args.omega * dt
    else:                       # ... then LIFT it out of the slot (screw is now free -> does it coast?)
        lift = np.array([0.0, 0.0, 0.10])
    Rc = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]) @ R_CONTACT
    pos = TIP_TARGET + lift - Rc @ TIP_LOCAL
    sd.write_root_pose_to_sim(torch.tensor([[*pos.tolist(), *R_to_quat(Rc).tolist()]], dtype=torch.float32), eidx)
    sim.step()
    art.update(dt)
    if t % 30 == 0 or t == args.steps - 1:
        jp = art.data.joint_pos[0]
        jv = art.data.joint_vel[0]
        print(f"t={t:3d} driver_ang={ang*180/np.pi:6.1f}deg | screw_joint_pos={jp.tolist()} "
              f"({(jp[0].item()*180/np.pi):.1f}deg) vel={jv[0].item():.3f} rad/s", flush=True)
app.close()
