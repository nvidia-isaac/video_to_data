"""Visualize the ACTUAL collision meshes of the 044 screwdriver and the flat screw, to check
whether the screwdriver-blade collider can enter the screw slot.

- Screw collider = triangle mesh ('none') -> collision == render mesh; we read its points/faces.
- Screwdriver collider = convexDecomposition, but the asset is instanceable and its CollisionAPI
  lives on a sublayer, so we (a) de-instance, (b) read the collision-mesh geometry under .../collisions,
  (c) RE-COOK the convex decomposition on a fresh collider prim (same approximation) to get the hulls.

Everything is placed at the tighten CONTACT pose (tip-down, blade along the slot, tip at the screw
head). Saves OBJs to /tmp/coll and RTX PNGs to videos/collision_*.png.
"""
import argparse
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import os  # noqa: E402
import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics, Gf, UsdUtils, PhysicsSchemaTools  # noqa: E402
from omni.physx import get_physx_cooking_interface  # noqa: E402

ASSETS = "/home/cning/simtoolreal_isaaclab/assets/usd"
SD_USD = f"{ASSETS}/044_screwdriver/044_screwdriver.usd"
SCREW_USD = f"{ASSETS}/flat_screw/flat_screw.usd"
SCREW_SCALE = 0.013
OUT_DIR = "/home/cning/simtoolreal_isaaclab/videos"
OBJ_DIR = "/tmp/coll"
os.makedirs(OBJ_DIR, exist_ok=True)
LOG = []


def log(m):
    LOG.append(str(m))
    with open(f"{OBJ_DIR}/log.txt", "w") as f:
        f.write("\n".join(LOG) + "\n")


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def align_R(la, wa, lb, wb):
    la, wa = unit(la), unit(wa)
    lb = unit(np.asarray(lb, float) - np.dot(lb, la) * la)
    wb = unit(np.asarray(wb, float) - np.dot(wb, wa) * wa)
    L = np.column_stack([la, lb, np.cross(la, lb)])
    W = np.column_stack([wa, wb, np.cross(wa, wb)])
    return W @ L.T


def R_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return np.array([w, x, y, z])


def world_mat(prim):
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T


def xform_pts(M, pts):
    p = np.asarray(pts, float)
    return (np.c_[p, np.ones(len(p))] @ M.T)[:, :3]


def tris_from_counts(counts, idx):
    faces, o = [], 0
    for c in counts:
        for k in range(1, c - 1):
            faces.append([idx[o], idx[o + k], idx[o + k + 1]])
        o += c
    return np.array(faces)


def save_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in faces:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


# ------------------------------------------------------------------ poses (contact)
SCREW_ROOT = np.array([0.0, 0.0, 0.10])
HEAD = SCREW_ROOT + np.array([0.0072, -0.0238, 0.0436])
TIP_LOCAL = np.array([0.134, 0.0, 0.0])
R_sd = align_R([1, 0, 0], [0, 0, -1], [0, 0, 1], [1, 0, 0])  # tool->down, blade->slot(world x)
SD_ROOT = HEAD - R_sd @ TIP_LOCAL
sd_quat = R_to_quat_wxyz(R_sd)
screw_quat = np.array([0.3852, 0.9223, -0.0306, 0.0])

# ------------------------------------------------------------------ scene
sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 120.0, device="cpu"))
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))
sim_utils.UsdFileCfg(usd_path=SCREW_USD, scale=(SCREW_SCALE,) * 3).func(
    "/World/screw", sim_utils.UsdFileCfg(usd_path=SCREW_USD, scale=(SCREW_SCALE,) * 3),
    translation=tuple(SCREW_ROOT), orientation=tuple(screw_quat))
sim_utils.UsdFileCfg(usd_path=SD_USD).func(
    "/World/sd", sim_utils.UsdFileCfg(usd_path=SD_USD),
    translation=tuple(SD_ROOT), orientation=tuple(sd_quat))

stage = sim.stage
# de-instance the screwdriver so its collision mesh becomes a real, readable prim
for prim in stage.Traverse():
    if str(prim.GetPath()).startswith("/World/sd") and prim.IsInstanceable():
        prim.SetInstanceable(False)

cam = Camera(CameraCfg(prim_path="/World/cam", height=900, width=1400, data_types=["rgb"],
                       spawn=sim_utils.PinholeCameraCfg(focal_length=30.0, clipping_range=(0.005, 50.0))))
sim.reset()
for _ in range(3):
    sim.step()

# ---- screw collision triangle mesh (world) ----
screw_mesh = next(p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/screw"))
                  if p.GetTypeName() == "Mesh" and p.HasAPI(UsdPhysics.CollisionAPI))
Msc = world_mat(screw_mesh)
sc_pts = xform_pts(Msc, np.array(screw_mesh.GetAttribute("points").Get()))
sc_faces = tris_from_counts(np.array(screw_mesh.GetAttribute("faceVertexCounts").Get()),
                            np.array(screw_mesh.GetAttribute("faceVertexIndices").Get()))
save_obj(f"{OBJ_DIR}/screw_collision.obj", sc_pts, sc_faces)
UsdGeom.Mesh(screw_mesh).CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.6, 0.68)])
log(f"screw collision: {len(sc_pts)} v {len(sc_faces)} tris")

# ---- screwdriver collision mesh (under /collisions) ----
sd_coll = next(p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/sd"))
               if p.GetTypeName() == "Mesh" and "/collisions/" in str(p.GetPath()))
Msd = world_mat(sd_coll)
sd_pts_local = np.array(sd_coll.GetAttribute("points").Get())
sd_counts = np.array(sd_coll.GetAttribute("faceVertexCounts").Get())
sd_idx = np.array(sd_coll.GetAttribute("faceVertexIndices").Get())
sd_src_world = xform_pts(Msd, sd_pts_local)
save_obj(f"{OBJ_DIR}/screwdriver_collision_source.obj", sd_src_world,
         tris_from_counts(sd_counts, sd_idx))
log(f"sd collision source: {len(sd_pts_local)} v (local), world bbox "
    f"{np.round(sd_src_world.min(0), 4)}..{np.round(sd_src_world.max(0), 4)}")

# ---- RE-COOK convex decomposition on a fresh collider prim from that geometry ----
cook = UsdGeom.Mesh.Define(stage, "/World/cook")
cook.CreatePointsAttr([Gf.Vec3f(*p) for p in sd_pts_local.astype(float)])
cook.CreateFaceVertexCountsAttr([int(c) for c in sd_counts])
cook.CreateFaceVertexIndicesAttr([int(i) for i in sd_idx])
cprim = cook.GetPrim()
UsdPhysics.CollisionAPI.Apply(cprim)
UsdPhysics.MeshCollisionAPI.Apply(cprim).CreateApproximationAttr(UsdPhysics.Tokens.convexDecomposition)
UsdGeom.Imageable(cprim).MakeInvisible()  # collider only, don't render the raw cook mesh
for _ in range(3):
    sim.step()
stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
prim_id = PhysicsSchemaTools.sdfPathToInt("/World/cook")
res = {"done": False, "data": None, "r": None}


def on_ready(result, convexes):
    res["r"] = int(result); res["data"] = convexes; res["done"] = True


get_physx_cooking_interface().request_convex_collision_representation(
    stage_id=stage_id, collision_prim_id=prim_id, run_asynchronously=True, on_result=on_ready)
for _ in range(600):
    if res["done"]:
        break
    app.update()
log(f"convex decomposition: result={res['r']} n_hulls={len(res['data']) if res['data'] else 0}")

colors = [(0.95, 0.35, 0.1), (0.1, 0.55, 0.95), (0.2, 0.8, 0.3), (0.95, 0.8, 0.1),
          (0.85, 0.2, 0.85), (0.1, 0.85, 0.85), (0.95, 0.55, 0.7), (0.55, 0.4, 0.95),
          (0.5, 0.5, 0.5), (0.9, 0.5, 0.2)]
all_v, all_f, voff = [], [], 0
UsdGeom.Xform.Define(stage, "/World/sd_hulls")
if res["data"]:
    for hi, cx in enumerate(res["data"]):
        vloc = np.array([[v.x, v.y, v.z] for v in cx.vertices])
        vworld = xform_pts(Msd, vloc)  # hull local == cook local == collision-mesh local
        idx = list(cx.indices)
        fc, fi, tris = [], [], []
        for poly in cx.polygons:
            b, n = poly.index_base, poly.num_vertices
            fc.append(n); fi.extend(idx[b:b + n])
            for k in range(1, n - 1):
                tris.append([idx[b], idx[b + k], idx[b + k + 1]])
        m = UsdGeom.Mesh.Define(stage, f"/World/sd_hulls/h{hi}")
        m.CreatePointsAttr([Gf.Vec3f(*p) for p in vworld.astype(float)])
        m.CreateFaceVertexCountsAttr(fc)
        m.CreateFaceVertexIndicesAttr(fi)
        m.CreateDisplayColorAttr([Gf.Vec3f(*colors[hi % len(colors)])])
        for t in tris:
            all_f.append([t[0] + voff, t[1] + voff, t[2] + voff])
        all_v.extend(vworld.tolist())
        voff += len(vworld)
    save_obj(f"{OBJ_DIR}/screwdriver_collision_hulls.obj", np.array(all_v), np.array(all_f))
    log(f"hulls -> {len(res['data'])} hulls, {voff} verts (world)")

# hide the screwdriver VISUAL so only collision geometry is rendered
UsdGeom.Imageable(stage.GetPrimAtPath("/World/sd")).MakeInvisible()


# ------------------------------------------------------------------ render
def capture(name, eye, target):
    cam.set_world_poses_from_view(eyes=torch.tensor([eye], dtype=torch.float32),
                                  targets=torch.tensor([target], dtype=torch.float32))
    for _ in range(14):
        sim.step(); cam.update(1 / 120.0)
    rgb = cam.data.output["rgb"][0].cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    imageio.imwrite(f"{OUT_DIR}/collision_{name}.png", rgb.astype(np.uint8))
    log(f"saved collision_{name}.png")


H = HEAD.tolist()
T = [H[0], H[1], H[2] + 0.01]  # aim just above the head (tip-meets-screw region)
capture("screw_top", [H[0], H[1] + 0.001, H[2] + 0.055], [H[0], H[1], H[2]])     # screw slot from above (close)
capture("contact_perp", [H[0], H[1] - 0.085, H[2] + 0.03], T)                    # across the slot
capture("contact_along", [H[0] - 0.085, H[1], H[2] + 0.025], T)                  # down the slot
capture("contact_iso", [H[0] - 0.16, H[1] - 0.16, H[2] + 0.11], [H[0], H[1], H[2] + 0.05])  # 3/4, pulled back
log("DONE")
app.close()
