"""Build screw_assembly043_180.usd: the 043 cross-slot physical screw assembly at +80% size.

Identical structure to the base screw_assembly043 (thread_test fixed base + revolute-jointed
cross-slot screw), but every geometric quantity scaled by f=1.8 so the screw + thread_test are
80% bigger. Used ONLY by the screwdriver043big experiment env (physics-validity test: does the
screw spin only when the tip is seated in the cross slot, or also from outer-rim contact?).

The base anchor (assembly origin -> world (0.0475,0,0.53)) is unchanged; the internal screw
translate + joint localPos + both geom scales all scale by f, so the bottom stays on the table and
the whole screw/fixture grows. The kinematic cfg constants in screwdriver043big_env_cfg.py are
derived from the SAME f so screw_head_world (the goal target) matches the physical slot.
"""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf  # noqa: E402
import os  # noqa: E402

F = 1.8                                   # +80%
A = "/home/cning/simtoolreal_isaaclab/assets/usd"
TT = f"{A}/thread_test/thread_test.usd"; SC = f"{A}/screw_new_sdf/screw_new_sdf.usd"
OUT = f"{A}/screw_assembly043_180/screw_assembly043_180.usd"; os.makedirs(os.path.dirname(OUT), exist_ok=True)

BASE_SCALE = 0.004875 * F                 # thread_test base
SCREW_SCALE = 0.012948 * F                # cross-slot screw
TX, TY, TZ = 0.0288 * F, 0.0, 0.0478 * F  # screw translate + joint localPos0 (scaled offsets)

stage = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, "/screw_assembly043_180"); stage.SetDefaultPrim(root.GetPrim())


def link(path, mass):
    x = UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim()); UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
    return x


# fixed base = +80% thread_test (transform on geom, reference on child)
link("/screw_assembly043_180/base", 1.0)
bg = UsdGeom.Xform.Define(stage, "/screw_assembly043_180/base/geom"); bg.AddScaleOp().Set(Gf.Vec3f(BASE_SCALE, BASE_SCALE, BASE_SCALE))
UsdGeom.Xform.Define(stage, "/screw_assembly043_180/base/geom/ref").GetPrim().GetReferences().AddReference(TT, "/threadTest/geometry")

# revolute-jointed cross-slot screw link (mass kept at 0.05 to isolate the friction/damping variable)
sc = link("/screw_assembly043_180/screw", 0.05); sc.AddTranslateOp().Set(Gf.Vec3d(TX, TY, TZ))
sg = UsdGeom.Xform.Define(stage, "/screw_assembly043_180/screw/geom")
sg.AddOrientOp().Set(Gf.Quatf(0.70710678, 0.70710678, 0.0, 0.0)); sg.AddScaleOp().Set(Gf.Vec3f(SCREW_SCALE, SCREW_SCALE, SCREW_SCALE))
UsdGeom.Xform.Define(stage, "/screw_assembly043_180/screw/geom/ref").GetPrim().GetReferences().AddReference(SC, "/screw/geometry")

# revolute joint (screw spins about world +Z)
UsdGeom.Scope.Define(stage, "/screw_assembly043_180/joints")
rev = UsdPhysics.RevoluteJoint.Define(stage, "/screw_assembly043_180/joints/screw_spin")
rev.CreateBody0Rel().SetTargets(["/screw_assembly043_180/base"]); rev.CreateBody1Rel().SetTargets(["/screw_assembly043_180/screw"])
rev.CreateAxisAttr("Z")
rev.CreateLocalPos0Attr(Gf.Vec3f(TX, TY, TZ)); rev.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
rev.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0)); rev.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))

# fixed root joint + articulation root
rj = UsdPhysics.FixedJoint.Define(stage, "/screw_assembly043_180/root_joint")
rj.CreateBody1Rel().SetTargets(["/screw_assembly043_180/base"])
UsdPhysics.ArticulationRootAPI.Apply(rj.GetPrim()); PhysxSchema.PhysxArticulationAPI.Apply(rj.GetPrim())

stage.Save()
os.makedirs("/tmp/coll", exist_ok=True)
open("/tmp/coll/asm043_big_built.txt", "w").write(
    f"built {OUT}\n base_scale={BASE_SCALE} screw_scale={SCREW_SCALE} translate=({TX},{TY},{TZ})\n")
print(f"[build] wrote {OUT}  base_scale={BASE_SCALE} screw_scale={SCREW_SCALE} translate=({TX},{TY},{TZ})")
app.close()
