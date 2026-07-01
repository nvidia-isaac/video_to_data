"""Build screw_assembly043_prismatic.usd: the 043 cross-slot screw on a PRISMATIC joint (nail-in).

Same structure as screw_assembly043 (thread_test FIXED base + jointed screw link) but the screw
is on a PRISMATIC joint along the screw axis (world +Z) instead of a revolute joint -- so the
hammer can drive ('nail') it down/up along the axis rather than spinning it. FREE both directions
(no joint limits), matching the requested nail behavior. Used by the `hammer` task.
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf  # noqa: E402

A = "/home/cning/simtoolreal_isaaclab/assets/usd"
TT = f"{A}/thread_test/thread_test.usd"
SC = f"{A}/screw_new_sdf/screw_new_sdf.usd"
OUT = f"{A}/screw_assembly043_prismatic/screw_assembly043_prismatic.usd"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# same geometry as the base (F=1) screw_assembly043
BASE_SCALE = 0.004875                 # thread_test base
SCREW_SCALE = 0.012948                # cross-slot screw
TX, TY, TZ = 0.0288, 0.0, 0.0478      # screw translate + joint localPos0

ROOT = "/screw_assembly043_prismatic"
stage = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, ROOT); stage.SetDefaultPrim(root.GetPrim())


def link(path, mass):
    x = UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim()); UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
    return x


# fixed base = thread_test (transform on geom, reference on child)
link(f"{ROOT}/base", 1.0)
bg = UsdGeom.Xform.Define(stage, f"{ROOT}/base/geom"); bg.AddScaleOp().Set(Gf.Vec3f(BASE_SCALE, BASE_SCALE, BASE_SCALE))
UsdGeom.Xform.Define(stage, f"{ROOT}/base/geom/ref").GetPrim().GetReferences().AddReference(TT, "/threadTest/geometry")

# prismatic-jointed cross-slot screw link
sc = link(f"{ROOT}/screw", 0.05); sc.AddTranslateOp().Set(Gf.Vec3d(TX, TY, TZ))
sg = UsdGeom.Xform.Define(stage, f"{ROOT}/screw/geom")
sg.AddOrientOp().Set(Gf.Quatf(0.70710678, 0.70710678, 0.0, 0.0)); sg.AddScaleOp().Set(Gf.Vec3f(SCREW_SCALE, SCREW_SCALE, SCREW_SCALE))
UsdGeom.Xform.Define(stage, f"{ROOT}/screw/geom/ref").GetPrim().GetReferences().AddReference(SC, "/screw/geometry")

# PRISMATIC joint: screw slides along world +Z (nail in/out). LIMITED travel so the hammer can't
# drive the nail straight through the bar (it would vanish). Measured geometry: at joint=0 the head
# sits +0.0097 m above the bar top, so the head is FLUSH with the bar top at joint=-0.0097. lower is
# set just ABOVE flush so the driven-in head stays VISIBLE (sitting ~on the surface), not swallowed.
# upper=+0.015 m (raised). The nail starts at +nail_start_height and is driven down to lower.
LOWER, UPPER = -0.008, 0.015
UsdGeom.Scope.Define(stage, f"{ROOT}/joints")
pri = UsdPhysics.PrismaticJoint.Define(stage, f"{ROOT}/joints/nail_slide")
pri.CreateBody0Rel().SetTargets([f"{ROOT}/base"]); pri.CreateBody1Rel().SetTargets([f"{ROOT}/screw"])
pri.CreateAxisAttr("Z")
pri.CreateLocalPos0Attr(Gf.Vec3f(TX, TY, TZ)); pri.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
pri.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0)); pri.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
pri.CreateLowerLimitAttr(LOWER); pri.CreateUpperLimitAttr(UPPER)   # travel limits -> nail can't fall through

# fixed root joint + articulation root
rj = UsdPhysics.FixedJoint.Define(stage, f"{ROOT}/root_joint")
rj.CreateBody1Rel().SetTargets([f"{ROOT}/base"])
UsdPhysics.ArticulationRootAPI.Apply(rj.GetPrim()); PhysxSchema.PhysxArticulationAPI.Apply(rj.GetPrim())

stage.Save()
print(f"[build] wrote {OUT}  base_scale={BASE_SCALE} screw_scale={SCREW_SCALE} translate=({TX},{TY},{TZ}) joint=PRISMATIC(Z, free)")
app.close()
