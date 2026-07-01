"""screw_assembly043_wideslot.usd: the 043 physical assembly with the 60%-WIDER cross slot.
Identical to screw_assembly043 (same scale 0.012948, orient, translate, revolute axis Z) — only the
screw geometry is the wide-slot SDF (screw_new_wideslot_sdf). For the 'wider slot' experiment."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a=p.parse_args(); a.headless=True
app=AppLauncher(a).app
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
import os
A="/home/cning/simtoolreal_isaaclab/assets/usd"
TT=f"{A}/thread_test/thread_test.usd"; SC=f"{A}/screw_new_wideslot_sdf/screw_new_wideslot_sdf.usd"
OUT=f"{A}/screw_assembly043_wideslot/screw_assembly043_wideslot.usd"; os.makedirs(os.path.dirname(OUT),exist_ok=True)
R="/screw_assembly043_wideslot"
stage=Usd.Stage.CreateNew(OUT); UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage,1.0)
root=UsdGeom.Xform.Define(stage,R); stage.SetDefaultPrim(root.GetPrim())
def link(path,mass):
    x=UsdGeom.Xform.Define(stage,path); UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass); return x
link(f"{R}/base",1.0)
bg=UsdGeom.Xform.Define(stage,f"{R}/base/geom"); bg.AddScaleOp().Set(Gf.Vec3f(0.004875,0.004875,0.004875))
UsdGeom.Xform.Define(stage,f"{R}/base/geom/ref").GetPrim().GetReferences().AddReference(TT,"/threadTest/geometry")
sc=link(f"{R}/screw",0.05); sc.AddTranslateOp().Set(Gf.Vec3d(0.0288,0.0,0.0478))
sg=UsdGeom.Xform.Define(stage,f"{R}/screw/geom")
sg.AddOrientOp().Set(Gf.Quatf(0.70710678,0.70710678,0.0,0.0)); sg.AddScaleOp().Set(Gf.Vec3f(0.012948,0.012948,0.012948))
UsdGeom.Xform.Define(stage,f"{R}/screw/geom/ref").GetPrim().GetReferences().AddReference(SC,"/screw_wideslot/geometry")
UsdGeom.Scope.Define(stage,f"{R}/joints")
rev=UsdPhysics.RevoluteJoint.Define(stage,f"{R}/joints/screw_spin")
rev.CreateBody0Rel().SetTargets([f"{R}/base"]); rev.CreateBody1Rel().SetTargets([f"{R}/screw"])
rev.CreateAxisAttr("Z"); rev.CreateLocalPos0Attr(Gf.Vec3f(0.0288,0.0,0.0478)); rev.CreateLocalRot0Attr(Gf.Quatf(1,0,0,0))
rev.CreateLocalPos1Attr(Gf.Vec3f(0,0,0)); rev.CreateLocalRot1Attr(Gf.Quatf(1,0,0,0))
rj=UsdPhysics.FixedJoint.Define(stage,f"{R}/root_joint"); rj.CreateBody1Rel().SetTargets([f"{R}/base"])
UsdPhysics.ArticulationRootAPI.Apply(rj.GetPrim()); PhysxSchema.PhysxArticulationAPI.Apply(rj.GetPrim())
stage.Save(); open("/tmp/coll/asm043_ws_built.txt","w").write(f"built {OUT}\n"); print(f"[build] {OUT}"); app.close()
