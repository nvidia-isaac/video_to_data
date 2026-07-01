"""Build screw_assembly_flat_sdf.usd: aligned + SDF flat (044) screw on a vertical revolute joint.

Stable physics version of the 044 flat screwdriver task's screw (the original convexDecomposition
screw filled the slot -> blade jammed -> NaN). Uses the PCA-aligned flat screw (shaft->+z, slot->x,
re-centered on the shaft axis) baked to SDF, so the blade enters the slot and the revolute axis = +z
through the screw origin (trivial joint). Same pattern as build_screw_asm043.py.
"""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a=p.parse_args(); a.headless=True
app=AppLauncher(a).app
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
import os
A="/home/cning/simtoolreal_isaaclab/assets/usd"
TT=f"{A}/thread_test/thread_test.usd"; SC=f"{A}/flat_screw_aligned_sdf/flat_screw_aligned_sdf.usd"
OUT=f"{A}/screw_assembly_flat_sdf/screw_assembly_flat_sdf.usd"; os.makedirs(os.path.dirname(OUT),exist_ok=True)
TX,TY,TZ=0.0318,0.0,0.0348            # aligned-screw centroid rel. to the assembly (thread_test) origin
stage=Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage,1.0)
root=UsdGeom.Xform.Define(stage,"/screw_assembly_flat"); stage.SetDefaultPrim(root.GetPrim())
def link(path,mass):
    x=UsdGeom.Xform.Define(stage,path)
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim()); UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
    return x
# fixed base = thread_test (scale 0.005)
link("/screw_assembly_flat/base",1.0)
bg=UsdGeom.Xform.Define(stage,"/screw_assembly_flat/base/geom"); bg.AddScaleOp().Set(Gf.Vec3f(0.005,0.005,0.005))
UsdGeom.Xform.Define(stage,"/screw_assembly_flat/base/geom/ref").GetPrim().GetReferences().AddReference(TT,"/threadTest/geometry")
# revolute-jointed flat screw (aligned: shaft +z, slot +x -> NO orient op; scale 0.013)
sc=link("/screw_assembly_flat/screw",0.05); sc.AddTranslateOp().Set(Gf.Vec3d(TX,TY,TZ))
sg=UsdGeom.Xform.Define(stage,"/screw_assembly_flat/screw/geom"); sg.AddScaleOp().Set(Gf.Vec3f(0.013,0.013,0.013))
UsdGeom.Xform.Define(stage,"/screw_assembly_flat/screw/geom/ref").GetPrim().GetReferences().AddReference(SC,"/flat_screw_aligned/geometry")
# revolute joint (screw spins about world +Z = its shaft)
UsdGeom.Scope.Define(stage,"/screw_assembly_flat/joints")
rev=UsdPhysics.RevoluteJoint.Define(stage,"/screw_assembly_flat/joints/screw_spin")
rev.CreateBody0Rel().SetTargets(["/screw_assembly_flat/base"]); rev.CreateBody1Rel().SetTargets(["/screw_assembly_flat/screw"])
rev.CreateAxisAttr("Z")
rev.CreateLocalPos0Attr(Gf.Vec3f(TX,TY,TZ)); rev.CreateLocalRot0Attr(Gf.Quatf(1,0,0,0))
rev.CreateLocalPos1Attr(Gf.Vec3f(0,0,0)); rev.CreateLocalRot1Attr(Gf.Quatf(1,0,0,0))
rj=UsdPhysics.FixedJoint.Define(stage,"/screw_assembly_flat/root_joint")
rj.CreateBody1Rel().SetTargets(["/screw_assembly_flat/base"])
UsdPhysics.ArticulationRootAPI.Apply(rj.GetPrim()); PhysxSchema.PhysxArticulationAPI.Apply(rj.GetPrim())
stage.Save()
open("/tmp/coll/asm_flat_built.txt","w").write(f"built {OUT} translate=({TX},{TY},{TZ})\n")
print(f"[build] wrote {OUT}")
app.close()
