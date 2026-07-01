"""Graft the 22-DOF Sharpa hands onto the BEHAVIOR R1 Pro arms (replacing the parallel grippers).

Produces a combined URDF (assets/r1pro_sharpa/r1pro_sharpa.urdf) for the GR00T REAL_R1_PRO_SHARPA
embodiment: dual 7-DOF arms + dual 22-DOF Sharpa hands. The left Sharpa sub-tree is used as-is
(joint names already left_*); the right hand is a sagittal MIRROR of it (y-negated origins, mirrored
orientations/axes, mesh scale (1,-1,1)), renamed right_*. The R1 Pro grippers' finger joints/links are
removed; the realsense wrist cameras + the zed head camera are kept (they map to GR00T's wrist/ego views).

Pure XML (no sim). Convert to USD afterwards with Isaac Lab's scripts/tools/convert_urdf.py.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

R1 = "/home/cning/simtoolreal_isaaclab/assets/behavior_r1pro/models/r1pro/urdf/r1pro.urdf"
R1_DIR = os.path.dirname(R1)
SHARPA = "/home/cning/simtoolreal_isaaclab/assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
SHARPA_DIR = os.path.dirname(SHARPA)
OUT_DIR = "/home/cning/simtoolreal_isaaclab/assets/r1pro_sharpa"
OUT = f"{OUT_DIR}/r1pro_sharpa.urdf"
os.makedirs(OUT_DIR, exist_ok=True)

# fixed-joint transform from <arm>_arm_link7 to the Sharpa mount (where the gripper used to be).
# The R1 Pro wrist's tool-forward is -z (the gripper hung at -0.16 z), but the Sharpa hand extends
# along the mount's +z, so flip 180deg about x to make the hand point DOWN/out (-z); keep the
# original 15deg (0.2618) mount z-roll. (left values; right is the sagittal mirror.)
# z = -0.05: hand-to-arm_link7 offset (closer to the wrist, per request).
MOUNT_XYZ = (-0.0295, 0.0, -0.05)
# roll=pi flips the hand to point -z (the R1 wrist tool-forward); yaw = 0.2618 (orig 15deg roll) + pi
# spins the hand 180deg about the arm axis so the palm faces the right way.
MOUNT_RPY = (3.14159265, 0.0, 0.261799 + 3.14159265)


def absify(elem, base):
    for mesh in elem.iter("mesh"):
        fn = mesh.get("filename")
        if fn and not fn.startswith("/"):
            mesh.set("filename", os.path.normpath(os.path.join(base, fn)))


def fix_missing_meshes(parent):
    """The R1 Pro dataset ships visual meshes but NOT the collision meshes the URDF references
    (meshes/collision/<x>-col-N.obj). Remap any missing mesh to the link's visual sibling
    (meshes/<x>.obj); drop the element if even that is absent."""
    import re
    for owner_tag in ("collision", "visual"):
        for link in parent.findall("link"):
            for el in list(link.findall(owner_tag)):
                m = el.find(".//mesh")
                if m is None:
                    continue
                fn = m.get("filename")
                if fn and os.path.exists(fn):
                    continue
                vis = re.sub(r"/collision/(.+?)-col-\d+\.(obj|stl|STL)$", r"/\1.obj", fn or "")
                if vis != fn and os.path.exists(vis):
                    m.set("filename", vis)
                else:
                    link.remove(el)   # no usable mesh -> drop this collision/visual element


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_rpy(R):
    p = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    y = np.arctan2(R[1, 0], R[0, 0])
    r = np.arctan2(R[2, 1], R[2, 2])
    return r, p, y


P = np.diag([1.0, -1.0, 1.0])   # reflection across the x-z (sagittal) plane


def mirror_origin(origin):
    xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
    xyz = (P @ np.array(xyz)).tolist()
    Rm = P @ rpy_to_R(*rpy) @ P
    rpy = R_to_rpy(Rm)
    origin.set("xyz", " ".join(f"{v:.6g}" for v in xyz))
    origin.set("rpy", " ".join(f"{v:.6g}" for v in rpy))


def mirror_axis(axis):
    a = [float(v) for v in axis.get("xyz", "1 0 0").split()]
    a = (-P @ np.array(a)).tolist()   # pseudovector under reflection: a -> -P a
    axis.set("xyz", " ".join(f"{v:.6g}" for v in a))


def rename(elem, attr):
    v = elem.get(attr)
    if v is None:
        return
    if v.startswith("sharpa_mount"):   # "sharpa_mount" -> "sharpa_mount_R", "sharpa_mount_joint" -> "sharpa_mount_R_joint"
        elem.set(attr, "sharpa_mount_R" + v[len("sharpa_mount"):])
    elif v.startswith("left_"):
        elem.set(attr, "right_" + v[len("left_"):])


# ---- load R1 Pro, abs-ify meshes, drop gripper fingers ----
r1 = ET.parse(R1); root = r1.getroot()
absify(root, R1_DIR)
fix_missing_meshes(root)   # remap the R1 Pro's missing collision meshes -> visual meshes
# drop the whole gripper + realsense camera-mounting chain on each arm; the Sharpa hand mounts
# directly on <side>_arm_link7 (its parent) and the env attaches the wrist camera to arm_link7.
DROP = {f"{s}_{p}" for s in ("left", "right") for p in (
    "gripper_finger_joint1", "gripper_finger_joint2", "gripper_finger_link1", "gripper_finger_link2",
    "gripper_joint", "gripper_link", "realsense_joint", "realsense_link")}
for e in list(root):
    if e.tag in ("joint", "link") and e.get("name") in DROP:
        root.remove(e)

# ---- extract the Sharpa hand sub-tree (sharpa_mount + left_* hand; exclude iiwa + the iiwa->mount joint) ----
sh = ET.parse(SHARPA); sroot = sh.getroot()
absify(sroot, SHARPA_DIR)
HAND_PREFIXES = ("left_thumb", "left_index", "left_middle", "left_ring", "left_pinky", "left_hand_", "left_1_", "left_2_", "left_3_", "left_4_", "left_5_")
def is_hand_link(n): return n == "sharpa_mount" or n.startswith(HAND_PREFIXES)
hand_links = [e for e in sroot.findall("link") if is_hand_link(e.get("name"))]
hand_joints = []
for j in sroot.findall("joint"):
    if j.get("name") == "iiwa14_sharpa":
        continue  # the iiwa->mount joint; we re-attach to the R1 Pro wrist instead
    c = j.find("child");
    if c is not None and is_hand_link(c.get("link")):
        hand_joints.append(j)


def attach_hand(side, link7):
    """Insert a Sharpa hand sub-tree attached to <side>_arm_link7. side in {left,right}; right is mirrored."""
    import copy
    mount_name = "sharpa_mount" if side == "left" else "sharpa_mount_R"
    # fixed mount joint from the arm wrist to the Sharpa mount
    mj = ET.SubElement(root, "joint"); mj.set("name", f"{side}_sharpa_mount_joint"); mj.set("type", "fixed")
    o = ET.SubElement(mj, "origin")
    xyz = list(MOUNT_XYZ); rpy = list(MOUNT_RPY)
    if side == "right":
        xyz = (P @ np.array(xyz)).tolist()
        rpy = list(R_to_rpy(P @ rpy_to_R(*rpy) @ P))   # mirror the mount rotation too
    o.set("xyz", " ".join(f"{v:.6g}" for v in xyz)); o.set("rpy", " ".join(f"{v:.6g}" for v in rpy))
    ET.SubElement(mj, "parent").set("link", link7)
    ET.SubElement(mj, "child").set("link", mount_name)
    for src in hand_links + hand_joints:
        e = copy.deepcopy(src)
        if side == "right":
            for tag, attr in (("link", "name"), ("joint", "name")):
                if e.tag == tag:
                    rename(e, attr)
            for sub in e.iter():
                if sub.tag in ("parent", "child"):
                    rename(sub, "link")
            if e.tag == "joint":
                for org in e.findall("origin"):
                    mirror_origin(org)
                ax = e.find("axis")
                if ax is not None:
                    mirror_axis(ax)
            for org in e.findall(".//origin"):     # mirror link visual/collision/inertial origins
                if e.tag == "link":
                    mirror_origin(org)
            for m in e.iter("mesh"):
                s = [float(v) for v in m.get("scale", "1 1 1").split()]
                s[1] = -s[1]
                m.set("scale", " ".join(f"{v:.6g}" for v in s))
        else:
            pass  # left: use as-is (names already left_*)
        root.append(e)


attach_hand("left", "left_arm_link7")
attach_hand("right", "right_arm_link7")

ET.indent(r1, space="  ")
r1.write(OUT, xml_declaration=True, encoding="utf-8")
nj = len(root.findall("joint")); nl = len(root.findall("link"))
print(f"[graft] wrote {OUT}  joints={nj} links={nl}")
rev = [j.get("name") for j in root.findall("joint") if j.get("type") == "revolute"]
print(f"[graft] revolute joints={len(rev)}  (expect 7+7 arm + 22+22 hand + 4 torso + steer = ~46)")
print("  left hand revolute:", [j for j in rev if j.startswith(("left_thumb","left_index","left_middle","left_ring","left_pinky","left_1","left_2","left_3","left_4","left_5"))][:3], "...")
print("  right hand revolute:", [j for j in rev if j.startswith(("right_thumb","right_index","right_middle","right_ring","right_pinky","right_1","right_2","right_3","right_4","right_5"))][:3], "...")
