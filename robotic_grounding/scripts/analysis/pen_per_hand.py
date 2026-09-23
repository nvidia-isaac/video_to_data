import sys, glob, numpy as np, pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0, '/workspace/video_to_data/robotic_grounding/scripts')
import filter_penetrations as F

seq = Path('/workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_refined/robot_name=sharpa_wave')
t = pq.read_table(glob.glob(str(seq/'*.parquet'))[0])
data = {c: [t.column(c)[0].as_py()] for c in t.column_names}

right_shapes, left_shapes = F._get_shapes()
rf = data['robot_right_frames'][0]; lf = data['robot_left_frames'][0]
rn = data['right_robot_frame_names'][0]; ln = data['left_robot_frame_names'][0]
op = data['object_body_position'][0]; ow = data['object_body_wxyz'][0]
mp = data['object_mesh_paths'][0]
rh = F._HandShapeCache(right_shapes, rn); lh = F._HandShapeCache(left_shapes, ln)
hull, ratio = F._load_hull(mp[0], {}, seq)
print("hull_volume_ratio", round(ratio, 3))
R = []; L = []; HH = []
for i in range(len(rf)):
    rc = rh.world_spheres(rf[i]); lc = lh.world_spheres(lf[i])
    pos = np.array(op[i][0], float); Rm = F._quat_wxyz_to_matrix(ow[i][0])
    R.append(F._max_hand_object_penetration(rc, hull, pos, Rm))
    L.append(F._max_hand_object_penetration(lc, hull, pos, Rm))
    HH.append(F._max_hand_hand_penetration(rc, lc))
R = np.array(R); L = np.array(L); HH = np.array(HH)
for n, v in (("RIGHT-object", R), ("LEFT-object", L), ("hand-hand", HH)):
    print(f"{n:14s} mean {v.mean()*100:6.2f}  p95 {np.percentile(v,95)*100:6.2f}  max {v.max()*100:6.2f} cm   frames>2cm {(v>0.02).sum():3d}/{len(v)}")
print("frame 0:  right %.2f  left %.2f  hh %.2f cm" % (R[0]*100, L[0]*100, HH[0]*100))
print("argmax right frame", int(R.argmax()), " left frame", int(L.argmax()))
