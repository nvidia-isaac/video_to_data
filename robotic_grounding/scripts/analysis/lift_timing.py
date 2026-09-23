import numpy as np, glob, pyarrow.parquet as pq
base='/workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_refined/robot_name=sharpa_wave/'
t=pq.read_table(glob.glob(base+'*.parquet')[0]); col=lambda n: np.array(t.column(n)[0].as_py())
z=col('object_body_position')[:,0,2]; T=len(z); z0=z.min()
h=z-z0
print(f"frames {T}, env horizon 518 (motion_speed 0.5)")
print(f"object height above rest: max {h.max()*100:.1f} cm")
for thr in (0.02,0.05,0.10):
    air=h>thr
    idx=np.where(air)[0]
    if len(idx)==0: print(f"  >{thr*100:.0f}cm: never"); continue
    print(f"  >{thr*100:4.0f} cm 인 프레임: {air.sum():3d}/{T} ({100*air.mean():4.1f}%)  최초 {idx[0]:3d}  최후 {idx[-1]:3d}")
first=np.where(h>0.02)[0][0]
print(f"\n집어올리기 구간: frame 0 ~ {first} (전체의 {100*first/T:.1f}%)")
print(f"→ 무작위 시작 시 집어올리기를 경험할 확률 = {100*first/T:.1f}%")
print(f"→ 그 외 {100*(1-first/T):.1f}% 는 '이미 들려 있거나 이미 내려놓은' 상태에서 시작")
