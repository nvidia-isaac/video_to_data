import numpy as np, glob, pyarrow.parquet as pq
base='/workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_refined/robot_name=sharpa_wave/'
t=pq.read_table(glob.glob(base+'*.parquet')[0]); col=lambda n: np.array(t.column(n)[0].as_py())
p=col('object_body_position')[:,0,:]; q=col('object_body_wxyz')[:,0,:]
def qmul(a,b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
def qerr(a,b):
    d=qmul(np.array([a[0],-a[1],-a[2],-a[3]]), b)
    return 2*np.arccos(np.clip(abs(d[0]),-1,1))
T=len(p)
dp=np.linalg.norm(p-p[0],axis=1)                      # 상자를 frame 0에 고정했을 때의 위치 오차
dq=np.array([qerr(q[0],q[i]) for i in range(T)])      # 같은 조건의 자세 오차
print("== 상자를 frame 0에 완전히 고정한 정책이 받는 objAway 입력 ==")
print(f"위치 오차  max {dp.max():.4f} m   (임계 0.20 m, 여유 {100*(1-dp.max()/0.20):.0f}%)")
print(f"자세 오차  max {dq.max():.4f} rad (임계 0.70 rad, 여유 {100*(1-dq.max()/0.70):.0f}%)")
print(f"→ 두 조건 모두 한 번도 발화하지 않음\n")
print("임계값을 낮추면 (위치 기준):")
for thr in (0.20,0.16,0.14,0.12,0.10,0.08):
    hit=np.where(dp>thr)[0]
    if len(hit)==0: print(f"  {thr:.2f} m : 발화 안 함")
    else: print(f"  {thr:.2f} m : frame {hit[0]:3d} 에서 최초 발화  ({100*hit[0]/T:4.1f}% 지점, 전체 {len(hit):3d}/{T} 프레임에서 초과)")
print(f"\n참고: 레퍼런스 수평 이동 {np.linalg.norm(p[:,:2]-p[0,:2],axis=1).max():.4f} m, 수직 {abs(p[:,2]-p[0,2]).max():.4f} m")
