import numpy as np, glob, pyarrow.parquet as pq
base='/workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_refined/robot_name=sharpa_wave/'
t=pq.read_table(glob.glob(base+'*.parquet')[0]); col=lambda n: np.array(t.column(n)[0].as_py())
p=col('object_body_position')[:,0,:]; q=col('object_body_wxyz')[:,0,:]
def Rm(w):
    w0,x,y,z=w
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w0),2*(x*z+y*w0)],[2*(x*y+z*w0),1-2*(x*x+z*z),2*(y*z-x*w0)],[2*(x*z-y*w0),2*(y*z+x*w0),1-2*(x*x+y*y)]])
Rs=np.stack([Rm(qq) for qq in q])
U=np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]],float)

def headroom(L=1.0, var=0.1, amp=1.0):
    """reward a do-nothing policy gets, for keypoint lever L, variance var, motion amplitude x amp."""
    pp = p[0] + (p - p[0])*amp
    K  = pp[:,None,:] + np.einsum('tij,kj->tki', Rs, U*L)
    Kf = np.repeat((pp[0][None,:] + (Rs[0]@(U*L).T).T)[None], len(pp), 0)
    return np.exp(-((K-Kf)**2).sum(-1)/var).mean()

print("== 무동작 정책이 받는 object_keypoints 보상 (1.0 = 완벽) ==\n")
print("A) var 조정 (레버 1 m 고정, 현재 var=0.1)")
for v in (0.1, 0.05, 0.02, 0.01, 0.005, 0.002):
    h=headroom(var=v); print(f"   var={v:<6} 무동작 {h:.4f}   과제 보상 폭 {100*(1-h):5.1f}%{'   ← 현재' if v==0.1 else ''}")
print("\nB) 키포인트 레버 조정 (var=0.1 고정)")
for L in (1.0, 0.5, 0.2, 0.1, 0.05):
    h=headroom(L=L); print(f"   L={L:<6}m 무동작 {h:.4f}   과제 보상 폭 {100*(1-h):5.1f}%{'   ← 현재' if L==1.0 else ''}")
print("\nC) 원 설계 그대로, 물체 이동 폭만 키우면 (레버 1 m, var 0.1)")
for a in (1, 2, 3, 5, 8):
    h=headroom(amp=a); print(f"   이동 x{a}  (z 범위 {0.1544*a:.2f} m)  무동작 {h:.4f}   과제 보상 폭 {100*(1-h):5.1f}%")
