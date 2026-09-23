import numpy as np, glob, pyarrow.parquet as pq
base='/workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_refined/robot_name=sharpa_wave/'
t=pq.read_table(glob.glob(base+'*.parquet')[0]); col=lambda n: np.array(t.column(n)[0].as_py())
p=col('object_body_position')[:,0,:]; q=col('object_body_wxyz')[:,0,:]
def R(w):
    w0,x,y,z=w
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w0),2*(x*z+y*w0)],
                     [2*(x*y+z*w0),1-2*(x*x+z*z),2*(y*z-x*w0)],
                     [2*(x*z-y*w0),2*(y*z+x*w0),1-2*(x*x+y*y)]])
V=np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]],float)
K=np.stack([p[i]+ (R(q[i])@V.T).T for i in range(len(p))])   # (T,6,3) reference keypoints
var=0.1
for name, idx in (("frozen at frame 0", 0), ("frozen at mid frame", len(p)//2)):
    Kf=K[idx][None].repeat(len(p),0)
    d2=((K-Kf)**2).sum(-1)
    print(f"{name:22s} mean reward = {np.exp(-d2/var).mean():.4f}")
d2p=((K-K)**2).sum(-1); print(f"{'perfect tracking':22s} mean reward = 1.0000")
# position-only: how much does the 15.4cm lift cost if orientation is perfect?
lift=p[:,2]-p[0,2]
d2_lift=(lift**2)[:,None].repeat(6,1)
print(f"{'lift error only':22s} mean reward = {np.exp(-d2_lift/var).mean():.4f}   (box z frozen, orientation perfect)")
ang=[]
for i in range(len(p)):
    Rr=R(q[i])@R(q[0]).T; ang.append(np.arccos(np.clip((np.trace(Rr)-1)/2,-1,1)))
ang=np.array(ang)
print(f"reference: z range {lift.max()-lift.min():.4f} m, orientation drift vs frame0: mean {ang.mean():.3f} max {ang.max():.3f} rad")
