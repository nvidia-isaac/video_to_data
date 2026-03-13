"""
if code works:
    Author: Xianghui Xie
else:
    Author: Anonymous
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""
import numpy as np
import torch
import pickle as pkl
from os.path import join
import yaml, sys
from .const import SMPL_ASSETS_ROOT


GRAB_MEAN_HAND=np.array([ 0.13566974,  0.09491789, -0.28316078, -0.06223104, -0.0483653 ,
       -0.39977205,  0.13620542, -0.13199732, -0.3829936 , -0.21186522,
        0.07707776, -0.5384531 ,  0.10212211, -0.01378017, -0.49732804,
       -0.0471581 , -0.08448984, -0.1955775 , -0.58500576, -0.1548803 ,
       -0.47505018,  0.17948975, -0.13303751, -0.24022132, -0.3436518 ,
        0.11407528, -0.02665429, -0.23750143, -0.07435384, -0.4635036 ,
       -0.07951606, -0.07775243, -0.43911096, -0.19834545, -0.03837305,
       -0.22386047,  0.74066657,  0.3301243 , -0.11117966, -0.4979891 ,
        0.00626109,  0.1454768 ,  0.62785035, -0.01757009, -0.16062371,
        0.16868931, -0.12404376,  0.35450554, -0.04718762,  0.04999495,
        0.4440688 ,  0.13983883,  0.14151372,  0.37325338, -0.21371473,
       -0.14219724,  0.5842063 ,  0.11580209,  0.0260711 ,  0.55343974,
       -0.07212783,  0.09037765,  0.21028592, -0.6847437 , -0.00735493,
        0.5761462 ,  0.3632393 ,  0.18621148,  0.3402348 , -0.57334983,
       -0.13106765, -0.03578933, -0.291134  ,  0.003825  ,  0.5634436 ,
       -0.10148321,  0.09694234,  0.47672924, -0.22845045,  0.04699614,
        0.26392558,  0.8213351 , -0.2821158 ,  0.1008013 , -0.6013597 ,
       -0.02904042,  0.01898805,  0.733293  ,  0.08564732,  0.02389174],
      dtype=np.float32) # 90x1 

def grab_prior(root_path):
    lhand_data, rhand_data = load_grab_prior(root_path)

    prior = np.concatenate([lhand_data['mean'], rhand_data['mean']], axis=0)
    lhand_prec = lhand_data['precision']
    rhand_prec = rhand_data['precision']

    return prior, lhand_prec, rhand_prec


def load_grab_prior(root_path):
    lhand_path = join(root_path, 'priors', 'lh_prior.pkl')
    rhand_path = join(root_path, 'priors', 'rh_prior.pkl')
    lhand_data = pkl.load(open(lhand_path, 'rb'))
    rhand_data = pkl.load(open(rhand_path, 'rb'))
    return lhand_data, rhand_data


def mean_hand_pose(root_path):
    "mean hand pose computed from grab dataset"
    lhand_data, rhand_data = load_grab_prior(root_path)
    lhand_mean = np.array(lhand_data['mean'])
    rhand_mean = np.array(rhand_data['mean'])
    mean_pose = np.concatenate([lhand_mean, rhand_mean])
    return mean_pose


class HandPrior:
    HAND_POSE_NUM=45
    def __init__(self, prior_path=SMPL_ASSETS_ROOT,
                 prefix=66,
                 device='cuda:0',
                 dtype=torch.float,
                 type='grab'):
        "prefix is the index from where hand pose starts, 66 for SMPL-H"
        self.prefix = prefix
        if type == 'grab':
            prior, lhand_prec, rhand_prec = grab_prior(prior_path)
            self.mean = torch.tensor(prior, dtype=dtype).unsqueeze(axis=0).to(device)
            self.lhand_prec = torch.tensor(lhand_prec, dtype=dtype).unsqueeze(axis=0).to(device)
            self.rhand_prec = torch.tensor(rhand_prec, dtype=dtype).unsqueeze(axis=0).to(device)
        else:
            raise NotImplemented("Only grab hand prior is supported!")

    def __call__(self, full_pose):
        "full_pose also include body poses, this function can be used to compute loss"
        temp = full_pose[:, self.prefix:] - self.mean
        if self.lhand_prec is None:
            return (temp*temp).sum(dim=1)
        else:
            lhand = torch.matmul(temp[:, :self.HAND_POSE_NUM], self.lhand_prec)
            rhand = torch.matmul(temp[:, self.HAND_POSE_NUM:], self.rhand_prec)
            temp2 = torch.cat([lhand, rhand], axis=1)
            return (temp2 * temp2).sum(dim=1)
