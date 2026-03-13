"""
simple wrapper to generate SMPLH instances

Author: Xianghui Xie
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""

import numpy as np
import sys, os
sys.path.append(os.getcwd())
from lib_mesh.mesh import Mesh
import torch

from lib_smpl.th_hand_prior import mean_hand_pose
from lib_smpl.wrapper_pytorch import SMPLPyTorchWrapperBatch, SMPL_MODEL_ROOT, SMPL_ASSETS_ROOT
from lib_smpl.th_smpl_prior import get_prior
from lib_smpl.const import (
SMPLH_POSE_PRAMS_NUM, SMPLH_HANDPOSE_START,
)


class SMPLHGenerator:
    def __init__(self):
        pass

    @staticmethod
    def get_mean_smplh(offsets=None,):
        "use mean smpl pose"
        return SMPLHGenerator.gen_smplh('male', np.zeros((1, 3)), 1, offsets=offsets)[0]

    @staticmethod
    def get_zero_smplh(offsets=None,):
        "use mean smpl pose"
        return SMPLHGenerator.gen_smplh('male', np.zeros((1, 3)), 1, offsets=offsets,
                                        smpl_pose_init=np.zeros((1, 72)))[0]

    @staticmethod
    def gen_smplh(gender, centers,
                  batch_sz,
                  beta_init=None,
                  smpl_pose_init=None,
                  offsets=None,
                  device='cuda:0',
                  return_mesh=True):
        """
        generate smpl mesh from a set of complete parameters
        """
        prior = get_prior()
        pose_init = torch.zeros((batch_sz, 156))
        pose_init[:, 3:SMPLH_HANDPOSE_START] = prior.mean
        betas_init = torch.zeros((batch_sz, 10))
        if beta_init is not None:
            betas_init[:, :] = torch.tensor(beta_init)
            beta_init = torch.tensor(betas_init).to(device)  # move to GPU for late loss computation
        if smpl_pose_init is not None:
            pose_init[:, :72] = torch.tensor(smpl_pose_init[:, :72])

        if smpl_pose_init.shape[1] != SMPLH_POSE_PRAMS_NUM:
            # use mean hand pose
            hand_mean = mean_hand_pose(SMPL_ASSETS_ROOT)
            hand_init = torch.tensor(hand_mean, dtype=torch.float).to(device)
            pose_init[:, SMPLH_HANDPOSE_START:] = hand_init
        else:
            pose_init = torch.Tensor(smpl_pose_init)

        betas, pose, trans = beta_init, pose_init, centers  # init SMPL with the translation

        if offsets is not None and isinstance(offsets, np.ndarray):
            offsets = torch.from_numpy(offsets)

        smplh = SMPLPyTorchWrapperBatch(SMPL_MODEL_ROOT, batch_sz, betas, pose, trans, offsets,
                                        gender=gender, num_betas=10, hands=True, device=device).to(device)

        if return_mesh:
            with torch.no_grad():
                # print(smplh.faces.cpu().shape)
                verts, _, _, _ = smplh()
                meshes = []
                faces = smplh.faces.cpu().numpy()
                for v in verts:
                    m = Mesh(v=v.cpu().numpy(), f=faces)
                    meshes.append(m)
            return meshes
        else:
            return smplh

    @staticmethod
    def get_smplh(poses, betas, trans, gender, device='cuda:0'):
        "generate smplh from a complete set of parameters"
        batch_sz = len(poses)
        pose_param_num = poses.shape[1]
        if pose_param_num != SMPLH_POSE_PRAMS_NUM:
            assert pose_param_num == 72, 'using unknown source of smpl poses'
            pose_init = torch.zeros((batch_sz, 156))
            pose_init[:, :pose_param_num] = torch.tensor(poses, dtype=torch.float32)
            pose_init[:, SMPLH_HANDPOSE_START:] = torch.tensor(mean_hand_pose(SMPL_ASSETS_ROOT), dtype=torch.float)
        else:
            pose_init = torch.tensor(poses, dtype=torch.float32)
        betas = torch.tensor(betas, dtype=torch.float32)
        smplh = SMPLPyTorchWrapperBatch(SMPL_MODEL_ROOT, batch_sz, betas.cpu(), pose_init.cpu(), trans.cpu(),
                                        gender=gender, num_betas=10, hands=True, device=device).to(device)
        return smplh
