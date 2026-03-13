'''
if code works:
    Author: Xianghui Xie
else:
    Author: Anonymous
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
'''
import torch
import torch.nn as nn

from .smplpytorch.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from .body_landmark import load_regressors
from .torch_functions import batch_sparse_dense_matmul
from .const import *
import yaml, sys
from .const import SMPL_MODEL_ROOT, SMPL_ASSETS_ROOT


class SMPLPyTorchWrapperBatch(nn.Module):
    def __init__(self, model_root,
                 batch_sz,
                 betas=None, pose=None,
                 trans=None, offsets=None,
                 gender='male', num_betas=300, hands=False,
                 device='cuda:0'):
        super(SMPLPyTorchWrapperBatch, self).__init__()
        self.model_root = model_root
        self.hands = hands # use smpl-h or not
        self.device = device

        if betas is None:
            self.betas = nn.Parameter(torch.zeros(batch_sz, 300))
        else:
            assert betas.ndim == 2
            self.betas = nn.Parameter(betas)
        pose_param_num = SMPLH_POSE_PRAMS_NUM if hands else SMPL_POSE_PRAMS_NUM
        if pose is None:
            self.pose = nn.Parameter(torch.zeros(batch_sz, pose_param_num))
        else:
            assert pose.ndim == 2, f'the given pose shape {pose.shape} is not a batch pose'
            assert pose.shape[1] == pose_param_num, f'given pose param shape {pose.shape} ' \
                                                    f'does not match the model selected: hands={hands}'
            self.pose = nn.Parameter(torch.Tensor(pose))
        if trans is None:
            self.trans = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert trans.ndim == 2
            self.trans = nn.Parameter(torch.Tensor(trans))
        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(batch_sz, 6890, 3))
        else:
            assert offsets.ndim == 3
            self.offsets = nn.Parameter(offsets)

        # self.faces = faces
        self.gender = gender

        # pytorch smpl
        self.smpl = SMPL_Layer(center_idx=0, gender=gender, num_betas=num_betas,
                               model_root=str(model_root), hands=hands)
        self.faces = self.smpl.th_faces.clone().to(device) # XH: no need to input face, it is loaded from model file

        # Landmarks
        self.body25_reg_torch, self.face_reg_torch, self.hand_reg_torch = \
            load_regressors(SMPL_ASSETS_ROOT, batch_size=batch_sz)

    def forward(self):
        verts, jtr, tposed, naked = self.smpl(self.pose,
                                              th_betas=self.betas,
                                              th_trans=self.trans,
                                              th_offsets=self.offsets)
        return verts, jtr, tposed, naked

    def get_landmarks(self):
        """Computes body25 joints for SMPL along with hand and facial landmarks"""

        verts, _, _, _ = self.smpl(self.pose,
                                  th_betas=self.betas,
                                  th_trans=self.trans,
                                  th_offsets=self.offsets)

        J = batch_sparse_dense_matmul(self.body25_reg_torch, verts)
        face = batch_sparse_dense_matmul(self.face_reg_torch, verts)
        hands = batch_sparse_dense_matmul(self.hand_reg_torch, verts)

        return J, face, hands


class SMPLPyTorchWrapperBatchSplitParams(nn.Module):
    """
    Alternate implementation of SMPLPyTorchWrapperBatch that allows us to independently optimise:
     1. global_pose
     2. body pose (63 numbers)
     3. hand pose (6 numbers for SMPL or 90 numbers for SMPLH)
     4. top betas (primarily adjusts bone lengths)
     5. other betas
    """

    def __init__(self, model_root, batch_sz,
                 top_betas=None,
                 other_betas=None,
                 global_pose=None,
                 body_pose=None,
                 hand_pose=None,
                 trans=None,
                 offsets=None,
                 faces=None,
                 gender='male',
                 hands=False,
                 num_betas=300):
        super(SMPLPyTorchWrapperBatchSplitParams, self).__init__()
        self.model_root = model_root
        if top_betas is None:
            self.top_betas = nn.Parameter(torch.zeros(batch_sz, TOP_BETA_NUM))
        else:
            assert top_betas.ndim == 2
            self.top_betas = nn.Parameter(top_betas)
        if other_betas is None:
            self.other_betas = nn.Parameter(torch.zeros(batch_sz, num_betas - TOP_BETA_NUM))
        else:
            assert other_betas.ndim == 2
            self.other_betas = nn.Parameter(other_betas)

        if global_pose is None:
            self.global_pose = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert global_pose.ndim == 2
            self.global_pose = nn.Parameter(global_pose)
        if body_pose is None:
            self.body_pose = nn.Parameter(torch.zeros(batch_sz, BODY_POSE_NUM))
        else:
            assert body_pose.ndim == 2
            self.body_pose = nn.Parameter(body_pose)
        hand_pose_num = HAND_POSE_NUM if hands else SMPL_HAND_POSE_NUM
        if hand_pose is None:
            self.hand_pose = nn.Parameter(torch.zeros(batch_sz, hand_pose_num))
        else:
            assert hand_pose.ndim == 2
            assert hand_pose.shape[
                       1] == hand_pose_num, f'given hand pose dim {hand_pose.shape} does not match target model hand pose num of {hand_pose_num}'
            self.hand_pose = nn.Parameter(hand_pose)

        if trans is None:
            self.trans = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert trans.ndim == 2
            self.trans = nn.Parameter(trans)

        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(batch_sz, 6890, 3))
        else:
            assert offsets.ndim == 3
            self.offsets = nn.Parameter(offsets)

        self.betas = torch.cat([self.top_betas, self.other_betas], axis=1)
        self.pose = torch.cat([self.global_pose, self.body_pose, self.hand_pose], axis=1)

        self.faces = faces
        self.gender = gender

        # pytorch smpl
        self.smpl = SMPL_Layer(center_idx=0, gender=gender, num_betas=num_betas,
                               model_root=str(model_root), hands=hands)

        # Landmarks
        self.body25_reg_torch, self.face_reg_torch, self.hand_reg_torch = \
            load_regressors(SMPL_ASSETS_ROOT, batch_size=batch_sz)

        # cache
        self.verts, self.jtr, self.tposed, self.naked = None, None, None, None

    def forward(self):
        self.betas = torch.cat([self.top_betas, self.other_betas], axis=1)
        self.pose = torch.cat([self.global_pose, self.body_pose, self.hand_pose], axis=1)

        verts, jtr, tposed, naked = self.smpl(self.pose,
                                              th_betas=self.betas,
                                              th_trans=self.trans,
                                              th_offsets=self.offsets)
        self.verts, self.jtr, self.tposed, self.naked = verts, jtr, tposed, naked
        return verts, jtr, tposed, naked

    def get_landmarks(self, use_cache=False):
        """Computes body25 joints for SMPL along with hand and facial landmarks"""

        # verts, _, _, _ = self.smpl(self.pose,
        #                            th_betas=self.betas,
        #                            th_trans=self.trans,
        #                            th_offsets=self.offsets)
        if use_cache:
            verts = self.verts
        else:
            verts, _, _, _ = self.forward()

        J = batch_sparse_dense_matmul(self.body25_reg_torch, verts)
        face = batch_sparse_dense_matmul(self.face_reg_torch, verts)
        hands = batch_sparse_dense_matmul(self.hand_reg_torch, verts)

        return J, face, hands

    @staticmethod
    def from_smpl(smpl: SMPLPyTorchWrapperBatch):
        """
        construct a split smpl from a smpl module, which allows to optimize separate parameters
        Args:
            smpl:

        Returns:

        """
        batch_sz = smpl.pose.shape[0]
        split_smpl = SMPLPyTorchWrapperBatchSplitParams(smpl.model_root,
                                                         batch_sz,
                                                         trans=smpl.trans.data,
                                                         top_betas=smpl.betas.data[:, :TOP_BETA_NUM],
                                                         other_betas=smpl.betas.data[:, TOP_BETA_NUM:],
                                                         global_pose=smpl.pose.data[:, :GLOBAL_POSE_NUM],
                                                         body_pose=smpl.pose.data[:, GLOBAL_POSE_NUM:GLOBAL_POSE_NUM + BODY_POSE_NUM],
                                                         hand_pose=smpl.pose.data[:, GLOBAL_POSE_NUM + BODY_POSE_NUM:],
                                                         faces=smpl.faces, gender=smpl.gender,
                                                        hands=smpl.hands).to(smpl.device)
        return split_smpl

