"""
Code taken from https://github.com/gulvarol/smplpytorch
Modified such that SMPL now also has per-vertex displacements and SMPL-H model
"""

import os

import numpy as np
import torch
from torch.nn import Module

from ..native.webuser.serialization import ready_arguments
from .tensutils import \
    (th_posemap_axisang, th_with_zeros, th_pack, make_list, subtract_flat_id)


class SMPL_Layer(Module):
    __constants__ = ['kintree_parents', 'gender', 'center_idx', 'num_joints']

    def __init__(self,
                 center_idx=None,
                 gender='neutral',
                 model_root='smpl/native/models',
                 num_betas=300,
                 hands=False):
        """
        Args:
            center_idx: index of center joint in our computations,
            model_root: path to pkl files for the model
            gender: 'neutral' (default) or 'female' or 'male', for smplh, only supports male or female
        """
        super().__init__()

        self.center_idx = center_idx
        self.gender = gender

        self.model_root = model_root
        self.hands = hands
        if self.hands:
            assert self.gender in ['male', 'female'], 'SMPL-H model only supports male or female, not {}'.format(self.gender)
            self.model_path = os.path.join(model_root, f"SMPLH_{self.gender}.pkl")
        else:
            # self.model_folder = os.path.join(model_root, "models_v1.0.0/models")
            self.model_path = os.path.join(model_root, f"SMPL_{self.gender}.pkl")

        smpl_data = ready_arguments(self.model_path)
        self.smpl_data = smpl_data
        self.faces = smpl_data['f'].astype(np.int32)

        self.register_buffer('th_betas',
                             torch.Tensor(smpl_data['betas'].r.copy()).unsqueeze(0))
        self.register_buffer('th_shapedirs',
                             torch.Tensor(smpl_data['shapedirs'][:, :, :num_betas].r.copy()))
        self.register_buffer('th_posedirs',
                             torch.Tensor(smpl_data['posedirs'].r.copy()))
        self.register_buffer(
            'th_v_template',
            torch.Tensor(smpl_data['v_template'].r.copy()).unsqueeze(0))
        self.register_buffer(
            'th_J_regressor',
            torch.Tensor(np.array(smpl_data['J_regressor'].toarray())))
        self.register_buffer('th_weights',
                             torch.Tensor(smpl_data['weights'].r.copy()))
        self.register_buffer('th_faces',
                             torch.Tensor(smpl_data['f'].astype(np.int32)).long())

        # Kinematic chain params
        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents
        self.num_joints = len(parents)  # 24

    def forward(self,
                th_pose_axisang,
                th_betas=torch.zeros(1),
                th_trans=torch.zeros(1, 3),
                th_offsets=None, scale=1., ret_glb_rot=False):
        """
        Args:
        th_pose_axisang (Tensor (batch_size x 72)): pose parameters in axis-angle representation
        th_betas (Tensor (batch_size x 10)): if provided, uses given shape parameters
        th_trans (Tensor (batch_size x 3)): if provided, applies trans to joints and vertices
        th_offsets (Tensor (batch_size x 6890 x 3)): if provided, adds per-vertex offsets in t-pose
        """

        batch_size = th_pose_axisang.shape[0]
        # Convert axis-angle representation to rotation matrix rep.
        th_pose_rotmat = th_posemap_axisang(th_pose_axisang)
        # Take out the first rotmat (global rotation)
        root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3)
        # Take out the remaining rotmats (23 joints)
        th_pose_rotmat = th_pose_rotmat[:, 9:]
        th_pose_map = subtract_flat_id(th_pose_rotmat, self.hands)

        # Below does: v_shaped = v_template + shapedirs * betas, change of template verts due to betas
        # If shape parameters are not provided
        if th_betas is None:
            th_v_shaped = self.th_v_template + torch.matmul(
                self.th_shapedirs, self.th_betas.transpose(1, 0)).permute(2, 0, 1)
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(
                batch_size, 1, 1)
        else:
            th_v_shaped = self.th_v_template + torch.matmul(self.th_shapedirs, th_betas.transpose(1, 0)).permute(2, 0, 1)
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped) # regress joints

        # Below does: v_posed = v_shaped + posedirs * pose_map, change of template verts due to pose
        naked = th_v_shaped + torch.matmul(
            self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)

        # Per vertex offsets
        if th_offsets is not None:
            th_v_posed = naked + th_offsets
        else:
            th_v_posed = naked
        # Final T pose with transformation done!

        # Global rigid transformation
        th_results = []
        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1) # th_j: joints in zero pose
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

        # Rotate each part: this does not change the root joint
        for i in range(self.num_joints - 1):
            i_val = int(i + 1)
            joint_rot = th_pose_rotmat[:, (i_val - 1) * 9:i_val * 9].contiguous().view(batch_size, 3, 3)
            joint_j = th_j[:, i_val, :].contiguous().view(batch_size, 3, 1)
            parent = make_list(self.kintree_parents)[i_val]
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)
            joint_rel_transform = th_with_zeros(torch.cat([joint_rot, joint_j - parent_j], 2))
            th_results.append(torch.matmul(th_results[parent], joint_rel_transform))
        th_results_global = th_results

        th_results2 = root_j.new_zeros((batch_size, 4, 4, self.num_joints))
        # th_results: transformation for each joint, and the global location of each joint
        # th_results2: transformation of ?
        for i in range(self.num_joints):
            padd_zero = th_j.new_zeros(1)
            joint_j = torch.cat([
                th_j[:, i],
                padd_zero.view(1, 1).repeat(batch_size, 1)
            ], 1)
            tmp = torch.bmm(th_results[i], joint_j.unsqueeze(2))
            th_results2[:, :, :, i] = th_results[i] - th_pack(tmp) # transformed joints?

        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))

        th_rest_shape_h = torch.cat([
            th_v_posed.transpose(2, 1),
            th_T.new_ones((batch_size, 1, th_v_posed.shape[1])),
        ], 1)

        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1) # th_T: (1, 4, 4, 6890)
        th_verts = th_verts[:, :, :3]
        th_jtr = torch.stack(th_results_global, dim=1)[:, :, :3, 3] # (B, 24, 3), root joint not changed

        # Scale
        th_verts *= scale
        th_jtr *= scale

        # Shift to new root
        # if self.center_idx is not None:
        #     center_joint = th_jtr[:, self.center_idx].unsqueeze(1)
        #     th_jtr = th_jtr - center_joint
        #     th_verts = th_verts - center_joint
        #
        # # If translation is provided
        # if not(th_trans is None or bool(torch.norm(th_trans) == 0)):
        #     th_jtr = th_jtr + th_trans.unsqueeze(1)
        #     th_verts = th_verts + th_trans.unsqueeze(1)

        # XH: not doing shift, apply translation instead
        th_jtr = th_jtr + th_trans.unsqueeze(1)
        th_verts = th_verts + th_trans.unsqueeze(1)

        if ret_glb_rot:
            return th_verts, th_jtr, th_v_posed, naked, torch.stack(th_results_global, dim=1)[:, :, :3, :3]
        # Vertices and joints in meters
        return th_verts, th_jtr, th_v_posed, naked

    def get_root_joint(self, th_pose_axisang, th_betas, th_trans):
        """
        compute the position of root joint
        Args:
            th_pose_axisang: (B, 72)
            th_betas: (B, num_betas)
            th_trans: (B, 3), global translation

        Returns: (B, 1, 3), the location of root joint

        """
        batch_size = th_pose_axisang.shape[0]
        # Convert axis-angle representation to rotation matrix rep.
        th_pose_rotmat = th_posemap_axisang(th_pose_axisang)
        # Take out the first rotmat (global rotation)
        root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3)
        # Take out the remaining rotmats (23 joints)
        th_pose_rotmat = th_pose_rotmat[:, 9:]
        th_pose_map = subtract_flat_id(th_pose_rotmat, self.hands)

        # Below does: v_shaped = v_template + shapedirs * betas, change of template verts due to betas
        th_v_shaped = self.th_v_template + torch.matmul(self.th_shapedirs, th_betas.transpose(1, 0)).permute(2, 0, 1)
        th_j = torch.matmul(self.th_J_regressor, th_v_shaped)

        # Global rigid transformation (does not affect root joint location)
        th_results = []
        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)  # th_j: joints in zero pose
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

        th_jtr = torch.stack(th_results, dim=1)[:, :, :3, 3]  # (B, 24, 3), root joint not changed
        th_jtr = th_jtr + th_trans.unsqueeze(1)
        return th_jtr

    def get_joints(self, th_pose_axisang, th_betas, th_trans, axis2rot=True, ret_rots=False):
        "regress joints only"
        batch_size = th_pose_axisang.shape[0]
        # Convert axis-angle representation to rotation matrix rep.
        if axis2rot:
            th_pose_rotmat = th_posemap_axisang(th_pose_axisang)
        else:
            th_pose_rotmat = th_pose_axisang # already (B, 9*(num_joints + 1))
        # Take out the first rotmat (global rotation)
        root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3)
        # Take out the remaining rotmats (23 joints)
        th_pose_rotmat = th_pose_rotmat[:, 9:]
        th_pose_map = subtract_flat_id(th_pose_rotmat, self.hands)

        # Below does: v_shaped = v_template + shapedirs * betas, change of template verts due to betas
        # If shape parameters are not provided
        if th_betas is None:
            th_v_shaped = self.th_v_template + torch.matmul(
                self.th_shapedirs, self.th_betas.transpose(1, 0)).permute(2, 0, 1)
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(
                batch_size, 1, 1)
        else:
            th_v_shaped = self.th_v_template + torch.matmul(self.th_shapedirs, th_betas.transpose(1, 0)).permute(2, 0, 1)
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped)  # regress joints
        # Global rigid transformation
        th_results = []
        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)  # th_j: joints in zero pose
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

        # Rotate each part: this does not change the root joint
        for i in range(self.num_joints - 1):
            i_val = int(i + 1)
            joint_rot = th_pose_rotmat[:, (i_val - 1) * 9:i_val * 9].contiguous().view(batch_size, 3, 3)
            joint_j = th_j[:, i_val, :].contiguous().view(batch_size, 3, 1)
            parent = make_list(self.kintree_parents)[i_val]
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)
            joint_rel_transform = th_with_zeros(torch.cat([joint_rot, joint_j - parent_j], 2))
            th_results.append(torch.matmul(th_results[parent], joint_rel_transform))
        th_results_global = th_results
        th_global = torch.stack(th_results_global, dim=1)  # (B, 24, 3)
        th_jtr = th_global[:, :, :3, 3] + th_trans.unsqueeze(1)
        if ret_rots:
            return th_jtr, th_global[:, :, :3, :3] # return the global rotation matrices as well 
        return th_jtr
