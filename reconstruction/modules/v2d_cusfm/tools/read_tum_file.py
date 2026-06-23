# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    return R.from_quat((qx, qy, qz, qw)).as_matrix()


# Load TUM pose file
def read_tum_file(file_path, return_rotation_matrix=False):
    # Skip commented lines starting with '#'
    data = np.loadtxt(file_path, comments='#')

    res = {
        'timestamp_seconds': data[:, 0],
        'xs': data[:, 1],
        'ys': data[:, 2],
        'zs': data[:, 3],
        'qxs': data[:, 4],
        'qys': data[:, 5],
        'qzs': data[:, 6],
        'qws': data[:, 7],
    }

    if return_rotation_matrix:
        res['rotation_matrices'] = np.array(
            [
                quaternion_to_rotation_matrix(
                    line[4], line[5], line[6], line[7]) for line in data
            ])

    return res
