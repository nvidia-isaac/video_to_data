# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#############################################################
# MANO parameters
#############################################################

MANO_JOINTS_ORDER = [
    "wrist",
    "thumb1",
    "thumb2",
    "thumb3",
    "thumb4",
    "index1",
    "index2",
    "index3",
    "index4",
    "middle1",
    "middle2",
    "middle3",
    "middle4",
    "ring1",
    "ring2",
    "ring3",
    "ring4",
    "pinky1",
    "pinky2",
    "pinky3",
    "pinky4",
]

MANO_TRANSFORMS_ORDER = [
    "wrist",
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "pinky1",
    "pinky2",
    "pinky3",
    "ring1",
    "ring2",
    "ring3",
    "thumb1",
    "thumb2",
    "thumb3",
]

TRANSFORMS_TO_JOINTS = [
    0,
    13,
    14,
    15,
    15,  # tip
    1,
    2,
    3,
    3,  # tip
    4,
    5,
    6,
    6,  # tip
    10,
    11,
    12,
    12,  # tip
    7,
    8,
    9,
    9,  # tip
]

MANO_JOINTS_PARENTS = [
    -1,
    0,
    1,
    2,
    3,
    0,
    5,
    6,
    7,
    0,
    9,
    10,
    11,
    0,
    13,
    14,
    15,
    0,
    17,
    18,
    19,
]

#############################################################
# IK parameters for Sharpa hand
#############################################################

SHARPA_TO_MANO_ROTATION_OFFSET = {
    # Sharpa hand frame: orientation wxyz offset
    ".*_hand_C_MC": (0.5, -0.5, 0.5, 0.5),
}

SHARPA_TO_MANO_MAPPING = {
    # Sharpa body frame: (target MANO joint, position cost, orientation cost)
    ".*_hand_C_MC": ("wrist", 0.5, 0.2),
    # ".*_thumb_CMC_VL_site": ("thumb1", 0.0, 0.0),
    ".*_thumb_MCP_VL_site": ("thumb2", 0.1, 0.0),
    # ".*_thumb_DP_site": ("thumb3", 0.0, 0.0),
    ".*_thumb_tip_site": ("thumb4", 1.0, 0.1),
    # ".*_index_MCP_VL_site": ("index1", 0.0, 0.0),
    ".*_index_MP_site": ("index1", 0.1, 0.0),
    # ".*_index_DP_site": ("index3", 0.0, 0.0),
    ".*_index_tip_site": ("index4", 1.0, 0.1),
    # ".*_middle_MCP_VL_site": ("middle1", 0.0, 0.0),
    ".*_middle_MP_site": ("middle1", 0.1, 0.0),
    # ".*_middle_DP_site": ("middle3", 0.0, 0.0),
    ".*_middle_tip_site": ("middle4", 1.0, 0.1),
    # ".*_ring_MCP_VL_site": ("ring1", 0.0, 0.0),
    ".*_ring_MP_site": ("ring1", 0.1, 0.0),
    # ".*_ring_DP_site": ("ring3", 0.0, 0.0),
    ".*_ring_tip_site": ("ring4", 1.0, 0.1),
    # ".*_pinky_MC_site": ("pinky1", 0.0, 0.0),
    ".*_pinky_MP_site": ("pinky1", 0.1, 0.0),
    # ".*_pinky_DP_site": ("pinky3", 0.0, 0.0),
    ".*_pinky_tip_site": ("pinky4", 1.0, 0.1),
}
