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

MANO_FINGERTIP_INDICES = [4, 8, 12, 16, 20]

#############################################################
# NVHuman parameters
#############################################################

NVHUMAN_JOINTS_ORDER = [
    "Hips",  # 0
    "Spine1",  # 1
    "Spine2",  # 2
    "Chest",  # 3
    "Neck1",  # 4
    "Neck2",  # 5
    "Head",  # 6
    "HeadEnd",  # 7
    "Jaw",  # 8
    "LeftEye",  # 9
    "RightEye",  # 10
    "LeftShoulder",  # 11
    "LeftArm",  # 12
    "LeftForeArm",  # 13
    "LeftHand",  # 14
    "LeftHandThumb1",  # 15
    "LeftHandThumb2",  # 16
    "LeftHandThumb3",  # 17
    "LeftHandThumbEnd",  # 18
    "LeftHandIndex1",  # 19
    "LeftHandIndex2",  # 20
    "LeftHandIndex3",  # 21
    "LeftHandIndex4",  # 22
    "LeftHandIndexEnd",  # 23
    "LeftHandMiddle1",  # 24
    "LeftHandMiddle2",  # 25
    "LeftHandMiddle3",  # 26
    "LeftHandMiddle4",  # 27
    "LeftHandMiddleEnd",  # 28
    "LeftHandRing1",  # 29
    "LeftHandRing2",  # 30
    "LeftHandRing3",  # 31
    "LeftHandRing4",  # 32
    "LeftHandRingEnd",  # 33
    "LeftHandPinky1",  # 34
    "LeftHandPinky2",  # 35
    "LeftHandPinky3",  # 36
    "LeftHandPinky4",  # 37
    "LeftHandPinkyEnd",  # 38
    "LeftForeArmTwist1",  # 39
    "LeftForeArmTwist2",  # 40
    "LeftArmTwist1",  # 41
    "LeftArmTwist2",  # 42
    "RightShoulder",  # 43
    "RightArm",  # 44
    "RightForeArm",  # 45
    "RightHand",  # 46
    "RightHandThumb1",  # 47
    "RightHandThumb2",  # 48
    "RightHandThumb3",  # 49
    "RightHandThumbEnd",  # 50
    "RightHandIndex1",  # 51
    "RightHandIndex2",  # 52
    "RightHandIndex3",  # 53
    "RightHandIndex4",  # 54
    "RightHandIndexEnd",  # 55
    "RightHandMiddle1",  # 56
    "RightHandMiddle2",  # 57
    "RightHandMiddle3",  # 58
    "RightHandMiddle4",  # 59
    "RightHandMiddleEnd",  # 60
    "RightHandRing1",  # 61
    "RightHandRing2",  # 62
    "RightHandRing3",  # 63
    "RightHandRing4",  # 64
    "RightHandRingEnd",  # 65
    "RightHandPinky1",  # 66
    "RightHandPinky2",  # 67
    "RightHandPinky3",  # 68
    "RightHandPinky4",  # 69
    "RightHandPinkyEnd",  # 70
    "RightForeArmTwist1",  # 71
    "RightForeArmTwist2",  # 72
    "RightArmTwist1",  # 73
    "RightArmTwist2",  # 74
    "LeftLeg",  # 75
    "LeftShin",  # 76
    "LeftFoot",  # 77
    "LeftToeBase",  # 78
    "LeftToeEnd",  # 79
    "LeftShinTwist1",  # 80
    "LeftShinTwist2",  # 81
    "LeftLegTwist1",  # 82
    "LeftLegTwist2",  # 83
    "RightLeg",  # 84
    "RightShin",  # 85
    "RightFoot",  # 86
    "RightToeBase",  # 87
    "RightToeEnd",  # 88
    "RightShinTwist1",  # 89
    "RightShinTwist2",  # 90
    "RightLegTwist1",  # 91
    "RightLegTwist2",  # 92
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
    ".*_hand_C_MC": ("wrist", 0.2, 0.2),
    # ".*_thumb_CMC_VL_site": ("thumb1", 0.0, 0.0),
    ".*_thumb_MCP_VL_site": ("thumb2", 0.1, 0.0),
    # ".*_thumb_DP_site": ("thumb3", 0.0, 0.0),
    ".*_thumb_tip_site": ("thumb4", 1.0, 0.05),
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
    ".*_pinky_tip_site": ("pinky4", 0.5, 0.1),
}

SHARPA_RELATIVE_FRAMES = [
    # Target site: (root site, position cost, orientation cost)
    (".*_thumb_tip_site", ".*_index_tip_site", 0.1, 0.0),
    (".*_thumb_tip_site", ".*_middle_tip_site", 0.1, 0.0),
    (".*_thumb_tip_site", ".*_ring_tip_site", 0.1, 0.0),
    (".*_thumb_tip_site", ".*_pinky_tip_site", 0.1, 0.0),
    (".*_index_tip_site", ".*_middle_tip_site", 0.1, 0.0),
    (".*_index_tip_site", ".*_ring_tip_site", 0.1, 0.0),
    (".*_index_tip_site", ".*_pinky_tip_site", 0.1, 0.0),
    (".*_middle_tip_site", ".*_ring_tip_site", 0.1, 0.0),
    (".*_middle_tip_site", ".*_pinky_tip_site", 0.1, 0.0),
    (".*_ring_tip_site", ".*_pinky_tip_site", 0.1, 0.0),
]

#############################################################
# IK parameters for Dex3 hand
#############################################################

DEX3_TO_NVHUMAN_MAPPING = {
    # Dex3 .* hand sites: (target NVHuman joint, position cost, orientation cost)
    ".*_hand_palm_link": (".*Hand", 1.0, 0.1),
    ".*_thumb_tip": (".*HandThumbEnd", 1.0, 0.0),
    ".*_index_tip": (".*HandIndexEnd", 1.0, 0.0),
    ".*_middle_tip": (".*HandRingEnd", 1.0, 0.0),  # Map ring finger to middle
}

# NVHuman convention: X=left, Y=up, Z=forward
# Robot convention: X=forward, Y=left, Z=up
# Mapping: NVHuman Z → Robot X, NVHuman X → Robot Y, NVHuman Y → Robot Z
R_NVHUMAN_TO_ROBOT = [
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
]

# Palm frame corrections to align robot palm frame with human hand orientation.
R_PALM_CORRECTION_LEFT = [
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
]

# Right hand: R_y(90°) @ R_z(-90°) - first -90° about Z, then 90° about Y
R_PALM_CORRECTION_RIGHT = [
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
]

#############################################################
# MANO hand link definitions
#############################################################

# MANO hand link definitions: (link_name, list of joint indices).
# Used to assign contact points to the closest link via joint distances.
MANO_HAND_LINKS = {
    "link_palm": [0, 1, 5, 9, 13, 17],
    "link_thumb1": [1, 2],
    "link_thumb2": [2, 3],
    "link_thumb3": [3, 4],
    "link_index1": [5, 6],
    "link_index2": [6, 7],
    "link_index3": [7, 8],
    "link_middle1": [9, 10],
    "link_middle2": [10, 11],
    "link_middle3": [11, 12],
    "link_ring1": [13, 14],
    "link_ring2": [14, 15],
    "link_ring3": [15, 16],
    "link_pinky1": [17, 18],
    "link_pinky2": [18, 19],
    "link_pinky3": [19, 20],
}

NUM_MANO_LINKS = len(MANO_HAND_LINKS)
