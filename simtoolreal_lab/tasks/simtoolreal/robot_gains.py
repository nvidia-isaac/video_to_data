"""Per-joint actuator parameters ported verbatim from the original SimToolReal
(isaacgymenvs/tasks/simtoolreal/utils.py: populate_dof_properties) and the robot URDF
(effort limits). Order is JOINT_NAMES_ISAACGYM: arm = IIWA14 (7), hand = left Sharpa (22).

- arm: stiffness/damping/effort set; armature intentionally NOT set ("matches real KUKA").
- hand: stiffness/damping/effort + armature + joint friction set.
"""

ARM_JOINTS = [
    "iiwa14_joint_1", "iiwa14_joint_2", "iiwa14_joint_3", "iiwa14_joint_4",
    "iiwa14_joint_5", "iiwa14_joint_6", "iiwa14_joint_7",
]
HAND_JOINTS = [
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE", "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA", "left_pinky_PIP", "left_pinky_DIP",
]

_ARM_STIFF = [600.0, 600.0, 500.0, 400.0, 200.0, 200.0, 200.0]
_ARM_DAMP = [
    27.027026473513512, 27.027026473513512, 24.672186769721083, 22.067474708266914,
    9.752538131173853, 9.147747263670984, 9.147747263670984,
]
_HAND_STIFF = [6.95, 13.2, 4.76, 6.62, 0.9, 4.76, 6.62, 0.9, 0.9, 4.76, 6.62, 0.9, 0.9,
               4.76, 6.62, 0.9, 0.9, 1.38, 4.76, 6.62, 0.9, 0.9]
_HAND_DAMP = [0.28676845, 0.40845109, 0.20394083, 0.24044435, 0.04190723, 0.20859232,
              0.24595532, 0.04243185, 0.03504461, 0.2085923, 0.24595532, 0.04243185,
              0.03504461, 0.20859226, 0.24595528, 0.04243183, 0.0350446, 0.02782345,
              0.20859229, 0.24595528, 0.04243183, 0.0350446]
_HAND_ARM = [0.0032, 0.0032, 0.00265, 0.00265, 0.0006, 0.00265, 0.00265, 0.0006, 0.00042,
             0.00265, 0.00265, 0.0006, 0.00042, 0.00265, 0.00265, 0.0006, 0.00042, 0.00012,
             0.00265, 0.00265, 0.0006, 0.00042]
_HAND_FRIC = [0.132, 0.132, 0.07456, 0.07456, 0.01276, 0.07456, 0.07456, 0.01276, 0.00378738,
              0.07456, 0.07456, 0.01276, 0.00378738, 0.07456, 0.07456, 0.01276, 0.00378738,
              0.012, 0.07456, 0.07456, 0.01276, 0.00378738]

# effort limits (N·m) from the URDF: arm 300; hand fingers small.
EFFORT = {
    "iiwa14_joint_1": 300.0, "iiwa14_joint_2": 300.0, "iiwa14_joint_3": 300.0, "iiwa14_joint_4": 300.0,
    "iiwa14_joint_5": 300.0, "iiwa14_joint_6": 300.0, "iiwa14_joint_7": 300.0,
    "left_1_thumb_CMC_FE": 3.3, "left_thumb_CMC_AA": 3.3, "left_thumb_MCP_FE": 1.864,
    "left_thumb_MCP_AA": 1.864, "left_thumb_IP": 0.638,
    "left_2_index_MCP_FE": 1.864, "left_index_MCP_AA": 1.864, "left_index_PIP": 0.638, "left_index_DIP": 0.189369,
    "left_3_middle_MCP_FE": 1.864, "left_middle_MCP_AA": 1.864, "left_middle_PIP": 0.638, "left_middle_DIP": 0.189369,
    "left_4_ring_MCP_FE": 1.864, "left_ring_MCP_AA": 1.864, "left_ring_PIP": 0.638, "left_ring_DIP": 0.189369,
    "left_5_pinky_CMC": 0.5285, "left_pinky_MCP_FE": 1.864, "left_pinky_MCP_AA": 1.864,
    "left_pinky_PIP": 0.638, "left_pinky_DIP": 0.189369,
}

ARM_STIFFNESS = dict(zip(ARM_JOINTS, _ARM_STIFF))
ARM_DAMPING = dict(zip(ARM_JOINTS, _ARM_DAMP))
ARM_EFFORT = {j: EFFORT[j] for j in ARM_JOINTS}
HAND_STIFFNESS = dict(zip(HAND_JOINTS, _HAND_STIFF))
HAND_DAMPING = dict(zip(HAND_JOINTS, _HAND_DAMP))
HAND_ARMATURE = dict(zip(HAND_JOINTS, _HAND_ARM))
HAND_FRICTION = dict(zip(HAND_JOINTS, _HAND_FRIC))
HAND_EFFORT = {j: EFFORT[j] for j in HAND_JOINTS}
