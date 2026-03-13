import os

# Define base directory for SMPL models within the module
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(BASE_DIR)
SMPL_MODEL_ROOT = os.environ.get('SMPL_MODEL_ROOT', os.path.join(MODULE_ROOT, 'data', 'smpl_models'))
SMPL_ASSETS_ROOT = os.path.join(SMPL_MODEL_ROOT, 'assets')

# related to smpl and smplh parameter count
SMPL_POSE_PRAMS_NUM = 72
SMPLH_POSE_PRAMS_NUM = 156
SMPLH_HANDPOSE_START = 66 # hand pose start index for smplh
NUM_BETAS = 10

# split smplh
GLOBAL_POSE_NUM = 3
BODY_POSE_NUM = 63
HAND_POSE_NUM = 90
TOP_BETA_NUM = 2

# split smpl
SMPL_HAND_POSE_NUM=6

SMPL_PARTS_NUM = 14

# 24 SMPL joints 
JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand', 
]
