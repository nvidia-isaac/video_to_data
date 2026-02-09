from .smplpytorch import SMPL_Layer
from .smpl_generator import SMPLHGenerator
from .const import SMPL_MODEL_ROOT, SMPL_ASSETS_ROOT
from .th_hand_prior import GRAB_MEAN_HAND
import numpy as np
import torch

def get_smpl(gender, hands, model_root=SMPL_MODEL_ROOT):
    "simple wrapper to get SMPL model"
    return SMPL_Layer(model_root=model_root,
               gender=gender, hands=hands)


def pose72to156(pose72):
    "convert 72 pose to 156 pose"
    # print(f'pose72: {pose72.shape}')
    zero_body_mean_hand = np.zeros((156, ))
    zero_body_mean_hand[66:] = GRAB_MEAN_HAND
    if isinstance(pose72, torch.Tensor):
        if len(pose72.shape) == 2: # bachied (B, 72) -> (B, 156)
            # pose156 = torch.zeros((pose72.shape[0], 156), device=pose72.device)
            pose156 = torch.from_numpy(zero_body_mean_hand).float().unsqueeze(0).repeat(pose72.shape[0], 1).to(pose72.device)
            pose156[:, :69] = pose72[:, :69]
            pose156[:, 69+45:69+48] = pose72[:, 69:72]
            return pose156
        elif len(pose72.shape) == 1: # single (72) -> (156)
            # pose156 = torch.zeros((156), device=pose72.device)
            pose156 = torch.from_numpy(zero_body_mean_hand).float().to(pose72.device)
            pose156[:69] = pose72[:69]
            pose156[69+45:69+48] = pose72[69:72]
            return pose156
        else:
            raise ValueError(f'pose72 must be a 2D tensor, got {pose72.shape}')
    elif isinstance(pose72, np.ndarray):
        if len(pose72.shape) == 2: # bachied (B, 72) -> (B, 156)
            # pose156 = np.zeros((pose72.shape[0], 156))
            pose156 = np.zeros((pose72.shape[0], 156))
            pose156[:, 66:] = GRAB_MEAN_HAND
            pose156[:, :69] = pose72[:, :69]
            pose156[:, 69+45:69+48] = pose72[:, 69:72]
            return pose156.astype(np.float32)
        elif len(pose72.shape) == 1: # single (72) -> (156)
            # pose156 = np.zeros((156))
            pose156 = np.zeros((156))
            pose156[66:] = GRAB_MEAN_HAND
            pose156[:69] = pose72[:69]
            pose156[69+45:69+48] = pose72[69:72]
            return pose156.astype(np.float32)
        else:
            raise ValueError(f'pose72 must be a 2D tensor, got {pose72.shape}')
    else:
        raise ValueError(f'pose72 must be a torch.Tensor or np.ndarray, got {type(pose72)}')

def pose156to72(pose156):
    "convert 156 pose to 72 pose"
    if isinstance(pose156, torch.Tensor):
        if len(pose156.shape) == 2: # bachied (B, 156) -> (B, 72)
            pose72 = torch.cat([pose156[:, :69], pose156[:, 69+45:69+48]], dim=1)
            return pose72
        elif len(pose156.shape) == 1: # single (156) -> (72)
            pose72 = torch.cat([pose156[:69], pose156[69+45:69+48]], dim=0)
            return pose72
        else:
            raise ValueError(f'pose156 must be a 2D tensor, got {pose156.shape}')
    elif isinstance(pose156, np.ndarray):
        if len(pose156.shape) == 2: # bachied (B, 156) -> (B, 72)
            pose72 = np.concatenate([pose156[:, :69], pose156[:, 69+45:69+48]], axis=1)
            return pose72
        elif len(pose156.shape) == 1: # single (156) -> (72)
            pose72 = np.concatenate([pose156[:69], pose156[69+45:69+48]], axis=0)
            return pose72
        else:
            raise ValueError(f'pose156 must be a 2D tensor, got {pose156.shape}')
    else:
        raise ValueError(f'pose156 must be a torch.Tensor or np.ndarray, got {type(pose156)}')


colors24 = np.array([
    [255,   0,   0],
    [  0, 255,   0],
    [  0,   0, 255],
    [255, 255,   0],
    [  0, 255, 255],
    [255,   0, 255],
    [255, 128,   0],
    [128,   0, 255],
    [  0, 128, 255],
    [128, 255,   0],
    [255,   0, 128],
    [  0, 255, 128],
    [128,   0,   0],
    [  0, 128,   0],
    [  0,   0, 128],
    [128, 128,   0],
    [  0, 128, 128],
    [128,   0, 128],
    [192, 192, 192],
    [128, 128, 128],
    [255, 192, 203],
    [210, 105,  30],
    [255, 215,   0],
    [ 70, 130, 180],

    # repeat for 52-24 more entries, to support SMPLH 
    [255,   0,   0],
    [  0, 255,   0],
    [  0,   0, 255],
    [255, 255,   0],
    [  0, 255, 255],
    [255,   0, 255],
    [255, 128,   0],
    [128,   0, 255],
    [  0, 128, 255],
    [128, 255,   0],
    [255,   0, 128],
    [  0, 255, 128],
    [128,   0,   0],
    [  0, 128,   0],
    [  0,   0, 128],
    [128, 128,   0],
    [  0, 128, 128],
    [128,   0, 128],
    [192, 192, 192],
    [128, 128, 128],
    [255, 192, 203],
    [210, 105,  30],
    [255, 215,   0],
    [ 70, 130, 180],

], dtype=np.uint8)
