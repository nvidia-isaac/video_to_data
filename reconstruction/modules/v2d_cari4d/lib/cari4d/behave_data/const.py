# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
some meta data for the dataset

"""
import json, glob
import numpy as np

# add HODome subjects: Subject01: female 
# 02: male, 
# 03: female, 
# 04: female 
# 06: male 
# 07: female, 
# 08: female 
# 09: male 
# 10: male  
_sub_gender = {
"Sub01": 'male',
"Sub02": 'male',
"Sub03": 'male',
"Sub04": 'male',
"Sub05": 'male',
"Sub06": 'female',
"Sub07": 'female',
"Sub08": 'female',

# HODome subjects
'subject01': 'female',
'subject02': 'male',
'subject03': 'female',
'subject04': 'female',
'subject06': 'male',
'subject07': 'female',
'subject08': 'female',
'subject09': 'male',
'subject10': 'male',

# InterCap subjects
'sub09': 'female',
'sub10': 'male',

# IMHD subjects
'songzn': 'male',
'wangwzh': 'male',
'xujr': 'male',
'yuzhm': 'male',
'zhangjy': 'male',
'zhaochf': 'male',
'af': 'female',
'dujsh': 'male',
'wangjy': 'male',
}

OBJ_NAMES=['backpack', 'basketball', 'boxlarge', 'boxlong', 'boxmedium',
           'boxsmall', 'boxtiny', 'chairblack', 'chairwood', 'keyboard',
           'monitor', 'plasticcontainer', 'stool', 'suitcase', 'tablesmall',
           'tablesquare', 'toolbox', 'trashbin', 'yogaball', 'yogamat']

USE_PSBODY = True # if True, use psbody library to process all meshes, otherwise use trimesh

def get_test_view_id(video_prefix):
    selected_views = json.load(open('splits/selected-views-map.json'))
    if video_prefix not in selected_views:
        print("Warning: video prefix not in selected views map, return None")
        return None  # use default 1 
    return int(selected_views[video_prefix][1])

def get_hy3d_mesh_file(video_prefix, meshes_root='/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/behave/selected-views/hy3d-aligned-center'):
    obj_name = video_prefix.split('_')[2]
    files = sorted(glob.glob(f'{meshes_root}/{video_prefix}*/*{obj_name}*_align.obj'))
    if len(files) == 0:
        print(f'no aligned hy3d template found for {video_prefix}')
        return None
    print('using HY3D mesh:', files[0])
    return files[0]

EXCLUDE_OBJECTS = ['boxtiny', 'boxsmall', 'basketball', 'keyboard', 'toolbox', 'yogaball'] # some behave objects that are exlcuded
BEHAVE_ROOT = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/behave'

HODOME_VIEW_IDS = [19, 26, 27, 34 ]
def get_camera_K_hodome(seq_name, view_id):
    "get the intrinsics for the given view id"
    calib_file = f'/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/HODome/calibration/{seq_name.split("_")[0]}/calibration.json'
    calib_data = json.load(open(calib_file))
    intrinsics = calib_data[f'{view_id-1}']['K']
    intrinsics = np.array(intrinsics).reshape(3, 3) # in original RGB resolution
    intrinsics[:2] /= 3. # scale down to 1/3 resolution: 1280x720, the mask resolution
    return intrinsics # (3x3)


def get_IMHD_camera_K(seq_name, view_id):
    ""
    date = seq_name.split("_")[0]
    extrin = f'/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/IMHD2/calibrations/{date}/extrin.json' # also camera id 1 params
    intrin = f'/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/IMHD2/calibrations/{date}/intrin.json' # this is camera id 1 params
    calib = f'/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/IMHD2/calibrations/{date}/calibration.json'
    calib_data = json.load(open(calib, 'r'))

    K = np.array(calib_data[str(view_id-1)]['K']).reshape((3, 3))
    scale_ratio = 2.
    K[:2] /= scale_ratio
    K[2, 2] = 1
    return K


IMHD_VIEW_IDS = [1,4,15, 29]
ICAP_FOCALs = np.array([[918.457763671875, 918.4373779296875], [915.29962158203125, 915.1966552734375],
                    [912.8626708984375, 912.67633056640625], [909.82025146484375, 909.62469482421875],
                    [920.533447265625, 920.09722900390625], [909.17633056640625, 909.23529052734375]])
ICAP_CENTERs = np.array([[956.9661865234375, 555.944580078125], [956.664306640625, 551.6165771484375],
                        [956.72003173828125, 554.2166748046875], [957.6181640625, 554.60296630859375],
                        [958.4615478515625, 550.42987060546875], [956.14801025390625, 555.01593017578125]])
shapenet_root = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/ShapeNetCore.v2'
objav_root = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/objaverse/obj'

# the start and end frames for BEHAVE seqs with annotations. 
START_END_FRAMES = {'Date03_Sub03_backpack_back': ['t0003.000', 't0047.467'], 'Date03_Sub03_backpack_hand': ['t0003.000', 't0050.133'], 'Date03_Sub03_backpack_hug': ['t0003.000', 't0049.333'], 'Date03_Sub03_boxlarge': ['t0003.000', 't0049.200'], 'Date03_Sub03_boxlong': ['t0003.000', 't0050.233'], 'Date03_Sub03_boxmedium': ['t0003.000', 't0047.667'], 'Date03_Sub03_chairblack_hand': ['t0003.000', 't0050.233'], 'Date03_Sub03_chairblack_lift': ['t0003.000', 't0050.167'], 'Date03_Sub03_chairwood_hand': ['t0003.000', 't0050.300'], 'Date03_Sub03_chairwood_lift': ['t0003.000', 't0050.500'], 'Date03_Sub03_chairwood_sit': ['t0003.000', 't0050.500'], 'Date03_Sub03_monitor_move': ['t0003.000', 't0049.300'], 'Date03_Sub03_plasticcontainer': ['t0003.000', 't0049.600'], 'Date03_Sub03_stool_lift': ['t0003.000', 't0050.200'], 'Date03_Sub03_stool_sit': ['t0003.000', 't0050.200'], 'Date03_Sub03_suitcase_lift': ['t0003.000', 't0046.700'], 'Date03_Sub03_tablesmall_lean': ['t0003.000', 't0047.300'], 'Date03_Sub03_tablesmall_lift': ['t0003.000', 't0047.333'], 'Date03_Sub03_tablesmall_move': ['t0003.000', 't0046.600'], 'Date03_Sub03_tablesquare_lift': ['t0003.000', 't0048.233'], 'Date03_Sub03_tablesquare_move': ['t0003.000', 't0045.767'], 'Date03_Sub03_tablesquare_sit': ['t0003.000', 't0046.100'], 'Date03_Sub03_trashbin': ['t0003.000', 't0046.433'], 'Date03_Sub03_yogamat': ['t0003.000', 't0049.133'], 'Date03_Sub04_backpack_back': ['t0003.000', 't0049.500'], 'Date03_Sub04_backpack_hand': ['t0003.000', 't0050.200'], 'Date03_Sub04_backpack_hug': ['t0003.000', 't0050.200'], 'Date03_Sub04_boxlarge': ['t0003.000', 't0050.400'], 'Date03_Sub04_boxlong': ['t0003.000', 't0050.300'], 'Date03_Sub04_boxmedium': ['t0003.000', 't0050.233'], 'Date03_Sub04_chairblack_hand': ['t0003.000', 't0050.300'], 'Date03_Sub04_chairblack_liftreal': ['t0003.000', 't0050.233'], 'Date03_Sub04_chairblack_sit': ['t0003.000', 't0045.500'], 'Date03_Sub04_chairwood_hand': ['t0003.000', 't0049.033'], 'Date03_Sub04_chairwood_lift': ['t0003.000', 't0050.200'], 'Date03_Sub04_monitor_hand': ['t0003.000', 't0050.233'], 'Date03_Sub04_monitor_move': ['t0003.000', 't0050.233'], 'Date03_Sub04_plasticcontainer_lift': ['t0003.000', 't0050.233'], 'Date03_Sub04_stool_move': ['t0003.000', 't0050.267'], 'Date03_Sub04_suitcase_ground': ['t0003.000', 't0050.200'], 'Date03_Sub04_suitcase_lift': ['t0003.000', 't0050.267'], 'Date03_Sub04_tablesmall_hand': ['t0003.000', 't0050.267'], 'Date03_Sub04_tablesmall_lean': ['t0003.000', 't0050.200'], 'Date03_Sub04_tablesmall_lift': ['t0003.000', 't0050.200'], 'Date03_Sub04_tablesquare_hand': ['t0003.000', 't0050.233'], 'Date03_Sub04_tablesquare_lift': ['t0003.000', 't0050.233'], 'Date03_Sub04_tablesquare_sit': ['t0003.000', 't0050.200'], 'Date03_Sub04_trashbin': ['t0003.000', 't0050.267'], 'Date03_Sub04_yogamat': ['t0003.000', 't0050.233'], 'Date03_Sub05_boxlarge': ['t0003.000', 't0050.233'], 'Date03_Sub05_boxlong': ['t0003.000', 't0050.300'], 'Date03_Sub05_boxmedium': ['t0003.000', 't0050.200'], 'Date03_Sub05_chairblack': ['t0011.000', 't0133.600'], 'Date03_Sub05_chairwood': ['t0003.000', 't0066.967'], 'Date03_Sub05_monitor': ['t0003.000', 't0066.933'], 'Date03_Sub05_plasticcontainer': ['t0003.000', 't0050.200'], 'Date03_Sub05_stool': ['t0003.000', 't0066.833'], 'Date03_Sub05_suitcase': ['t0003.000', 't0066.933'], 'Date03_Sub05_tablesmall': ['t0003.000', 't0133.633'], 'Date03_Sub05_tablesquare': ['t0003.000', 't0133.733'], 'Date03_Sub05_trashbin': ['t0003.000', 't0050.200'], 'Date03_Sub05_yogamat': ['t0003.000', 't0050.300']}
