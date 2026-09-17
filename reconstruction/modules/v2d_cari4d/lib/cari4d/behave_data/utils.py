# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
loads calibrations
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import os, sys

import trimesh
sys.path.append("/")
import json
from os.path import join, basename, dirname, isfile
import numpy as np
import cv2
import os.path as osp
from PIL import Image
from behave_data.kinect_calib import KinectCalib


def rotate_yaxis(R, t):
    "rotate the transformation matrix around z-axis by 180 degree ==>> let y-axis point up"
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    global_trans = np.eye(4)
    global_trans[0, 0] = global_trans[1, 1] = -1  # rotate around z-axis by 180
    rotated = np.matmul(global_trans, transform)
    return rotated[:3, :3], rotated[:3, 3]


def load_intrinsics(intrinsic_folder, kids):
    """
    kids: list of kinect id that should be loaded
    """
    intrinsic_calibs = [json.load(open(join(intrinsic_folder, f"{x}/calibration.json"))) for x in kids]
    pc_tables = [np.load(join(intrinsic_folder, f"{x}/pointcloud_table.npy")) for x in kids]
    kinects = [KinectCalib(cal, pc) for cal, pc in zip(intrinsic_calibs, pc_tables)]

    return kinects


def load_kinect_poses(config_folder, kids):
    pose_calibs = [json.load(open(join(config_folder, f"{x}/config.json"))) for x in kids]
    rotations = [np.array(pose_calibs[x]['rotation']).reshape((3, 3)) for x in kids]
    translations = [np.array(pose_calibs[x]['translation']) for x in kids]
    return rotations, translations


def load_kinects(intrinsic_folder, config_folder, kids):
    intrinsic_calibs = [json.load(open(join(intrinsic_folder, f"{x}/calibration.json"))) for x in kids]
    pc_tables = [np.load(join(intrinsic_folder, f"{x}/pointcloud_table.npy")) for x in kids]
    pose_files = [join(config_folder, f"{x}/config.json") for x in kids]
    kinects = [KinectCalib(cal, pc) for cal, pc in zip(intrinsic_calibs, pc_tables)]
    return kinects


def load_kinect_poses_back(config_folder, kids, rotate=False):
    """
    backward transform
    rotate: kinect y-axis pointing down, if rotate, then return a transform that make y-axis pointing up
    """
    rotations, translations = load_kinect_poses(config_folder, kids)
    rotations_back = []
    translations_back = []
    for r, t in zip(rotations, translations):
        trans = np.eye(4)
        trans[:3, :3] = r
        trans[:3, 3] = t

        trans_back = np.linalg.inv(trans) # now the y-axis point down

        r_back = trans_back[:3, :3]
        t_back = trans_back[:3, 3]
        if rotate:
            r_back, t_back = rotate_yaxis(r_back, t_back)

        rotations_back.append(r_back)
        translations_back.append(t_back)
    return rotations_back, translations_back


def availabe_kindata(input_video, kinect_count=3):
    # all available kinect videos in this folder, return the list of kinect id, and str representation
    fname_split = os.path.basename(input_video).split('.')
    idx = int(fname_split[1])
    kids = []
    comb = ''
    for k in range(kinect_count):
        file = input_video.replace(f'.{idx}.', f'.{k}.')
        if os.path.exists(file):
            kids.append(k)
            comb = comb + str(k)
        else:
            print("Warning: {} does not exist in this folder!".format(file))
    return kids, comb


def save_color_depth(out_dir, color, depth, kid, color_only=False, ext='jpg'):
    color_file = join(out_dir, f'k{kid}.color.{ext}')
    Image.fromarray(color).save(color_file)
    if not color_only:
        depth_file = join(out_dir, f'k{kid}.depth.png')
        cv2.imwrite(depth_file, depth)

# path to the simplified mesh used for registration
_mesh_template = {
    "backpack":"backpack/backpack_f1000.ply",
    'basketball':"basketball/basketball_f1000.ply",
    'boxlarge':"boxlarge/boxlarge_f1000.ply",
    'boxtiny':"boxtiny/boxtiny_f1000.ply",
    'boxlong':"boxlong/boxlong_f1000.ply",
    'boxsmall':"boxsmall/boxsmall_f1000.ply",
    'boxmedium':"boxmedium/boxmedium_f1000.ply",
    'chairblack': "chairblack/chairblack_f2500.ply",
    'chairwood': "chairwood/chairwood_f2500.ply",
    'monitor': "monitor/monitor_closed_f1000.ply",
    'keyboard':"keyboard/keyboard_f1000.ply",
    'plasticcontainer':"plasticcontainer/plasticcontainer_f1000.ply",
    'stool':"stool/stool_f1000.ply",
    'tablesquare':"tablesquare/tablesquare_f2000.ply",
    'toolbox':"toolbox/toolbox_f1000.ply",
    "suitcase":"suitcase/suitcase_f1000.ply",
    'tablesmall':"tablesmall/tablesmall_f1000.ply",
    'yogamat': "yogamat/yogamat_f1000.ply",
    'yogaball':"yogaball/yogaball_f1000.ply",
    'trashbin':"trashbin/trashbin_f1000.ply",
}

# for HODome: lambda function to get template path given obj_name
get_hodome_template_path = lambda obj_name: f"/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/HODome/obj-newtex/{obj_name}/{obj_name}.obj"
_hodome_objs = ['badminton', 'bigsofa', 'box', 'chair', 'flower', 'monitor', 'pillow', 'pink', 'table', 'tennis', 'trolleycase', 'baseball', 'book', 'case', 'desk', 'keyboard', 'pan', 'pingpong', 'smallsofa', 'tabletall', 'talltable', 'trashcan']
# IHMD: bat  broom  chair  dumbbell  kettlebell  pan  skateboard  suitcase  tennis
_imhd_objs = ['bat', 'broom', 'chair', 'dumbbell', 'kettlebell', 'pan', 'skateboard', 'suitcase', 'tennis']


_icap_template = {
    'obj01': '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/InterCap/objects/01.ply', #'suitcase'
    'obj02': '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/InterCap/objects/02.ply', #  'skateboard'
    'obj03': '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/InterCap/objects/03.ply', # 'football'
    'obj04': '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/InterCap/objects/04.ply', # 'umbrella'
    'obj05': '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/InterCap/objects/05.ply', # 'tennis-racket'
    'obj06': '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/InterCap/objects/06.ply', # toolbox
    'obj07': '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/InterCap/objects/07.ply', #  chair01
    'obj08': '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/InterCap/objects/08.ply', #'bottle'
    'obj09': '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/InterCap/objects/09.ply', # 'cup'
    'obj10': '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/InterCap/objects/10.ply', # 'chair02', stool
}

# path to original full-reso scan reconstructions
orig_scan = {
        "backpack": "/BS/xxie2020/work/objects/backpack/backpack_closed.ply",
        'basketball': "/BS/xxie2020/work/objects/basketball/basketball_closed.ply",
        'boxlarge': "/BS/xxie2020/work/objects/box_large/box_large_closed_centered.ply",
        'boxtiny': "/BS/xxie2020/work/objects/box_tiny/box_tiny_closed.ply",
        'boxlong': "/BS/xxie2020/work/objects/box_long/box_long_close.ply",
        'boxsmall': "/BS/xxie2020/work/objects/box_small/boxsmall_closed.ply",
        'boxmedium': "/BS/xxie2020/work/objects/box_medium/box_medium_closed.ply",
        'chairblack': "/BS/xxie2020/work/objects/chair_black/chair_black.ply",
        'chairwood': "/BS/xxie2020/work/objects/chair_wood/chair_wood_clean.ply",
        'monitor': "/BS/xxie2020/work/objects/monitor/monitor_closed_centered.ply",
        'keyboard': "/BS/xxie2020/work/objects/keyboard/keyboard_closed_centered.ply",
        'plasticcontainer': "/BS/xxie2020/work/objects/plastic_container/container_closed_centered.ply",
        "stool":"/BS/xxie2020/work/objects/stool/stool_clean_centered.ply",
        'tablesquare': "/BS/xxie2020/work/objects/table_square/table_square_closed.ply",
        'toolbox': "/BS/xxie2020/work/objects/toolbox/toolbox_closed_centered.ply",
        "suitcase": "/BS/xxie2020/work/objects/suitcase_small/suitcase_closed_centered.ply",
        'tablesmall': "/BS/xxie2020/work/objects/table_small/table_small_closed_aligned.ply",
        'yogamat': "/BS/xxie2020/work/objects/yoga_mat/yogamat_closed.ply",
        'yogaball': "/BS/xxie2020/work/objects/yoga_ball/yoga_ball_closed.ply",
        'trashbin': "/BS/xxie2020/work/objects/trash_bin/trashbin_closed.ply"
    }


def get_template_path(behave_path, obj_name, obj=False):
    if obj_name in _mesh_template.keys():
        path = osp.join(behave_path, 'objects', _mesh_template[obj_name])
    elif obj_name in _icap_template.keys():
        path = _icap_template[obj_name]
        if obj:
            path = path.replace('.ply', '.obj')
    elif obj_name in _hodome_objs:
        path = get_hodome_template_path(obj_name)
    else:
        raise ValueError(f'{obj_name} not found in template paths!')
    if not osp.isfile(path):
        raise ValueError(f'{path} does not exist, please make sure you have downloaded the dataset and placed in {behave_path}')
    return path

def get_render_template_path_from_seq(seq_name):
    "a method that works for all datasets, the path for rendering"
    date = seq_name.split('_')[0]
    if '2022' in date:
        return get_hodome_template_path(seq_name.split('_')[2])
    elif 'ICap' in seq_name:
        return _icap_template[seq_name.split('_')[2]]
    elif '_Subxx' in seq_name:
        return json.load(open('splits/procigen-video-obj-path.json', 'r'))[seq_name]
    elif 'Date' in date:
        obj_name = seq_name.split('_')[2]
        dataset_path = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/behave'
        return join(dataset_path, 'objects', f'{obj_name}/{obj_name}.obj')
    elif '2023' in date:
        obj_name = seq_name.split('_')[2]
        mesh_file = f'/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/IMHD2/hy3d-texgen-simp/{obj_name}/{obj_name}_simplified_transformed.obj'
        return mesh_file
    else:
        raise ValueError(f'{seq_name} not found in template paths!')

def load_scan_centered(scan_path, cent=True):
    """load a scan and centered it around origin"""
    import trimesh
    mesh = trimesh.load(scan_path, process=False)
    if cent:
        center = np.mean(mesh.vertices, 0)
        mesh.vertices = mesh.vertices - center
    return mesh


def load_template(obj_name, cent=True, dataset_path=None):
    assert dataset_path is not None, 'please specify BEHAVE dataset path!'
    temp_path = get_template_path(dataset_path, obj_name)
    return load_scan_centered(temp_path, cent)

def load_templates_all(dataset_path=None, orig=True, aligned=False):
    import trimesh 
    out = {}
    for obj in list(_mesh_template.keys()):
        if orig:
            temp = load_template_orig(obj, dataset_path)
            temp_simp = load_template(obj, cent=False, dataset_path=dataset_path)
            cent = np.mean(temp_simp.vertices, 0)
            temp.vertices = temp.vertices - cent
        else:
            if aligned:
                if obj in _mesh_template.keys():
                    path = f'{dataset_path}/objects/{obj}_aligned.ply'
                else:
                    path = _icap_template[obj]
                    print(f"Warning: no alignment for InterCap object {obj}!")
                temp = trimesh.load(path, process=False)
            else:
                temp = load_template(obj, cent=True, dataset_path=dataset_path) # simplified template
        out[obj] = temp
    # now also load IMHD and HODome templates 
    # for obj in _hodome_objs:
    #     temp_file = get_hodome_template_path(obj)
    #     temp = trimesh.load(temp_file, process=False)
    #     temp.vertices = temp.vertices - np.mean(temp.vertices, 0)
    #     out['hodome+' + obj] = temp
    # for obj in _imhd_objs:
    #     temp_file = f'/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/IMHD2/hy3d-texgen-simp/{obj}/{obj}_simplified_transformed.obj'
    #     temp = trimesh.load(temp_file, process=False)
    #     temp.vertices = temp.vertices - np.mean(temp.vertices, 0)
    #     out['imhd+' + obj] = temp
    
    return out

def load_template_orig(obj_name, dataset_path=None):
    import trimesh
    path = join(dataset_path, 'objects', f'{obj_name}/{obj_name}.obj')
    return trimesh.load(path, process=False, maintain_order=True)

def write_pointcloud(filename,xyz_points,rgb_points=None):
    """
    updated on March 22, use trimesh for writing
    """
    import trimesh
    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'
    outfolder = dirname(filename)
    os.makedirs(outfolder, exist_ok=True)
    pc = trimesh.points.PointCloud(xyz_points, rgb_points)
    pc.export(filename)


def init_video_controllers(args, input_color, kids):
    from behave_data.video_reader import VideoController, ColorDepthController
    video_prefix = basename(input_color).split('.')[0]
    video_folder = dirname(input_color)
    if args.nodepth:
        controllers = [VideoController(os.path.join(video_folder, f'{video_prefix}.{k}.color.mp4')) for k in kids]
    else:
        controllers = [ColorDepthController(os.path.join(video_folder, video_prefix), k) for k in kids]
    return controllers, video_prefix


def load_hodome_object_template(root_path, seq_name):
    object_name = seq_name.split('_')[-1]
    object_template_path = join(root_path, 'scaned_object', object_name, f'{object_name}_face1000.obj')
    object_mesh = trimesh.load(object_template_path, process=False)
    object_mesh_vertices = object_mesh.vertices
    return object_mesh, object_mesh_vertices


def get_intrinsics_unified(data_source, seq_name, kid, wild_video=False):
    if data_source == 'behave':
        if not wild_video:
            if kid == 0:
                fx, fy = 976.212, 976.047
                cx, cy = 1017.958, 787.313
            elif kid == 1:
                fx, fy = 979.784, 979.840  # for original kinect coordinate system
                cx, cy = 1018.952, 779.486
            elif kid == 2:
                fx, fy = 974.899, 974.337
                cx, cy = 1018.747, 786.176
            else:
                fx, fy = 972.873, 972.790
                cx, cy = 1022.0565, 770.397
        else:
            raise ValueError(f'Invalid wild video: {wild_video}')
        # Run inference. TODO: check if we can speed this up by batching
        intrinsics = np.array([[fx, 0, cx],
                            [0., fy, cy],
                            [0, 0., 1]])
    elif data_source == 'hodome':
        from behave_data.const import get_camera_K_hodome, HODOME_VIEW_IDS
        intrinsics = get_camera_K_hodome(seq_name, HODOME_VIEW_IDS[kid])
    elif data_source == 'intercap':
        from behave_data.const import ICAP_CENTERs
        from behave_data.const import ICAP_FOCALs
        intrinsics = ICAP_FOCALs[kid]
        intrinsics = np.array([[intrinsics[0], 0, ICAP_CENTERs[kid][0]],
                            [0, intrinsics[1], ICAP_CENTERs[kid][1]],
                            [0, 0, 1]])
    elif data_source == 'imhd':
        from behave_data.const import IMHD_VIEW_IDS
        from behave_data.const import get_IMHD_camera_K
        intrinsics = get_IMHD_camera_K(seq_name, IMHD_VIEW_IDS[kid])
    elif data_source == 'procigen':
        K = np.array([[979.784, 0, 1018.952],
                    [0, 979.840, 779.486],
                    [0, 0, 1]])
        K[:2] /= 2. 
        intrinsics = K 
    else:
        raise ValueError(f'Invalid data source: {data_source}')

    return intrinsics