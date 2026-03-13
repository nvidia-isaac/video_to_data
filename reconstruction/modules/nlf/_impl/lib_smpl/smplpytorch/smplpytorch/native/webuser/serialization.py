"""
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license
More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de
About this file:
================
This file defines the serialization functions of the SMPL model.
Modules included:
- save_model:
  saves the SMPL model to a given file location as a .pkl file
- load_model:
  loads the SMPL model from a given file location (i.e. a .pkl file location),
  or a dictionary object.

This file is copied from https://github.com/Rubikplayer/flame-fitting
"""
import pickle

import chumpy as ch
import numpy as np
from chumpy.ch import MatVecMult

from .posemapper import posemap
from .verts import verts_core


def backwards_compatibility_replacements(dd):
    # replacements
    if 'default_v' in dd:
        dd['v_template'] = dd['default_v']
        del dd['default_v']
    if 'template_v' in dd:
        dd['v_template'] = dd['template_v']
        del dd['template_v']
    if 'joint_regressor' in dd:
        dd['J_regressor'] = dd['joint_regressor']
        del dd['joint_regressor']
    if 'blendshapes' in dd:
        dd['posedirs'] = dd['blendshapes']
        del dd['blendshapes']
    if 'J' not in dd:
        dd['J'] = dd['joints']
        del dd['joints']

    # defaults
    if 'bs_style' not in dd:
        dd['bs_style'] = 'lbs'


def ready_arguments(fname_or_dict):
    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict, 'rb'), encoding='latin-1')
    else:
        dd = fname_or_dict

    backwards_compatibility_replacements(dd)

    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1] * 3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas']) + dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:, 0])
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:, 1])
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:, 2])
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
    else:
        dd['v_posed'] = dd['v_template'] + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))

    return dd


def load_model(fname_or_dict):
    dd = ready_arguments(fname_or_dict)

    args = {
        'pose': dd['pose'],
        'v': dd['v_posed'],
        'J': dd['J'],
        'weights': dd['weights'],
        'kintree_table': dd['kintree_table'],
        'xp': ch,
        'want_Jtr': True,
        'bs_style': dd['bs_style']
    }

    result, Jtr = verts_core(**args)
    result = result + dd['trans'].reshape((1, 3))
    result.J_transformed = Jtr + dd['trans'].reshape((1, 3))

    for k, v in dd.items():
        setattr(result, k, v)

    return result
