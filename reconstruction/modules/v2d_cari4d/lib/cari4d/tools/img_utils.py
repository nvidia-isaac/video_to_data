# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
common functions for image operations
"""

import cv2
import numpy as np

def crop(img, center, crop_size):
    """
    crop image around the given center, pad zeros for borders
    :param img:
    :param center: np array
    :param crop_size: np array or a float size of the resulting crop
    :return: a square crop around the center
    """
    assert isinstance(img, np.ndarray)
    h, w = img.shape[:2]
    topleft = np.round(center - crop_size / 2).astype(int)
    bottom_right = np.round(center + crop_size / 2).astype(int)

    x1 = max(0, topleft[0])
    y1 = max(0, topleft[1])
    x2 = min(w - 1, bottom_right[0])
    y2 = min(h - 1, bottom_right[1])
    cropped = img[y1:y2, x1:x2]

    p1 = max(0, -topleft[0])  # padding in x, top
    p2 = max(0, -topleft[1])  # padding in y, top
    p3 = max(0, bottom_right[0] - w + 1)  # padding in x, bottom
    p4 = max(0, bottom_right[1] - h + 1)  # padding in y, bottom

    dim = len(img.shape)
    if dim == 3:
        padded = np.pad(cropped, [[p2, p4], [p1, p3], [0, 0]])
    elif dim == 2:
        padded = np.pad(cropped, [[p2, p4], [p1, p3]])
    else:
        raise NotImplemented
    return padded


def resize(img, img_size, mode=cv2.INTER_LINEAR):
    """
    resize image to the input
    :param img:
    :param img_size: (width, height) of the target image size
    :param mode:
    :return:
    """
    h, w = img.shape[:2]
    load_ratio = 1.0 * w / h
    netin_ratio = 1.0 * img_size[0] / img_size[1]
    assert load_ratio == netin_ratio, "image aspect ration not matching, given image: {}, net input: {}".format(
        img.shape, img_size)
    resized = cv2.resize(img, img_size, interpolation=mode)
    return resized


def masks2bbox(masks, threshold=127):
    """

    :param masks:
    :param threshold:
    :return: bounding box corner coordinate
    """
    mask_comb = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        mask_comb = mask_comb | (m > threshold)

    yid, xid = np.where(mask_comb)
    bmin = np.array([xid.min(), yid.min()])
    bmax = np.array([xid.max(), yid.max()])
    return bmin, bmax
