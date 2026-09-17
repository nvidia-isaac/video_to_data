"""
Author: Xianghui
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""

import numpy as np
import cv2
from sklearn.neighbors import KDTree
from scipy.interpolate import RectBivariateSpline
from scipy import interpolate


class KinectCalib:
    def __init__(self, calibration, pc_table):
        """
        the localize file is the transformation matrix from this kinect RGB to kinect zero RGB
        """
        self.pc_table_ext = np.dstack([pc_table, np.ones(pc_table.shape[:2] + (1,), dtype=pc_table.dtype)])
        color2depth_R = np.array(calibration['color_to_depth']['rotation']).reshape(3, 3)
        color2depth_t = np.array(calibration['color_to_depth']['translation'])
        depth2color_R = np.array(calibration['depth_to_color']['rotation']).reshape(3, 3)
        depth2color_t = np.array(calibration['depth_to_color']['translation'])

        self.color2depth_R = color2depth_R
        self.color2depth_t = color2depth_t
        self.depth2color_R = depth2color_R
        self.depth2color_t = depth2color_t

        color_calib = calibration['color']
        self.image_size = (color_calib['width'], color_calib['height'])
        self.focal_dist = (color_calib['fx'], color_calib['fy'])
        self.center = (color_calib['cx'], color_calib['cy'])
        self.calibration_matrix = np.eye(3)
        self.calibration_matrix[0, 0], self.calibration_matrix[1, 1] = self.focal_dist
        self.calibration_matrix[:2, 2] = self.center
        self.dist_coeffs = np.array(color_calib['opencv'][4:])

        # depth intrinsic
        depth_calib = calibration['depth']
        self.depth_size = (depth_calib['width'], depth_calib['height'])
        self.depth_center = (depth_calib['cx'], depth_calib['cy'])
        self.depth_focal = (depth_calib['fx'], depth_calib['fy'])
        self.depth_matrix = np.eye(3)
        self.depth_matrix[0,0], self.depth_matrix[1,1] = self.depth_focal
        self.depth_matrix[:2,2] = self.depth_center
        self.depth_distcoeffs = np.array(depth_calib['opencv'][4:])

        # additional parameters for distortion
        if 'codx' in color_calib and 'codx' in depth_calib:
            self.depth_codx = depth_calib['codx']
            self.depth_cody = depth_calib['cody']
            self.depth_metric_radius = depth_calib['metric_radius']
            self.color_codx = color_calib['codx']
            self.color_cody = color_calib['cody']
            self.color_metric_radius = color_calib['metric_radius']
        else:
            # for backward compatibility
            self.depth_codx = 0
            self.depth_cody = 0
            self.depth_metric_radius = np.nan
            self.color_codx = 0
            self.color_cody = 0
            self.color_metric_radius = np.nan

    def undistort(self, img):
        return cv2.undistort(img, self.calibration_matrix, self.dist_coeffs)

    def project_points(self, points):
        """
        given points in the color camera coordinate, project it into color image
        return: (N, 2)
        """
        return cv2.projectPoints(points[..., np.newaxis],
                                 np.zeros(3), np.zeros(3), self.calibration_matrix, self.dist_coeffs)[0].reshape(-1, 2)

    def dmap2pc(self, depth, return_mask=False):
        """
        use precomputed table to convert depth map to point cloud
        """
        nanmask = depth == 0
        d = depth.copy().astype(np.float) / 1000.
        d[nanmask] = np.nan
        pc = self.pc_table_ext * d[..., np.newaxis]
        validmask = np.isfinite(pc[:, :, 0])
        pc = pc[validmask]
        if return_mask:
            return pc, validmask
        return pc

    def interpolate_depth(self, depth_im):
        "borrowed from PROX"
        # fill depth holes to avoid black spots in aligned rgb image
        zero_mask = np.array(depth_im == 0.).ravel()
        depth_im_flat = depth_im.ravel()
        depth_im_flat[zero_mask] = np.interp(np.flatnonzero(zero_mask), np.flatnonzero(~zero_mask),
                                             depth_im_flat[~zero_mask])
        depth_im = depth_im_flat.reshape(depth_im.shape)
        return depth_im

    def pc2color(self, pointcloud):
        """
        given point cloud, return its pixel coordinate in RGB image
        """
        # project the point cloud in depth camera to RGB camera
        pointcloud_color = np.matmul(pointcloud, self.depth2color_R.T) + self.depth2color_t
        projected_color_pc = self.project_points(pointcloud_color)
        return projected_color_pc

    def pc2color_valid(self, pointcloud):
        """
        given point cloud in depth camera,
        return its pixel coordinate in RGB image, invalid pixels (out of range are removed)
        """
        # project the point cloud in depth camera to RGB camera
        pointcloud_color = np.matmul(pointcloud, self.depth2color_R.T) + self.depth2color_t
        projected_color_pc = self.project_points(pointcloud_color)
        mask = self.valid_pixmask(projected_color_pc)
        return projected_color_pc[mask, :], pointcloud[mask, :]

    def valid_pixmask(self, color_pixels):
        w, h = self.image_size
        valid_x = np.logical_and(color_pixels[:,0]<w, color_pixels[:,0]>=0)
        valid_y = np.logical_and(color_pixels[:, 1] < h, color_pixels[:, 1] >= 0)
        valid = np.logical_and(valid_y, valid_x)
        return valid

    def color_to_pc(self, colorpts, pc_depth, projected_color_pc=None, k=4, std=1.):
        """
        project point clouds to RGB image, use KDTree to query each color point's closest point cloud
        """
        def weight_func(x, std=1.):
            return np.exp(-x / (2 * std ** 2))

        if projected_color_pc is None:
            projected_color_pc = self.pc2color(pc_depth)
        tree = KDTree(projected_color_pc)
        dists, inds = tree.query(colorpts, k=k) # return the closest distance of each colorpts to the tree
        weights = weight_func(dists, std=std)
        weights_sum = weights.sum(axis=1)
        w = weights / weights_sum[:, np.newaxis]
        pts_world = (pc_depth[inds.flatten(), :].reshape(-1, k, 3) * w[:, :, np.newaxis]).sum(axis=1)
        return pts_world

    def get_pc_colors(self, pointcloud, color_frame, projected_color_pc=None):
        """
        given point cloud and color frame, return the colors for the point cloud
        """
        if projected_color_pc is None:
            projected_color_pc = self.pc2color(pointcloud)
        pc_colors = np.ones_like(pointcloud)
        for i in range(3):
            # the project pixel coordinate in color frame is non-integer, interpolate the color mesh to get best result
            spline = RectBivariateSpline(np.arange(color_frame.shape[0]), np.arange(color_frame.shape[1]),
                                         color_frame[:, :, i])

            pc_colors[:, i] = spline(projected_color_pc[:, 1], projected_color_pc[:, 0], grid=False)
        pc_colors /= 255.
        pc_colors = np.clip(pc_colors, 0, 1)
        return pc_colors

    def pc2dmap(self, points):
        """
        reproject pc to image plane, find closes grid for each one
        """
        p2d = self.project_points(points)
        cw, ch = self.image_size
        px, py = np.meshgrid(np.linspace(0, cw-1, cw), np.linspace(0, ch-1, ch))
        # does not work due to memory limitation
        depth = interpolate.griddata(p2d, points[:, 2], (px, py), method='nearest')
        dmap = np.zeros((ch, cw))
        dmap[py.astype(int), px.astype(int)] = depth
        return dmap

    def dmap2colorpc(self, color, depth):
        "convert depth in color camera to pc"
        pc, mask = self.dmap2pc(depth, return_mask=True)
        pc_colors = color[mask]
        return pc, pc_colors






