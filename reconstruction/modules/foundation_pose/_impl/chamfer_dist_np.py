
"""
adpated from BPS
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def chamfer_distance(x, y, metric='l2', direction='bi', ret_intermediate=False):
    """Chamfer distance between two point clouds

    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default l2
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||_metric}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||_metric}}

        this is the squared root distance, while pytorch3d is the squared distance
        distance y to x: (N, 1)
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
        if ret_intermediate:
            return chamfer_dist, min_x_to_y
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y) # bidirectional errors are accumulated
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
    if ret_intermediate:
        return chamfer_dist, min_x_to_y, min_y_to_x # return distance for recall and precision

    return chamfer_dist


def compute_fscore(gt, pred, thres=0.01):
    """

    :param gt: (N, 3)
    :param pred: (M, 3)
    :param thres: 0.01m
    :return:
    """
    chamf, d1, d2 = chamfer_distance(gt, pred, ret_intermediate=True)  #

    # scores = np.zeros((3, len(th_list)))
    recall = float(sum(d < thres for d in d2)) / float(len(d2))
    precision = float(sum(d < thres for d in d1)) / float(len(d1))

    if recall + precision > 0:
        fscore = 2 * recall * precision / (recall + precision)
    else:
        fscore = 0
    # print(f'recall: {recall:.3f}, precision: {precision:.3f}, fscore: {fscore:.3f}, chamf: {chamf}')
    return fscore, chamf