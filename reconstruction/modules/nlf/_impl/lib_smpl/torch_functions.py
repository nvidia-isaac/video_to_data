"""
Code modified from Kaolin.
Original code from: LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration, NeurIPS' 20.
Author: Bharat Lal Bhatnagar
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interactions, CVPR'22
"""
import torch


def np2tensor(x, device=None):
    return torch.tensor(x, device=device)


def tensor2np(x):
    return x.detach().cpu().numpy()


def closest_index(src_points: torch.Tensor, tgt_points: torch.Tensor, K=1):
    """
    Given two point clouds, finds closest point id
    :param src_points: B x N x 3
    :param tgt_points: B x M x 3
    :return B x N
    """
    from pytorch3d.ops import knn_points
    closest_index_in_tgt = knn_points(src_points, tgt_points, K=K)
    return closest_index_in_tgt.idx.squeeze(-1)

def closest_dist(src_points: torch.Tensor, tgt_points: torch.Tensor, K=1):
    """
    Given two point clouds, finds closest point id
    :param src_points: B x N x 3
    :param tgt_points: B x M x 3
    :return B x N
    """
    from pytorch3d.ops import knn_points
    closest_index_in_tgt = knn_points(src_points, tgt_points, K=K)
    return closest_index_in_tgt.dists.squeeze(-1)


def batch_gather(arr, ind):
    """
    :param arr: B x N x D
    :param ind: B x M
    :return: B x M x D
    """
    dummy = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), arr.size(2))
    out = torch.gather(arr, 1, dummy)
    return out


def batch_sparse_dense_matmul(S, D):
    """
    Batch sparse-dense matrix multiplication

    :param torch.SparseTensor S: a sparse tensor of size (batch_size, p, q)
    :param torch.Tensor D: a dense tensor of size (batch_size, q, r)
    :return: a dense tensor of size (batch_size, p, r)
    :rtype: torch.Tensor
    """

    num_b = D.shape[0]
    S_shape = S.shape
    if not S.is_coalesced():
        S = S.coalesce()

    indices = S.indices().view(3, num_b, -1)
    values = S.values().view(num_b, -1)
    ret = torch.stack([
        torch.sparse.mm(
            torch.sparse_coo_tensor(indices[1:, i], values[i], S_shape[1:], device=D.device),
            D[i]
        )
        for i in range(num_b)
    ])
    return ret

def batch_sparse_dense_matmul_fast(S, D):
    """
    Fast batch sparse-dense matrix multiplication using baddbmm.

    :param torch.SparseTensor S: a sparse tensor of size (batch_size, p, q)
    :param torch.Tensor D: a dense tensor of size (batch_size, q, r)
    :return: a dense tensor of size (batch_size, p, r)
    :rtype: torch.Tensor

    Converts S to dense for batched matmul.
    """
    if not S.is_coalesced():
        S = S.coalesce()
    B, p, q = S.shape
    _, q2, r = D.shape
    assert q == q2, "Incompatible shapes"
    out = torch.zeros((B, p, r), dtype=D.dtype, device=D.device)
    out = torch.baddbmm(out, S.to_dense(), D)
    return out


def chamfer_distance(s1, s2, w1=1., w2=1.):
    """
    :param s1: B x N x 3
    :param s2: B x M x 3
    :param w1: weight for distance from s1 to s2
    :param w2: weight for distance from s2 to s1
    retun: B, bidirectional distance for each example
    """
    from pytorch3d.ops import knn_points

    assert s1.is_cuda and s2.is_cuda
    closest_dist_in_s2 = knn_points(s1, s2, K=1)
    closest_dist_in_s1 = knn_points(s2, s1, K=1)

    return (closest_dist_in_s2.dists**0.5 * w1).mean(axis=1).squeeze(-1) + (closest_dist_in_s1.dists**0.5 * w2).mean(axis=1).squeeze(-1)

if __name__ == "__main__":
    from psbody.mesh import Mesh
    from pytorch3d.io import load_obj, load_objs_as_meshes
    import numpy as np
    from pytorch3d.structures import Meshes

    pts = np.zeros((1,3)).astype('float32')
    temp = Mesh(filename='/BS/bharat-2/static00/renderings/renderpeople_rigged/rp_eric_rigged_005_zup_a/rp_eric_rigged_005_zup_a_smpl.obj')
    closest_face, closest_points = temp.closest_faces_and_points(pts)
    dist = np.linalg.norm(pts - closest_points, axis=1)

    print('done')

