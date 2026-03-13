/*
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <Eigen/Dense>

namespace py = pybind11;

// Helper to convert numpy array to vector of Matrix4f
std::vector<Eigen::Matrix4f> numpy_to_matrix4f_vector(py::array_t<float> arr) {
    auto buf = arr.request();
    if (buf.ndim != 3 || buf.shape[1] != 4 || buf.shape[2] != 4) {
        throw std::runtime_error("Input must be (N, 4, 4) array");
    }
    
    int n = buf.shape[0];
    float* ptr = static_cast<float*>(buf.ptr);
    std::vector<Eigen::Matrix4f> result;
    result.reserve(n);
    
    for (int i = 0; i < n; i++) {
        Eigen::Matrix4f mat;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                mat(r, c) = ptr[i * 16 + r * 4 + c];
            }
        }
        result.push_back(mat);
    }
    return result;
}

// Helper to convert vector of Matrix4f to numpy array
py::array_t<float> matrix4f_vector_to_numpy(const std::vector<Eigen::Matrix4f>& matrices) {
    int n = matrices.size();
    auto result = py::array_t<float>({n, 4, 4});
    auto buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);
    
    for (int i = 0; i < n; i++) {
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                ptr[i * 16 + r * 4 + c] = matrices[i](r, c);
            }
        }
    }
    return result;
}

//@angle_diff: unit is degree
//@dist_diff: unit is meter
std::vector<Eigen::Matrix4f> cluster_poses_impl(
    float angle_diff, 
    float dist_diff, 
    const std::vector<Eigen::Matrix4f>& poses_in,
    const std::vector<Eigen::Matrix4f>& symmetry_tfs)
{
    printf("symmetry_tfs: %d\n", (int)symmetry_tfs.size());
    printf("num original candidates = %d\n", (int)poses_in.size());
    
    if (poses_in.empty()) {
        printf("ERROR: poses_in is empty!\n");
        return std::vector<Eigen::Matrix4f>();
    }
    
    std::vector<Eigen::Matrix4f> poses_out;
    poses_out.push_back(poses_in[0]);
    
    const float radian_thres = angle_diff / 180.0f * M_PI;
    
    for (size_t i = 1; i < poses_in.size(); i++) {
        bool isnew = true;
        Eigen::Matrix4f cur_pose = poses_in[i];
        
        for (const auto& cluster : poses_out) {
            Eigen::Vector3f t0 = cluster.block<3,1>(0,3);
            Eigen::Vector3f t1 = cur_pose.block<3,1>(0,3);
            
            if ((t0 - t1).norm() >= dist_diff) {
                continue;
            }
            
            for (const auto& tf : symmetry_tfs) {
                Eigen::Matrix4f cur_pose_tmp = cur_pose * tf;
                float rot_diff = Utils::rotationGeodesicDistance(
                    cur_pose_tmp.block<3,3>(0,0), 
                    cluster.block<3,3>(0,0)
                );
                
                if (rot_diff < radian_thres) {
                    isnew = false;
                    break;
                }
            }
            
            if (!isnew) break;
        }
        
        if (isnew) {
            poses_out.push_back(poses_in[i]);
        }
    }
    
    printf("num of pose after clustering: %d\n", (int)poses_out.size());
    return poses_out;
}

// Python wrapper
py::array_t<float> cluster_poses(
    float angle_diff,
    float dist_diff,
    py::array_t<float> poses_in_np,
    py::array_t<float> symmetry_tfs_np)
{
    // Manual conversion from numpy to Eigen
    auto poses_in = numpy_to_matrix4f_vector(poses_in_np);
    auto symmetry_tfs = numpy_to_matrix4f_vector(symmetry_tfs_np);
    
    // Call implementation
    auto result = cluster_poses_impl(angle_diff, dist_diff, poses_in, symmetry_tfs);
    
    // Convert back to numpy
    return matrix4f_vector_to_numpy(result);
}

PYBIND11_MODULE(mycpp, m) {
    m.def("cluster_poses", &cluster_poses,
          py::arg("angle_diff"),
          py::arg("dist_diff"),
          py::arg("poses_in"),
          py::arg("symmetry_tfs"),
          "Cluster poses based on angle and distance thresholds");
}