# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# This script plots TUM poses in 3D or 2D (on a fitted plane).
# Usage:
# python scripts/visual/metrics/plot_tum_file.py
#   <tum_file1> <tum_file2> ..
#   [--tags tag1 tag2 .. ]
#   [--plot_x_dir, adding a line indicating X direction on each sample]
#   [--plot_by_ith_time, sampling poses using timestamps from the ith TUM file]
#   [--connect, adding lines between same-timestamped samples]
#   [--plot_2d_plane, fit a plane to all poses and plot in 2D on that plane]
#   [--interactive, show interactive plot instead of saving to file]
#   [--output_file, specify output file path (default: <input_file>_plot.png)]
#
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from read_tum_file import read_tum_file


def fit_plane_to_points(points):
    """
    Fit a plane to 3D points using PCA.
    Returns the plane normal and center point.
    """
    # Center the points
    center = np.mean(points, axis=0)
    centered_points = points - center

    # Compute PCA
    _, _, vh = np.linalg.svd(centered_points)

    # The normal is the last component (smallest eigenvalue)
    normal = vh[-1]

    return normal, center


def project_points_to_plane(points, normal, center):
    """
    Project 3D points onto a plane defined by normal and center.
    Returns 2D coordinates in the plane's coordinate system.
    """
    # Create an orthonormal basis for the plane
    # First, find two vectors orthogonal to the normal
    if abs(normal[0]) < 0.9:
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u = u / np.linalg.norm(u)

    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    # Project points onto the plane
    centered_points = points - center

    # Get 2D coordinates in the plane's coordinate system
    u_coords = np.dot(centered_points, u)
    v_coords = np.dot(centered_points, v)

    return u_coords, v_coords, u, v


def project_vectors_to_plane(vectors, u, v):
    """
    Project 3D vectors onto a plane using the plane's basis vectors u and v.
    Returns 2D vectors in the plane's coordinate system.
    """
    u_coords = np.dot(vectors, u)
    v_coords = np.dot(vectors, v)

    return u_coords, v_coords


def plot_trajectory(args):

    fig = plt.figure(figsize=(20, 15))

    # Determine if we're plotting in 2D plane mode
    if args.plot_2d_plane:
        # Collect all positions to fit plane
        all_positions = []
        for file_path in args.input_files:
            pose_data = read_tum_file(file_path)
            positions = np.column_stack(
                [pose_data['xs'], pose_data['ys'], pose_data['zs']])
            all_positions.append(positions)
        all_positions = np.vstack(all_positions)

        # Fit plane to all positions
        normal, center = fit_plane_to_points(all_positions)
        print(f"Fitted plane normal: {normal}")
        print(f"Plane center: {center}")

        # Pre-compute plane basis vectors for consistent projection
        _, _, u, v = project_points_to_plane(all_positions[:1], normal, center)

        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection='3d')

    # Use a color map
    colormap = plt.cm.hsv  # choose any colormap

    timestamps_to_plot = []
    xs_to_plot = []
    ys_to_plot = []
    zs_to_plot = []

    if args.plot_by_ith_time >= 0:
        pose = read_tum_file(
            args.input_files[args.plot_by_ith_time],
            return_rotation_matrix=True)
        timestamps_to_plot = pose['timestamp_seconds']
        xs_to_plot = pose['xs']
        ys_to_plot = pose['ys']
        zs_to_plot = pose['zs']

    for i in range(len(args.input_files)):
        pose_i = read_tum_file(
            args.input_files[i], return_rotation_matrix=True)

        timestamps = pose_i['timestamp_seconds']
        tag_i = args.tags[i] if args.tags and len(args.tags) == len(
            args.input_files) else ''
        color_i = colormap(i / len(args.input_files))

        ref_xs = []
        ref_ys = []
        ref_zs = []

        if args.plot_by_ith_time >= 0:
            pick_indices = []
            for j in range(len(timestamps)):
                for k in range(len(timestamps_to_plot)):
                    # time diff less than 0.001 seconds
                    if abs(timestamps[j] - timestamps_to_plot[k]) < 1e-3:
                        pick_indices.append(j)
                        ref_xs.append(xs_to_plot[k])
                        ref_ys.append(ys_to_plot[k])
                        ref_zs.append(zs_to_plot[k])
                        break

            update_pose_i = {
                'timestamp_seconds': pose_i['timestamp_seconds'][pick_indices],
                'xs': pose_i['xs'][pick_indices],
                'ys': pose_i['ys'][pick_indices],
                'zs': pose_i['zs'][pick_indices],
                'rotation_matrices': pose_i['rotation_matrices'][pick_indices]
            }

            pose_i = update_pose_i

        if args.plot_2d_plane:
            # Project poses onto the fitted plane
            positions = np.column_stack(
                [pose_i['xs'], pose_i['ys'], pose_i['zs']])
            u_coords, v_coords, _, _ = project_points_to_plane(
                positions, normal, center)

            ax.plot(u_coords, v_coords, '.-', color=color_i, label=tag_i)
        else:
            ax.plot(
                pose_i['xs'],
                pose_i['ys'],
                pose_i['zs'],
                '.-',
                color=color_i,
                label=tag_i)

        if args.connect:
            if args.plot_2d_plane:
                # Project reference positions to plane
                ref_positions = np.column_stack([ref_xs, ref_ys, ref_zs])
                ref_u_coords, ref_v_coords, _, _ = project_points_to_plane(
                    ref_positions, normal, center)

                for j in range(len(ref_u_coords)):
                    ax.plot(
                        [ref_u_coords[j], u_coords[j]],
                        [ref_v_coords[j], v_coords[j]], '--')
            else:
                for j in range(len(ref_xs)):
                    ax.plot(
                        [ref_xs[j], pose_i['xs'][j]],
                        [ref_ys[j], pose_i['ys'][j]],
                        [ref_zs[j], pose_i['zs'][j]], '--')

        # plot arrow for forward direction: x
        vec_x = np.array([0.3, 0, 0])
        orientations = np.array(
            [m.dot(vec_x)
             for m in pose_i['rotation_matrices']]).reshape(-1, 3)
        dxs = orientations[:, 0]
        dys = orientations[:, 1]
        dzs = orientations[:, 2]

        if args.plot_x_dir:
            if args.plot_2d_plane:
                # Project orientation vectors to plane
                orientation_vectors = np.column_stack([dxs, dys, dzs])
                proj_dxs, proj_dys = project_vectors_to_plane(
                    orientation_vectors, u, v)

                for pu, pv, pdu, pdv in zip(u_coords, v_coords, proj_dxs,
                                            proj_dys):
                    ax.plot([pu, pu + pdu], [pv, pv + pdv], color=color_i)
            else:
                for x, y, z, dx, dy, dz in zip(pose_i['xs'], pose_i['ys'],
                                               pose_i['zs'], dxs, dys, dzs):
                    ax.plot(
                        [x, x + dx], [y, y + dy], [z, z + dz], color=color_i)

    if args.plot_2d_plane:
        # Set equal aspect ratio for 2D plot
        ax.set_aspect('equal')
        ax.set_xlabel('Plane U axis')
        ax.set_ylabel('Plane V axis')
        ax.grid(True)
    else:
        # Set equal aspect ratio for 3D plot
        # Get the range of each axis
        max_range = np.array(
            [
                np.max(
                    [np.max(read_tum_file(f)['xs'])
                     for f in args.input_files]) -
                np.min(
                    [np.min(read_tum_file(f)['xs'])
                     for f in args.input_files]),
                np.max(
                    [np.max(read_tum_file(f)['ys'])
                     for f in args.input_files]) -
                np.min(
                    [np.min(read_tum_file(f)['ys'])
                     for f in args.input_files]),
                np.max(
                    [np.max(read_tum_file(f)['zs'])
                     for f in args.input_files]) -
                np.min(
                    [np.min(read_tum_file(f)['zs']) for f in args.input_files])
            ]).max() / 2.0

        # Get the center of the data
        mid_x = (
            np.max([np.max(read_tum_file(f)['xs'])
                    for f in args.input_files]) +
            np.min([np.min(read_tum_file(f)['xs'])
                    for f in args.input_files])) * 0.5
        mid_y = (
            np.max([np.max(read_tum_file(f)['ys'])
                    for f in args.input_files]) +
            np.min([np.min(read_tum_file(f)['ys'])
                    for f in args.input_files])) * 0.5
        mid_z = (
            np.max([np.max(read_tum_file(f)['zs'])
                    for f in args.input_files]) +
            np.min([np.min(read_tum_file(f)['zs'])
                    for f in args.input_files])) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()

    if args.interactive:
        # Show interactive plot
        plt.show()
    elif args.output_file:
        plt.savefig(args.output_file)
        print('Save fig to', args.output_file)
    else:
        # Generate default output filename based on first input file
        first_input_file = args.input_files[0]
        base_path = os.path.splitext(first_input_file)[0]
        default_output_file = base_path + '_plot.png'
        plt.savefig(default_output_file)
        print('Save fig to', default_output_file)


def main(args):
    parser = argparse.ArgumentParser(description="Plot pose errors")

    parser.add_argument(
        "input_files", nargs='+', help="The path to the json file.")
    parser.add_argument(
        "--tags", nargs='+', required=False, help="The tags for each plot.")
    parser.add_argument('--plot_x_dir', action=argparse.BooleanOptionalAction)
    parser.add_argument('--plot_by_ith_time', type=int, default=-1)
    parser.add_argument('--connect', action=argparse.BooleanOptionalAction)
    parser.add_argument(
        '--plot_2d_plane',
        action=argparse.BooleanOptionalAction,
        help='Fit a plane to all poses and plot in 2D on that plane')
    parser.add_argument(
        '--output_file', required=False, help='Whether to save plot file')
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Show interactive plot instead of saving to file')

    args = parser.parse_args()

    plot_trajectory(args)


if __name__ == '__main__':
    main(sys.argv[1:])
