"""
    Initially written by Ming Hsiao in MATLAB
    Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
    Author: Aaron Guan (zhongg@andrew.cmu.edu)
"""

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import scipy.linalg

import argparse
import transforms
import o3d_utility


def find_projective_correspondence(source_points,
                                   source_normals,
                                   target_vertex_map,
                                   target_normal_map,
                                   intrinsic,
                                   T_init,
                                   dist_diff=0.07):
    """
    Find the correspondences of source points p from the target point cloud q for point-to-plane ICP
    It projects the source points p onto the 2D image map (vertex_map) of target points q and
    get the corresponding target points q from the vertex_map.
    In such case, p and q are not strictly nearest neighbors in 3D, but projective nearest neighbors.
    They are both in image coordinates but have a depth value.
    :param source_points: Source point cloud locations, (N, 3)
    :param source_normals: Source point cloud normals, (N, 3)
    :param target_vertex_map: Target vertex map, (H, W, 3)
    :param target_normal_map: Target normal map, (H, W, 3)
    :param intrinsic: Camera intrinsic matrix, (3, 3)
    :param T_init: Initial transformation matrix from source to target, (4, 4)
    :param dist_diff: Distance difference threshold to filter correspondences
    :return source_indices: indices of points in the source point cloud with a valid projective
            correspondence in the target map, (M, 1)
    :return target_us: associated u coordinate of points in the target map, (M, 1)
    :return target_vs: associated v coordinate of points in the target map, (M, 1)
    """
    h, w, _ = target_vertex_map.shape

    R = T_init[:3, :3]
    t = T_init[:3, 3:]

    # Transform source points from the source coordinate system to the target coordinate system
    T_source_points = (R @ source_points.T + t).T

    # Set up initial correspondences from source to target
    source_indices = np.arange(len(source_points)).astype(int)
    target_us, target_vs, target_ds = transforms.project(T_source_points, intrinsic)
    target_us = np.round(target_us).astype(int)
    target_vs = np.round(target_vs).astype(int)

    # First filter: boundary check to make sure all projected points remain within the image map
    mask = (target_us >= 0) & (target_us < w) & (target_vs >= 0) & (target_vs < h) & (target_ds > 0)
    source_indices = source_indices[mask]
    target_us = target_us[mask]
    target_vs = target_vs[mask]
    T_source_points = T_source_points[mask]

    # Second filter: apply distance threshold to get the correspondences
    target_points = target_vertex_map[target_vs, target_us]
    target_normals = target_normal_map[target_vs, target_us]
    mask = np.linalg.norm((target_points - T_source_points) * target_normals, axis=1) < dist_diff

    source_indices = source_indices[mask]
    target_us = target_us[mask]
    target_vs = target_vs[mask]

    return source_indices, target_us, target_vs


def build_linear_system(source_points, target_points, target_normals, T):
    """
    Build the approximated linear system for the point-to-plane ICP with small angle approximations.
    Reference: https://www.cs.unc.edu/techreports/04-004.pdf
    :param source_points: Source point cloud locations, (N, 3)
    :param target_points: Target point cloud locations, (N, 3)
    :param target_normals: Normals of target points, (N, 3)
    :param T: 4x4 Transformation matrix from source to target
    :return: A, b matrix for |Ax - b| linear least-square problem
    """
    M = len(source_points)
    assert len(target_points) == M and len(target_normals) == M

    R = T[:3, :3]
    t = T[:3, 3:]

    p_prime = (R @ source_points.T + t).T
    q = target_points
    n_q = target_normals

    A = np.zeros((M, 6))
    b = np.zeros((M, ))

    # Build the linear system
    A[:, :3] = np.cross(p_prime, n_q)
    A[:, 3:] = n_q
    b[:] = np.sum(n_q * (q - p_prime), axis=1)

    return A, b


def pose2transformation(delta):
    """
    Convert the 6D incremental transformation update (alpha, beta, gamma, tx, ty, tz)
    into 4x4 rigid-body transformation matrix
    Reference: https://en.wikipedia.org/wiki/Euler_angles in the ZYX order
    :param delta: Vector (6, ) in the tangent space with the small angle assumption.
    :return: T Matrix (4, 4) transformation matrix recovered from delta
    """
    w = delta[:3]
    u = np.expand_dims(delta[3:], axis=1)

    T = np.eye(4)

    # yapf: disable
    R = np.array([[np.cos(w[2]) * np.cos(w[1]),
                   -np.sin(w[2]) * np.cos(w[0]) + np.cos(w[2]) * np.sin(w[1]) * np.sin(w[0]),
                   np.sin(w[2]) * np.sin(w[0]) + np.cos(w[2]) * np.sin(w[1]) * np.cos(w[1])],
                  [np.sin(w[2]) * np.cos(w[1]),
                   np.cos(w[2]) * np.cos(w[0]) + np.sin(w[2]) * np.sin(w[1]) * np.sin(w[0]),
                   -np.cos(w[2]) * np.sin(w[0]) + np.sin(w[2]) * np.sin(w[1]) * np.cos(w[0])],
                  [-np.sin(w[1]),
                   np.cos(w[1]) * np.sin(w[0]),
                   np.cos(w[1]) * np.cos(w[0])
                   ]])
    # yapf: enable

    T[:3, :3] = R
    T[:3, 3:] = u

    return T


def solve(A, b, solver='qr'):
    """
    Solve the |Ax - b| least-square problem using Pseudo-Inverse/QR factorization/LU factorization
    :param A: (N, 6) matrix A in |Ax - b| least-square problem
    :param b: (N, 1) matrix b in |Ax - b| least-square problem
    :param solver: Type of the least-square problem solver
    :return: (6, ) vector x by solving the linear system. You may directly use dense solvers from numpy.
    """
    x = np.zeros((A.shape[1],))
    if solver == 'pinv':
        x = np.linalg.inv(A.T @ A) @ A.T @ b
    elif solver == 'qr':
        Q, R = np.linalg.qr(A)  # QR decomposition with qr function
        y = np.dot(Q.T, b)      # Let y = Q^T.b using matrix multiplication
        x = np.linalg.solve(R, y)   # Solve Rx = y
    elif solver == 'lu':
        c, low = scipy.linalg.cho_factor(A.T @ A)   # Cholesky factorization for A.T.A
        x = scipy.linalg.cho_solve((c, low), A.T @ b)   # Solve  cx = A.T.b
    return x


def icp(source_points,
        source_normals,
        target_vertex_map,
        target_normal_map,
        intrinsic,
        T_init=np.eye(4),
        debug_association=False,
        iters=10):
    """
    Point-to-Plane ICP to find the transformation between source cloud points and target cloud points.
    It starts with a transformation initialization and then find the correspondence points by projecting the
    source points onto the 2D image plane of target points.
    It approximate the non-linear system as a linear system due to the small-angle approximation and solves
    the |Ax-b| linear least-square problem to get the incremental transformation update x.
    It iteratively calculates the transformation increment and update the transformation until convergence
    :param source_points: Source point cloud locations, (N, 3)
    :param source_normals: Source point cloud normals, (N, 3)
    :param target_vertex_map: Target vertex map, (H, W, 3)
    :param target_normal_map: Target normal map, (H, W, 3)
    :param intrinsic: Intrinsic matrix, (3, 3)
    :param T_init: Initial transformation from source to target, (4, 4)
    :param debug_association: Visualize association between sources and targets for debug
    :param iters: Number of iterations for ICP
    :return: Final ICP result, T (4, 4) transformation matrix from source to target
    """
    T = T_init

    for i in range(iters):
        # Find the correspondences by Projective Data Association (PDA)
        source_indices, target_us, target_vs = find_projective_correspondence(source_points, source_normals,
                                                                              target_vertex_map, target_normal_map,
                                                                              intrinsic, T)

        # Select associated source and target points
        corres_source_points = source_points[source_indices]
        corres_target_points = target_vertex_map[target_vs, target_us]
        corres_target_normals = target_normal_map[target_vs, target_us]

        # Debug, if necessary
        if debug_association:
            o3d_utility.visualize_correspondences(corres_source_points, corres_target_points, T)

        # Solve the approximated linear least-square system to get the incremental transformation
        A, b = build_linear_system(corres_source_points, corres_target_points, corres_target_normals, T)
        delta = solve(A, b)

        # Update and output
        T = pose2transformation(delta) @ T
        loss = np.mean(b**2)
        print('iter {}: avg loss = {:.4e}, inlier count = {}'.format(
            i, loss, len(corres_source_points)))

    return T


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', help='path to the dataset folder containing rgb/ and depth/')
    parser.add_argument('--source_idx',
                        type=int,
                        help='index to the source depth/normal maps',
                        default=10)
    parser.add_argument('--target_idx',
                        type=int,
                        help='index to the source depth/normal maps',
                        default=50)
    args = parser.parse_args()

    intrinsic_struct = o3d.io.read_pinhole_camera_intrinsic('intrinsics.json')
    intrinsic = np.array(intrinsic_struct.intrinsic_matrix)

    depth_path = os.path.join(args.path, 'depth')
    normal_path = os.path.join(args.path, 'normal')

    # TUM convention -- uint16 value to float meters
    depth_scale = 5000.0

    # Source: load depth and rescale to meters
    source_depth = o3d.io.read_image('{}/{}.png'.format(depth_path, args.source_idx))
    source_depth = np.asarray(source_depth) / depth_scale

    # Unproject depth to vertex map (H, W, 3) and reshape to a point cloud (H*W, 3)
    source_vertex_map = transforms.unproject(source_depth, intrinsic)
    source_points = source_vertex_map.reshape((-1, 3))

    # Load normal map (H, W, 3) and reshape to point cloud normals (H*W, 3)
    source_normal_map = np.load('{}/{}.npy'.format(normal_path, args.source_idx))
    source_normals = source_normal_map.reshape((-1, 3))

    # Similar preparation for target, but keep the image format for projective association
    target_depth = o3d.io.read_image('{}/{}.png'.format(depth_path, args.target_idx))
    target_depth = np.asarray(target_depth) / depth_scale
    target_vertex_map = transforms.unproject(target_depth, intrinsic)
    target_normal_map = np.load('{}/{}.npy'.format(normal_path, args.target_idx))

    # Visualize before ICP
    o3d_utility.visualize_icp(source_points, target_vertex_map.reshape((-1, 3)), np.eye(4))

    # Perform point-to-plane ICP to get the transformation from source point cloud to target point cloud
    T = icp(source_points,
            source_normals,
            target_vertex_map,
            target_normal_map,
            intrinsic,
            np.eye(4),
            debug_association=False)

    # Visualize after ICP
    o3d_utility.visualize_icp(source_points, target_vertex_map.reshape((-1, 3)), T)
