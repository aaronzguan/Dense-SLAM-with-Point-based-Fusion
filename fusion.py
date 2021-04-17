"""
    Initially written by Ming Hsiao in MATLAB
    Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
    Author: Aaron Guan (zhongg@andrew.cmu.edu)
"""
import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import quaternion  # pip install numpy-quaternion

import transforms
import o3d_utility

from preprocess import load_gt_poses


class Map:
    def __init__(self):
        """
        Maintain the points with their properties in the fused Map/World coordinate system
        """
        self.points = np.empty((0, 3))
        self.normals = np.empty((0, 3))
        self.colors = np.empty((0, 3))
        self.weights = np.empty((0, 1))
        self.times = np.empty((0, 1))
        self.initialized = False

    def merge(self, indices, points, normals, colors, R, t, time):
        """
        The merge operation updates existing points in the Map by calculating a weight average.
        The input point cloud need to be transformed into the Map coordinates using transformation
        Then the input transformed vertex and normals, and colors are merged with the matched points
        in the Map with a weight average. For the input points, their weights are just 1.
        Normalization of normals is required after weight average.
        :param indices: Indices of matched points in current Map, (N,). Used for IN PLACE modification.
        :param points: Input associated points, (N, 3)
        :param normals: Input associated normals, (N, 3)
        :param colors: Input associated colors, (N, 3)
        :param R: rotation from camera (input) to world (map), (3, 3)
        :param t: translation from camera (input) to world (map), (3, )
        :param time: Time step of input cloud points (1)
        :return None, update map properties IN PLACE
        """
        # Define lambda function for the weight average
        weight_average = lambda current, weight, input: (current*weight + input) / (weight + 1)
        # Transform input points and normals to world/map coordinates
        T_input_points = (R @ points.T + t).T
        T_input_normals = (R @ normals.T).T

        # Merge the transformed points and normals into the Map with weighted average
        self.points[indices, :] = weight_average(self.points[indices, :], self.weights[indices], T_input_points)
        merged_normals = weight_average(self.normals[indices, :], self.weights[indices], T_input_normals)
        self.normals[indices, :] = merged_normals / np.linalg.norm(merged_normals, axis=1, keepdims=True)

        # Merge the input color into the existing color in Map without geometry transformation
        self.colors[indices, :] = weight_average(self.colors[indices, :], self.weights[indices], colors)

        # Update the weight for the matched points
        self.weights[indices] += 1
        # Update the time for the matched points
        self.times[indices] = time

    def add(self, points, normals, colors, R, t, time):
        """
        Add the new points with properties into the existing Map
        Need to transform the new points and their normals to the Map/World coordinate
        For the weights, we just assume the weights for new input points are 1
        :param points: Input associated points, (N, 3)
        :param normals: Input associated normals, (N, 3)
        :param colors: Input associated colors, (N, 3)
        :param R: rotation from camera (input) to world (map), (3, 3)
        :param t: translation from camera (input) to world (map), (3, )
        :return None, update map properties by concatenation
        """
        T_input_points = (R @ points.T + t).T
        T_input_normals = (R @ normals.T).T
        self.points = np.vstack((self.points, T_input_points))
        self.normals = np.vstack((self.normals, T_input_normals))
        self.colors = np.vstack((self.colors, colors))
        self.weights = np.vstack((self.weights, np.ones((len(points), 1))))
        self.times = np.vstack((self.times, np.ones((len(points), 1)) * time))

    def remove_outliers(self, time, weight_thre, time_thre):
        """
        Remove the outliers that remain in the unstable state for a long time
        :param time:
        :param weight_thre:
        :param time_thre:
        :return:
        """
        indices = np.arange(len(self.points)).astype(int)
        mask = (self.weights < weight_thre) & ((time - self.times) > time_thre)
        indices = indices[~mask.flatten()]
        self.points = self.points[indices, :]
        self.normals = self.normals[indices, :]
        self.weights = self.weights[indices, :]
        self.colors = self.colors[indices]
        self.times = self.times[indices]

    def filter_pass1(self, us, vs, ds, h, w):
        """
        Get the mask indicating valid correspondences within the boundary of image map
        :param us: Putative corresponding u coordinates on an image, (N, 1)
        :param vs: Putative corresponding v coordinates on an image, (N, 1)
        :param ds: Putative corresponding d depth on an image, (N, 1)
        :param h: Height of the image projected to
        :param w: Width of the image projected to
        :return mask: (N, 1) in bool indicating the valid coordinates
        """
        mask = (us >= 0) & (us < w) & (vs >= 0) & (vs < h) & (ds > 0)
        return mask

    def filter_pass2(self, points, normals, input_points, input_normals,
                     dist_diff, angle_diff):
        """
        Get the mask indicating valid correspondences by applying distance and angle threshold
        :param points: Maintained associated points, (M, 3)
        :param normals: Maintained associated normals, (M, 3)
        :param input_points: Input associated points, (M, 3)
        :param input_normals: Input associated normals, (M, 3)
        :param dist_diff: Distance difference threshold to filter correspondences by positions
        :param angle_diff: Angle difference threshold in radians to filter correspondences by normals
        :return mask (N, 1) in bool indicating the valid correspondences
        """
        dist_mask = np.linalg.norm((input_points - points) * input_normals, axis=1) < dist_diff
        unit_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        unit_input_normals = input_normals / np.linalg.norm(input_normals, axis=1, keepdims=True)
        cosang = np.sum(unit_normals * unit_input_normals, axis=1)  # Dot product of each normal, which is cos(angle)
        angle_mask = np.arccos(cosang) < angle_diff
        mask = dist_mask & angle_mask
        return mask

    def fuse(self,
             vertex_map,
             normal_map,
             color_map,
             intrinsic,
             T,
             time,
             dist_diff=0.03,
             angle_diff=np.deg2rad(5),
             weight_thre=8,
             time_thre=40):
        """
        Given an input point clouds, we project the existing points in global map onto the 2D image map of
        the input point clouds (vertex_map) and find the corresponding points.
        The valid corresponding points in the input point clouds are merged with the 3D points in global map.
        We assume the ground truth transformations from camera to world are known. Actually this transformation
        need to be calculated by ICP
        The unassociated points will be added to the existing Map as well.
        Points that remain in the unstable state for a long time are likely outliers or artifacts from
        moving objects and will be removed after t_max time steps.
        :param vertex_map: Input vertex map, (H, W, 3)
        :param normal_map: Input normal map, (H, W, 3)
        :param color_map: Input color map, (H, W, 3)
        :param intrinsic: Camera intrinsic matrix, (3, 3)
        :param T: transformation from camera (input) to world (map), (4, 4)
        :param dist_diff: Distance threshold to filter correspondences
        :param angle_diff: Angle threshold of normals to filter correspondences
        :param weight_thre: Weight threshold to remove unstable points with weights less than the threshold
        :param time_thre: Time threshold to remove unstable points after a fixed time steps
        :return: None, update map properties on demand
        """
        # Camera to world
        R = T[:3, :3]
        t = T[:3, 3:]

        # World to camera
        T_inv = np.linalg.inv(T)
        R_inv = T_inv[:3, :3]
        t_inv = T_inv[:3, 3:]

        if not self.initialized:
            # Initialize the Map with the first input cloud points
            points = vertex_map.reshape((-1, 3))
            normals = normal_map.reshape((-1, 3))
            colors = color_map.reshape((-1, 3))

            # add first input cloud points into the Map
            self.add(points, normals, colors, R, t, time)
            self.initialized = True

        else:
            h, w, _ = vertex_map.shape

            # Transform from world to camera for projective association
            indices = np.arange(len(self.points)).astype(int)
            T_points = (R_inv @ self.points.T + t_inv).T
            R_normals = (R_inv @ self.normals.T).T

            # Projective association
            us, vs, ds = transforms.project(T_points, intrinsic)
            us = np.round(us).astype(int)
            vs = np.round(vs).astype(int)

            # First filter: valid projection within boundary
            mask = self.filter_pass1(us, vs, ds, h, w)

            indices = indices[mask]
            us = us[mask]
            vs = vs[mask]

            T_points = T_points[indices]
            R_normals = R_normals[indices]
            valid_points = vertex_map[vs, us]
            valid_normals = normal_map[vs, us]

            # second filter: apply distance and angle thresholds on the correspondences
            mask = self.filter_pass2(T_points, R_normals, valid_points, valid_normals, dist_diff, angle_diff)

            indices = indices[mask]
            us = us[mask]
            vs = vs[mask]

            updated_entries = len(indices)

            merged_points = vertex_map[vs, us]
            merged_normals = normal_map[vs, us]
            merged_colors = color_map[vs, us]

            # Merge step - compute weight average after transformation
            self.merge(indices, merged_points, merged_normals, merged_colors, R, t, time)

            associated_mask = np.zeros((h, w)).astype(bool)
            associated_mask[vs, us] = True
            new_points = vertex_map[~associated_mask]
            new_normals = normal_map[~associated_mask]
            new_colors = color_map[~associated_mask]

            # Add the unassociated points into existing Map
            self.add(new_points, new_normals, new_colors, R, t, time)
            added_entries = len(new_points)

            # Remove the points that remain in unstable state for a long time
            total_entries = len(self.points)
            self.remove_outliers(time, weight_thre, time_thre)
            removed_entries = total_entries - len(self.points)

            print('updated: {}, added: {}, removed: {}, total: {}'.format(updated_entries, added_entries,
                                                                          removed_entries, len(self.points)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path', type=str, help='path to the dataset folder containing rgb/ and depth/',
        default='./dataset/')
    parser.add_argument('--start_idx',
                        type=int,
                        help='index to the source depth/normal maps',
                        default=1)
    parser.add_argument('--end_idx',
                        type=int,
                        help='index to the source depth/normal maps',
                        default=200)
    parser.add_argument('--downsample_factor', type=int, default=2)
    args = parser.parse_args()

    intrinsic_struct = o3d.io.read_pinhole_camera_intrinsic('intrinsics.json')
    intrinsic = np.array(intrinsic_struct.intrinsic_matrix)
    indices, gt_poses = load_gt_poses(os.path.join(args.path, 'livingRoom2.gt.freiburg'))
    # TUM convention
    depth_scale = 5000.0

    rgb_path = os.path.join(args.path, 'rgb')
    depth_path = os.path.join(args.path, 'depth')
    normal_path = os.path.join(args.path, 'normal')

    m = Map()

    down_factor = args.downsample_factor
    intrinsic /= down_factor
    intrinsic[2, 2] = 1

    for i in range(args.start_idx, args.end_idx + 1):
        print('Fusing frame {:03d}'.format(i))
        source_depth = o3d.io.read_image('{}/{}.png'.format(depth_path, i))
        source_depth = np.asarray(source_depth) / depth_scale
        source_depth = source_depth[::down_factor, ::down_factor]
        source_vertex_map = transforms.unproject(source_depth, intrinsic)

        source_color_map = np.asarray(o3d.io.read_image('{}/{}.png'.format(rgb_path, i))).astype(float) / 255.0
        source_color_map = source_color_map[::down_factor, ::down_factor]

        source_normal_map = np.load('{}/{}.npy'.format(normal_path, i))
        source_normal_map = source_normal_map[::down_factor, ::down_factor]

        m.fuse(source_vertex_map, source_normal_map, source_color_map, intrinsic, gt_poses[i], time=i)

    global_pcd = o3d_utility.make_point_cloud(m.points, colors=m.colors, normals=m.normals)
    o3d.visualization.draw_geometries([global_pcd.transform(o3d_utility.flip_transform)])
