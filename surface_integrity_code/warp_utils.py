"""
This script contains utility functions to warp images to a bird's eye view.
"""
import os
import cv2
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import pyransac3d as pyrsc
import open3d as o3d
import math

def get_3d_points(color_image, mask, depth_image, intrinsics, pose_matrix):
    # Get the 2D points from the mask
    y_indices, x_indices = np.where(mask > 0)
    z_values = depth_image[y_indices, x_indices] / 1000.0  # Convert mm to meters

    # Convert to 3D points
    image_points = np.vstack((x_indices, y_indices, np.ones_like(x_indices)))  # Shape: (3, N)
    intrinsic_matrix_inv = np.linalg.inv(intrinsics)
    rays = intrinsic_matrix_inv @ image_points  # Shape: (3, N)
    
    camera_points = rays * z_values  # Shape: (3, N)
    # camera_points[1] = -camera_points[1]  # Invert Y axis to match coordinate system
    # camera_points[2] = -camera_points[2]  # Invert Z axis to match coordinate system
    
    # Convert to homogeneous coordinates
    camera_points_homogeneous = np.vstack((camera_points, np.ones((1, camera_points.shape[1]))))  # Shape: (4, N)
    
    # Transform to world coordinates
    world_points_homogeneous = pose_matrix @ camera_points_homogeneous  # Shape: (4, N)
    points_3d_transformed = world_points_homogeneous[:3].T  # Shape: (N, 3)
    
    # Get the colors of the 3D points
    colors_3d = color_image[y_indices, x_indices]  # Shape: (N, 4) ## alpha
    colors_3d = colors_3d[:, :3]  # Discard alpha channel
    print(f"Total Points: {points_3d_transformed.shape[0]}")

    return points_3d_transformed, colors_3d

def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def invert_transform(T_wc):
    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    T_cw = np.eye(4)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = t_cw
    return T_cw

def plane_basis(normal: np.ndarray):
    """
    Given a normal vector, compute two orthogonal basis vectors on the plane defined by the normal.
    """
    a = np.array([1.0, 0.0, 0.0])
    if np.allclose(normal, a):
        a = np.array([0.0, 1.0, 0.0])
    u_w: np.ndarray = normalize(a - np.dot(a, normal) * normal)
    v_w: np.ndarray = normalize(np.cross(normal, u_w))
    return u_w, v_w

def homography_plane_to_image(intrinsics, T_wc, origin, normal):
    T_cw = invert_transform(T_wc)
    R_cw = T_cw[:3,:3]
    t_cw = T_cw[:3, 3:4]  # (3,1)
    
    # orthonormal basis on plane
    n_w = normalize(normal)
    u_w, v_w = plane_basis(n_w)

    # Homography from plane to camera
    col1 = R_cw @ u_w.reshape(3,1)
    col2 = R_cw @ v_w.reshape(3,1)
    col3 = (R_cw @ origin.reshape(3, 1)) + t_cw
    H_cp = np.hstack((col1, col2, col3))  # (3,3) # plane to camera
    
    # Homography from plane to image
    H_ip = intrinsics @ H_cp  # (3,3) # plane to image
    return H_ip, u_w, v_w

def _warp_image(image, intrinsics, T_wc, origin, normal, W_m, H_m, s=0.01, *,
                cx_p=None, cy_p=None, s_x=None, s_y=None):
    """
    Warps the input image to a bird's eye view of the plane defined by (origin, normal).
    
    Naming convention:
    M_ab: Homography from b to a
    
    Args:
        image: Input image (H, W, 3)
        intrinsics: Camera intrinsics matrix (3, 3)
        T_wc: Camera-to-world transformation matrix (4, 4)
        origin: A point on the plane in world coordinates (3,)
        normal: Normal vector of the plane in world coordinates (3,)
        W_m: Width of the output image in meters
        H_m: Height of the output image in meters
        s: Scale factor to convert meters to pixels
    """
    # Compute homography from plane to image
    H_ip, u_w, v_w = homography_plane_to_image(intrinsics, T_wc, origin, normal)
    
    W_px = int(W_m / s)
    H_px = int(H_m / s)
    
    # We want plane coords for each destination pixel:
    # Let's map pixel (i,j) to plane coords:
    #   U = (i - cx_p) * s, V = (j - cy_p) * s
    # Center the patch around p0 by default:
    if s_x is None: s_x = s
    if s_y is None: s_y = s
    if cx_p is None or cy_p is None:
        cx_p, cy_p = (W_px / 2 * s_x), (H_px / 2 * s_y)

    # Create a 3x3 homography matrix to map bird's eye view pixel coords to plane coords
    H_pb = np.array([
        [s_x, 0, cx_p],
        [0, s_y, cy_p],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Combined homography from bird's eye view to image
    H_ib = H_ip @ H_pb  # (3,3)
    H_bi = np.linalg.inv(H_ib)  # (3,3)
    
    warped_image = cv2.warpPerspective(image, H_bi, (W_px, H_px), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return warped_image, (u_w, v_w), (W_px, H_px), H_bi

def warp_image_to_birds_eye_view(image, points, intrinsics, T_wc, plane_eq, s=0.01):
    """
    Warps the input image to a bird's eye view of the plane defined by the plane equation.
    
    Args:
        image: Input image (H, W, 3)
        points: 3D points on the plane (N, 3)
        intrinsics: Camera intrinsics matrix (3, 3)
        T_wc: Camera-to-world transformation matrix (4, 4)
        plane_eq: Plane equation coefficients (a, b, c, d) for ax + by + cz + d = 0
        s: Scale factor to convert meters to pixels
    """
    a, b, c, d = plane_eq
    normal = np.array([a, b, c])
    
    # Compute centroid of the points to use as origin
    origin = np.mean(points, axis=0)
    
    # Compute bounding box of the points in the plane's local coordinate system
    u_w, v_w = plane_basis(normal)
    local_coords = np.dot(points - origin, np.vstack((u_w, v_w)).T)  # (N, 2)
    min_u, min_v = np.min(local_coords, axis=0)
    max_u, max_v = np.max(local_coords, axis=0)
    
    W_m = max_u - min_u
    H_m = max_v - min_v
    print(f"Sidewalk Patch Size: {W_m:.2f}m x {H_m:.2f}m")
    
    # Add some padding
    padding = 0#0.01 * max(W_m, H_m)
    W_m += 2 * padding
    H_m += 2 * padding
    
    warped_image, (u_w, v_w), (W_px, H_px), H_bi = _warp_image(
        image, intrinsics, T_wc, origin, normal, W_m, H_m, s,
        s_x=s, s_y=-s, cx_p=min_u-padding, cy_p=max_v+padding
    )
    
    return warped_image, H_bi