import os
import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R
from PIL import Image

DATASET_CSV_COLUMNS = [
    'frame_index',
    'original_path',
    'rgb_frame_path',
    'depth_frame_path',
    'depth_confidence_frame_path',
    # intrinsics matrix
    'intrinsics_00', 'intrinsics_01', 'intrinsics_02',
    'intrinsics_10', 'intrinsics_11', 'intrinsics_12',
    'intrinsics_20', 'intrinsics_21', 'intrinsics_22',
    # odometry data
    'odometry_timestamp',
    'odometry_x', 'odometry_y', 'odometry_z',
    'odometry_qx', 'odometry_qy', 'odometry_qz', 'odometry_qw',
    # imu data
    'imu_timestamp',
    'a_x', 'a_y', 'a_z',
    'alpha_x', 'alpha_y', 'alpha_z',
    # location data
    'location_timestamp',
    'latitude', 'longitude', 'altitude',
    'horizontal_accuracy', 'vertical_accuracy', 
    'speed', 'course', 'floor_level',
    # heading data
    'heading_timestamp',
    'magnetic_heading', 'true_heading', 'heading_accuracy'
]

def get_rgb(data: pd.Series, dataset_path: str, *, 
            width: int = -1, height: int = -1, rotation_code: int = -1) -> np.ndarray:
    """Extracts the RGB image from a DataFrame row."""
    rgb_path = os.path.join(dataset_path, data['rgb_frame_path'])
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB file not found: {rgb_path}")
    rgb_image = Image.open(rgb_path).convert('RGBA')  # Ensure the image has an alpha channel
    rgb_image = np.array(rgb_image)
    if width == -1 or height == -1:
        height, width = rgb_image.shape[:2]
    if rotation_code != -1:
        rgb_image = cv2.rotate(rgb_image, rotation_code)
    rgb_image = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return rgb_image

def get_depth(data: pd.Series, dataset_path: str, *,
              width: int, height: int, rotation_code: int = -1) -> np.ndarray:
    """Extracts the depth image from a DataFrame row."""
    depth_path = os.path.join(dataset_path, data['depth_frame_path'])
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    depth_image = Image.open(depth_path).convert('F')  # Convert to float format
    depth_image = np.array(depth_image)
    if width == -1 or height == -1:
        height, width = depth_image.shape[:2]
    if rotation_code != -1:
        depth_image = cv2.rotate(depth_image, rotation_code)
    depth_image = cv2.resize(depth_image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return depth_image

def get_depth_confidence(data: pd.Series, dataset_path: str, *,
                         width: int, height: int, rotation_code: int = -1) -> np.ndarray:
    """Extracts the depth confidence image from a DataFrame row."""
    conf_path = os.path.join(dataset_path, data['depth_confidence_frame_path'])
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"Depth confidence file not found: {conf_path}")
    conf_image = Image.open(conf_path).convert('L')  # Convert to grayscale
    conf_image = np.array(conf_image)
    if width == -1 or height == -1:
        height, width = conf_image.shape[:2]
    if rotation_code != -1:
        conf_image = cv2.rotate(conf_image, rotation_code)
    conf_image = cv2.resize(conf_image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return conf_image

def get_intrinsics(data: pd.Series, scale_x = 1.0, scale_y = 1.0) -> np.ndarray:
    """Extracts the camera intrinsics matrix from a DataFrame row."""
    intrinsics = np.array([
        [data['intrinsics_00'], data['intrinsics_01'], data['intrinsics_02']],
        [data['intrinsics_10'], data['intrinsics_11'], data['intrinsics_12']],
        [data['intrinsics_20'], data['intrinsics_21'], data['intrinsics_22']]
    ])
    # if scale_x != 1.0 or scale_y != 1.0:
    #     intrinsics = _resize_camera_matrix(intrinsics, scale_x, scale_y)
    # print(f"{yaw_from_K(intrinsics, IMG_SIZE[1], IMG_SIZE[0])} degrees yaw, pitch")
    return intrinsics

def get_pose_matrix(data: pd.Series) -> np.ndarray:
    """Constructs the pose matrix from odometry data."""
    translation = np.array([data['odometry_x'], data['odometry_y'], data['odometry_z']])
    rotation = R.from_quat([
        data['odometry_qx'], data['odometry_qy'], data['odometry_qz'], data['odometry_qw']
    ]).as_matrix()

    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation
    pose_matrix[:3, 3] = translation
    pose_matrix = np.linalg.inv(pose_matrix)
    return pose_matrix