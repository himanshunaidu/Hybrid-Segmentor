"""
This script processes an image, along with its depth map, ground truth mask and camera pose.
Currently, it uses these inputs to get the 'sidewalk' mask, which would generally be the area in front of the camera.

Procedure:
- Apply the mask to only get the sidewalk patch from the images. 
- Fit a local plane to the sidewalk (would be easier if you have a polygon mesh).
- Warp the sidewalk patch using the ARKit camera transform and the fitted local plane to get a quasi bird’s eye-view. 
    (Try to get a rectangular patch, perhaps by trimming some of the patch. 
    May want to avoid padding because the color difference between the sidewalk patch and the padding 
        may be seen as an additional crack
    )
    (Optionally: Divide the rectangular patch into smaller patches because the ML model uses 256x256 patches)
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
import torch

from read_utils import Frame
from read_utils import get_rgb, get_depth, get_mask, get_depth_confidence, get_intrinsics, get_pose_matrix
from warp_utils import get_3d_points, normalize, warp_image_to_birds_eye_view
from ml_utils import get_model, predict

DATASET_PATH = os.path.join("..", "data", "iOSPointMapper_1_Cityscapes_2")
OUTPUT_PATH = "output"

DATASET_CSV_PATH = os.path.join(DATASET_PATH, "dataset.csv")

IMG_PATH_COLUMN = "rgb_frame_path"
DEPTH_PATH_COLUMN = "depth_frame_path"
MASK_PATH_COLUMN = "annotation_frame_path"
POSE_COLUMNS = ["odometry_x","odometry_y","odometry_z","odometry_qx","odometry_qy","odometry_qz","odometry_qw"]

DEPTH_THRESHOLD = 5000.0 # mm
SIDEWALK_MASK_VALUE = 18
SIDEWALK_DEPTH_IN_THRESHOLD_PROPORTION = 0.1

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df

def load_frame(row: pd.Series) -> Frame:
    color_image = get_rgb(row, DATASET_PATH, rotation_code=cv2.ROTATE_90_COUNTERCLOCKWISE)
    depth_image = get_depth(row, DATASET_PATH, rotation_code=cv2.ROTATE_90_COUNTERCLOCKWISE)
    mask_image = get_mask(row, DATASET_PATH, rotation_code=cv2.ROTATE_90_COUNTERCLOCKWISE)
    intrinsics = get_intrinsics(row)
    pose_matrix = get_pose_matrix(row)
    frame = Frame(color_image, depth_image, mask_image, intrinsics, pose_matrix)
    return frame

def save_3d_points_to_ply(points_3d, colors_3d, filename):
    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_3d / 255.0)  # Normalize colors to [0, 1]

    # Save to PLY file
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved {len(points_3d)} points to {filename}")

def postprocess_contour_mask(contour_mask, depth_image):
    """
    Post-process the contour mask to:
    a) Remove borders through erosion
    b) Remove points with depth > DEPTH_THRESHOLD
    """
    kernel = np.ones((5,5), np.uint8)
    eroded_mask = cv2.erode(contour_mask, kernel, iterations=1)
    depth_mask = (depth_image > 0) & (depth_image < DEPTH_THRESHOLD)
    final_mask = eroded_mask & depth_mask.astype(np.uint8)
    return final_mask    

def divide_image(image, patch_size=(256, 256)):
    """
    Divides the input image into smaller patches of given size.
    For now, it ignores the patch_size if the image size is not perfectly divisible.
    
    Args:
        image: Input image (H, W, 3)
        patch_size: Tuple of (height, width) for each patch
    """
    H, W = image.shape[:2]
    ph, pw = patch_size
    patches = []
    patch_rows = math.ceil(H / ph)
    patch_cols = math.ceil(W / pw)
    actual_ph, actual_pw = H // patch_rows, W // patch_cols

    print(f"Dividing image of size {W}x{H} into {patch_cols}x{patch_rows} patches of size {actual_pw}x{actual_ph}")
    for i in range(patch_rows):
        for j in range(patch_cols):
            y_start = i * actual_ph
            y_end = min((i + 1) * actual_ph, H)
            x_start = j * actual_pw
            x_end = min((j + 1) * actual_pw, W)

            patch = image[y_start:y_end, x_start:x_end]
            patches.append(patch)
    return patches

def process_patch(patch, model) -> tuple[np.ndarray, np.ndarray]:
    patch_actual_size = patch.shape[:2]
    patch_resized = patch
    if patch_actual_size != (256, 256):
        patch_resized = cv2.resize(patch, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    patch_rgb = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2RGB)
    patch_tensor = torch.from_numpy(patch_rgb).float().permute(2, 0, 1) / 255.0
    pred_mask = predict(model, patch_tensor)  # (256, 256)
    pred_mask = cv2.resize(pred_mask.numpy(), (patch_actual_size[1], patch_actual_size[0]), interpolation=cv2.INTER_NEAREST)
    # Remove predictions for alpha = 0 areas
    alpha_channel = patch[:, :, 3] if patch.shape[2] == 4 else np.ones(patch_actual_size, dtype=np.uint8) * 255
    pred_mask = pred_mask * (alpha_channel > 0).astype(np.uint8)
    # Convert pred_mask to a 3-channel image
    pred_mask_np = (pred_mask * 255).astype(np.uint8)
    pred_mask_color = cv2.cvtColor(pred_mask_np, cv2.COLOR_GRAY2BGRA)
    pred_mask_color[:, :, 1] = 0  # Zero out G channel
    pred_mask_color[:, :, 0] = 0  # Zero out B channel
    # Add pred_mask_color on top of the original patch
    processed_patch = cv2.addWeighted(patch.astype(np.uint8), 1.0, pred_mask_color.astype(np.uint8), 1.0, 0)
    return processed_patch, pred_mask_np

def fuse_patches(patches, original_image_size, patch_size):
    if len(patches) == 0:
        return None
    H, W = original_image_size
    actual_ph, actual_pw = patches[0].shape[:2]
    channels = patches[0].shape[2] if len(patches[0].shape) == 3 else 1
    if channels == 1:
        fused_image = np.zeros((H, W), dtype=patches[0].dtype)
    else:
        fused_image = np.zeros((H, W, channels), dtype=patches[0].dtype)
    patch_rows = math.ceil(H / actual_ph)
    patch_cols = math.ceil(W / actual_pw)
    
    for i in range(patch_rows):
        for j in range(patch_cols):
            idx = i * patch_cols + j
            if idx >= len(patches):
                continue
            y_start = i * actual_ph
            y_end = min((i + 1) * actual_ph, H)
            x_start = j * actual_pw
            x_end = min((j + 1) * actual_pw, W)
            fused_image[y_start:y_end, x_start:x_end] = patches[idx]
    return fused_image

def process_frame(frame: Frame, model=None):
    # Extract sidewalk mask
    sidewalk_mask = (frame.mask_image == SIDEWALK_MASK_VALUE).astype(np.uint8)

    # Find contours of the sidewalk mask
    contours, _ = cv2.findContours(sidewalk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    depth_image = cv2.resize(frame.depth_image, (frame.color_image.shape[1], frame.color_image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    
    # Create a mask out of the biggest contour
    for index, contour in enumerate(largest_contours):
        contour_mask = np.zeros_like(frame.mask_image)
        cv2.drawContours(contour_mask, [contour], -1, (255), thickness=cv2.FILLED)

        sidewalk_depth = cv2.bitwise_and(depth_image, depth_image, mask=contour_mask)
        if len(sidewalk_depth) == 0:
            print("No sidewalk depth values found.")
            continue
        
        sidewalk_in_range = sidewalk_depth[(sidewalk_depth > 0) & (sidewalk_depth < DEPTH_THRESHOLD)]
        if len(sidewalk_in_range) / len(sidewalk_depth.flatten()) < SIDEWALK_DEPTH_IN_THRESHOLD_PROPORTION:
            print("Not enough sidewalk depth values in range.")
            continue
        else:
            print(f"Found Main Sidewalk")
            break

    if len(sidewalk_in_range) == 0:
        print("No valid sidewalk depth found in any contour.")
        return None
    print(f"Main Sidewalk Contour to be used: {index} {contour.shape}, Area: {cv2.contourArea(contour)}")
    final_contour_mask = postprocess_contour_mask(contour_mask, depth_image) # For eventual post-processing
    color_masked_image = cv2.bitwise_and(frame.color_image, frame.color_image, mask=contour_mask)
    
    # For the sidewalk patch, get the 3D points using the contour mask, depth image, intrinsics and pose
    # Also get the colors of the 3D points
    points_3d, colors_3d = get_3d_points(frame.color_image, contour_mask, depth_image, frame.intrinsics, frame.pose_matrix)
    save_3d_points_to_ply(points_3d, colors_3d, os.path.join(OUTPUT_PATH, "sidewalk_point_cloud.ply"))

    if len(points_3d) < 50:
        print("Not enough 3D points to fit a plane.")
        return color_masked_image
    # Fit a plane to the 3D points using RANSAC
    plane = pyrsc.Plane()
    best_eq, best_inliers = plane.fit(points_3d, thresh=0.1, maxIteration=1000)
    a, b, c, d = best_eq
    normal_vector = np.array([a, b, c])
    slope = np.degrees(np.arccos(normalize(normal_vector) @ np.array([0,1,0])))
    if slope > 90:
        best_eq = [-a, -b, -c, -d]
        normal_vector = -normal_vector
        slope = 180 - slope
    print(f"Fitted Plane Equation: {best_eq}, Inliers: {len(best_inliers)}, Slope: {slope:.2f} degrees")
    
    # Warp the sidewalk patch to get a bird's eye view
    warped_image, H_bi = warp_image_to_birds_eye_view(color_masked_image, points_3d[best_inliers], frame.intrinsics, frame.pose_matrix, best_eq, s=0.01)
    cv2.imwrite(os.path.join(OUTPUT_PATH, "warped_sidewalk.png"), warped_image)
    
    # Divide the image into 256x256 patches
    patches = divide_image(warped_image, patch_size=(256, 256))
    # for i, patch in enumerate(patches):
    #     cv2.imwrite(os.path.join(OUTPUT_PATH, f"patch_{i:03d}.png"), patch)
    print(f"Divided into {len(patches)} patches.")
    
    # Run each patch through the model to get the crack mask
    if model is None:
        print("No model provided, skipping prediction.")
        return color_masked_image
    print("Running model predictions on patches...")
    processed_patches = []
    patch_pred_masks = []
    for i, patch in enumerate(patches):
        processed_patch, pred_mask = process_patch(patch, model)
        processed_patches.append(processed_patch)
        patch_pred_masks.append(pred_mask)
    
    print("Fusing patches back to full image...")
    fused_image = fuse_patches(processed_patches, warped_image.shape[:2], patch_size=(256, 256))
    fused_pred_mask = fuse_patches(patch_pred_masks, warped_image.shape[:2], patch_size=(256, 256))
    if fused_image is None:
        print("No patches to fuse.")
        return color_masked_image
    
    cv2.imwrite(os.path.join(OUTPUT_PATH, "warped_fused_output.png"), fused_image)
    cv2.imwrite(os.path.join(OUTPUT_PATH, "warped_fused_pred_mask.png"), fused_pred_mask)
    
    # Warp back the fused image to the original view
    H_ib = np.linalg.inv(H_bi)
    W_color, H_color = frame.color_image.shape[1], frame.color_image.shape[0]
    unwarped_image = cv2.warpPerspective(fused_image, H_ib, (W_color, H_color), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    unwarped_image = cv2.bitwise_and(unwarped_image, unwarped_image, mask=final_contour_mask)
    cv2.imwrite(os.path.join(OUTPUT_PATH, "fused_output.png"), unwarped_image)

    unwarped_pred_mask = cv2.warpPerspective(fused_pred_mask, H_ib, (W_color, H_color), flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    unwarped_pred_mask = cv2.bitwise_and(unwarped_pred_mask, unwarped_pred_mask, mask=final_contour_mask)
    cv2.imwrite(os.path.join(OUTPUT_PATH, "fused_pred_mask.png"), unwarped_pred_mask)

    return color_masked_image

if __name__=="__main__":
    df = load_csv(DATASET_CSV_PATH)
    print(f"Loaded {len(df)} rows from {DATASET_CSV_PATH}")
    
    model = get_model()

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        frame = load_frame(row)

        processed_image = process_frame(frame, model=model)
        if processed_image is not None: cv2.imwrite(os.path.join(OUTPUT_PATH, f"output_frame_{idx:04d}.png"), processed_image)

        break