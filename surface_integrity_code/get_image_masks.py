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

DATASET_PATH = os.path.join("../", "data/", "iOSPointMapper_1_Cityscapes_2")

DATASET_CSV_PATH = os.path.join(DATASET_PATH, "dataset.csv")

IMG_PATH_COLUMN = "rgb_frame_path"
DEPTH_PATH_COLUMN = "depth_frame_path"
MASK_PATH_COLUMN = "annotation_frame_path"
POSE_COLUMNS = ["odometry_x","odometry_y","odometry_z","odometry_qx","odometry_qy","odometry_qz","odometry_qw"]

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df

