"""
Run the homography adaptation for all images in a given folder
to generate ground truth heatmap using superpoint.
"""

import os
import argparse
import numpy as np
import cv2
import h5py
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from gluefactory.models.extractors.superpoint_open import SuperPoint

from gluefactory.geometry.homography import sample_homography_corners
from gluefactory.utils.image import numpy_image_to_torch
from gluefactory.geometry.homography import warp_points

conf = {
    "patch_shape": [800, 800],
    "difficulty": 0.8,
    "translation": 1.0,
    "n_angles": 10,
    "max_angle": 60,
    "min_convexity": 0.05,
}

sp_conf = {
    "max_num_keypoints": None,
    "nms_radius": 4,
    "detection_threshold": 0.005,
    "remove_borders": 4,
    "descriptor_dim": 256,
    "channels": [64, 64, 128, 128, 256],
    "dense_outputs": None,
    "weights": None,  # local path of pretrained weights
}

homography_params = {
    'translation': True,
    'rotation': True,
    'scaling': True,
    'perspective': True,
    'scaling_amplitude': 0.2,
    'perspective_amplitude_x': 0.2,
    'perspective_amplitude_y': 0.2,
    'patch_ratio': 0.85,
    'max_angle': 1.57,
    'allow_artifacts': True
}


def sample_homography(img, conf: dict, size: list):
    data = {}
    H, _, coords, _ = sample_homography_corners(img.shape[:2][::-1], **conf)
    data["image"] = cv2.warpPerspective(img, H, tuple(size))
    data["H_"] = H.astype(np.float32)
    data["coords"] = coords.astype(np.float32)
    data["image_size"] = np.array(size, dtype=np.float32)
    return data


def ha_df(img, num=100, border_margin=3, min_counts=5):
    """ Perform homography adaptation to regress line distance function maps.
    Args:
        img: a grayscale np image.
        num: number of homographies used during HA.
        border_margin: margin used to erode the boundaries of the mask.
        min_counts: any pixel which is not activated by more than min_count is BG.
    Returns:
        The aggregated distance function maps in pixels
        and the angle to the closest line.
    """
    h, w = img.shape[:2]

    aggregated_heatmap = np.zeros((w, h, num), dtype=np.float32)
    agg_hm_old = np.zeros((w, h), dtype=np.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SuperPoint(sp_conf).to(device)
    model.eval().to(device)

    with torch.no_grad():
        # iterate over num homographies
        for i in range(num):

            # warp image
            homography = sample_homography(img, conf, [w, h])

            # apply detector
            image_warped = homography["image"]
            pred = model({"image": numpy_image_to_torch(image_warped)[None].to(device)})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            #warped_heatmap = pred["heatmap"]
            keypoints = pred["keypoints"]
            scores = pred["keypoint_scores"]

            warped_back_kp = warp_points(keypoints, homography['H_'], inverse=True)
            warped_back_kp = np.floor(warped_back_kp).astype(int)

            scores_xy = np.zeros((w, h), dtype=np.float32)

            for j in range(len(warped_back_kp)):
                x, y = warped_back_kp[j][0] + 1, warped_back_kp[j][1] + 1
                if x < w and y < h:
                    aggregated_heatmap[y, x, i] = scores[j]

            # for each row j in warped_back_kp add scores[j] to aggregated_heatmap
            for j in range(len(warped_back_kp)):
                x, y = warped_back_kp[j][0], warped_back_kp[j][1]
                if x < w and y < h:
                    agg_hm_old[y, x] += scores[j]

    mask = aggregated_heatmap > 0

    aggregated_heatmap_nan = aggregated_heatmap.copy()
    aggregated_heatmap_nan[~mask] = np.nan

    median_scores_non_zero = np.nanmedian(aggregated_heatmap_nan, axis=2)

    return median_scores_non_zero


def process_image(img_path, randomize_contrast, num_H, output_folder):
    img = cv2.imread(img_path)

    new_size = (800, 800)
    resize_img = cv2.resize(img, new_size)
    # convert from BGR to grayscale
    img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)

    # Run homography adaptation
    superpoint_heatmap = ha_df(img, num=num_H)

    # Save the DF in a hdf5 file
    out_path = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(output_folder, out_path) + '.hdf5'
    with h5py.File(out_path, "w") as f:
        f.create_dataset("superpoint_heatmap", data=superpoint_heatmap)


def export_ha(images_list, output_folder, num_H, n_jobs):
    # Parse the data
    with open(images_list, 'r') as f:
        image_files = f.readlines()
    image_files = [path.strip('\n') for path in image_files]

    multiprocessing.set_start_method('spawn')

    # Process each image in parallel
    Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(process_image)(
        img_path, None, num_H, output_folder)
                                                       for img_path in tqdm(image_files, total=len(image_files)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('images_list', type=str, help='Path to a txt file containing the image paths.')
    parser.add_argument('output_folder', type=str, help='Output folder.')
    parser.add_argument('--num_H', type=int, default=100, help='Number of homographies used during HA.')
    parser.add_argument('--random_contrast', action='store_true',
                        help='Add random contrast to the images (disabled by default).')
    parser.add_argument('--n_jobs', type=int, default=5, help='Number of jobs to run in parallel.')
    args = parser.parse_args()

    print("IMAGE LIST: ", args.images_list)
    print("OUTPUT FOLDER: ", args.output_folder)
    print("NUM H: ", args.num_H)
    print("N JOBS: ", args.n_jobs)

    export_ha(args.images_list, args.output_folder, args.num_H, args.n_jobs)
    print("Done !")
