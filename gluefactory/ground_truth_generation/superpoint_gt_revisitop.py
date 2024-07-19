import argparse
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from joblib import Parallel, delayed
from kornia.geometry.transform import warp_perspective
from kornia.morphology import erosion
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.datasets import get_dataset
from gluefactory.geometry.homography import sample_homography_corners
from gluefactory.models.extractors.superpoint_open import SuperPoint
from gluefactory.settings import EVAL_PATH

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

H_params = {
    "difficulty": 0.8,
    "translation": 1.0,
    "max_angle": 60,
    "n_angles": 10,
    "min_convexity": 0.05,
}

ha = {
    "enable": False,
    "num_H": 100,
    "mini_bs": 3,
    "aggregation": "mean",
}

homography_params = {
    "translation": True,
    "rotation": True,
    "scaling": True,
    "perspective": True,
    "scaling_amplitude": 0.2,
    "perspective_amplitude_x": 0.2,
    "perspective_amplitude_y": 0.2,
    "patch_ratio": 0.85,
    "max_angle": 1.57,
    "allow_artifacts": True,
}


def get_image_paths(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


def sample_homography(img, conf: dict, size: list):
    data = {}
    H, _, coords, _ = sample_homography_corners(img.shape[:2][::-1], **conf)
    data["image"] = cv2.warpPerspective(img, H, tuple(size))
    data["H_"] = H.astype(np.float32)
    data["coords"] = coords.astype(np.float32)
    data["image_size"] = np.array(size, dtype=np.float32)
    return data


def ha_forward(img, num=100):
    h, w = img.shape[:2]

    aggregated_heatmap = np.zeros((w, h, num), dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SuperPoint(sp_conf).to(device)
    model.eval().to(device)

    Hs = []
    for i in range(num):
        if i == 0:
            Hs.append(torch.eye(3, dtype=torch.float, device=device))
        else:
            Hs.append(
                torch.tensor(
                    sample_homography_corners((w, h), patch_shape=(w, h), **H_params)[0],
                    dtype=torch.float,
                    device=device,
                )
            )
    Hs = torch.stack(Hs, dim=0)

    bs = ha["mini_bs"]
    B = 1

    erosion_kernel = torch.tensor(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=torch.float,
    ).to(device)

    sp_image_tensor = (
        torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    )
    n_mini_batch = int(np.ceil(num / bs))
    scores = torch.empty((B, 0, h, w), dtype=torch.float, device=device)
    counts = torch.empty((B, 0, h, w), dtype=torch.float, device=device)

    for i in range(n_mini_batch):
        H = Hs[i * bs : (i + 1) * bs]
        nh = len(H)
        H = H.repeat(B, 1, 1).to(device)

        a = torch.repeat_interleave(sp_image_tensor, nh, dim=0)
        warped_imgs = warp_perspective(a, H, (h, w), mode="bilinear")

        for j, img in enumerate(warped_imgs):
            with torch.no_grad():
                img1 = img / 255.0  # Normalize image
                img1 = img1.unsqueeze(0)  # Add batch dimension
                pred = model({"image": img1.to(device)})
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

                warped_heatmap = pred["heatmap"]

                score = torch.tensor(
                    warped_heatmap, dtype=torch.float32, device=device
                ).unsqueeze(0)

                H_inv = torch.inverse(H[j])
                count = warp_perspective(
                    torch.ones_like(score).unsqueeze(1),
                    H[j].unsqueeze(0),
                    (h, w),
                    mode="nearest",
                )

                count = erosion(count, erosion_kernel)
                count = warp_perspective(
                    count, H_inv.unsqueeze(0), (h, w), mode="nearest"
                )[:, 0]
                score = warp_perspective(
                    score[:, None], H_inv.unsqueeze(0), (h, w), mode="bilinear"
                )[:, 0]

            scores = torch.cat([scores, score.reshape(B, 1, h, w)], dim=1)
            counts = torch.cat([counts, count.reshape(B, 1, h, w)], dim=1)
            scores[counts == 0] = 0
            score = scores.max(dim=1)[0]

            scoremap = score.squeeze(0)
    return scoremap


def process_image(image_path, num_H, output_folder_path):
    if not os.path.exists(image_path):
        print(f"File does not exist: {image_path}")
        return

    print(f"Attempting to load image: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    sp_image = (img * 255).astype(np.uint8)
    superpoint_heatmap = ha_forward(sp_image, num=num_H)
    superpoint_heatmap = superpoint_heatmap.cpu()

    print("output_folder_path: ", output_folder_path)
    print("image_path: ", image_path)

    unique_name = Path(image_path).parts[-3] + Path(image_path).parts[-2]
    output_file_path = output_folder_path / f"{unique_name}.hdf5"

    print("output_file_path: ", output_file_path)

    

    with h5py.File(output_file_path, "w") as f:
        f.create_dataset("superpoint_heatmap", data=superpoint_heatmap)


def export_ha(image_paths, output_folder_path, num_H, n_jobs):
    Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(process_image)(image_path, num_H, output_folder_path)
        for image_path in tqdm(image_paths)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_list", type=str, required=True, help="Path to the image list file."
    )
    parser.add_argument(
        "--output_folder", type=str, help="Output folder.", default="superpoint_gt"
    )
    parser.add_argument(
        "--num_H", type=int, default=100, help="Number of homographies used during HA."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=2,
        help="Number of jobs (that perform HA) to run in parallel.",
    )
    args = parser.parse_args()

    out_folder_path = Path(args.output_folder)
    out_folder_path.mkdir(exist_ok=True, parents=True)

    image_paths = get_image_paths(args.image_list)
    base_image_path = '/home/egoedeke/Downloads/revisitop1m_POLD2/jpg/'
    full_image_paths = [os.path.join(base_image_path, os.path.splitext(path)[0], 'base_image.jpg') for path in image_paths]
    
    for path in full_image_paths:
        print(f"Full image path: {path}")

    export_ha(full_image_paths, out_folder_path, args.num_H, args.n_jobs)
    print("Done!")
