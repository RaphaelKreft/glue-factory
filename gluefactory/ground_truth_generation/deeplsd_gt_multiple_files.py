"""
Run the homography adaptation with Superpoint for all images in the minidepth dataset.
Goal: create groundtruth with superpoint. Format: stores groundtruth for every image in a separate file.
"""

import argparse
from pathlib import Path

import numpy as np
import cv2
import h5py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from tqdm import tqdm
from joblib import Parallel, delayed
import os
from gluefactory.settings import EVAL_PATH,DATA_PATH
from gluefactory.datasets import get_dataset
from gluefactory.models.lines.deeplsd import DeepLSD
from gluefactory.ground_truth_generation.generate_gt_deeplsd import generate_ground_truth_with_homography_adaptation


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


def get_dataset_and_loader(num_workers,distributed:bool=False):  # folder where dataset images are placed
    config = {
        'name': 'minidepth',  # name of dataset class in gluefactory > datasets
        'grayscale': True,  # commented out things -> dataset must also have these keys but has not
        'preprocessing': {
            'resize': [800, 800]
        },
        'test_batch_size': 1,  # prefix must match split mode
        'num_workers': num_workers,
        'split': 'test'  # if implemented by dataset class gives different splits
    }
    omega_conf = OmegaConf.create(config)
    dataset = get_dataset(omega_conf.name)(omega_conf)
    loader = dataset.get_data_loader(omega_conf.get('split', 'test'),shuffle=False,pinned=True,distributed=distributed)
    return loader



def process_image(img_data, net, num_H, output_folder_path, device):

    img = img_data["image"].to(device)  # B x C x H x W
    # Run homography adaptation
    distance_field, angle_field, _ = generate_ground_truth_with_homography_adaptation(img,net, num_H=num_H, bs=8)
    assert len(img_data["name"]) == 1, f"Image data name is {img_data['name']}"  # Currently expect batch size one!
    # store gt in same structure as images of minidepth
    img_name = img_data["name"][0]
    complete_out_folder = (output_folder_path / img_name).parent
    complete_out_folder.mkdir(parents=True, exist_ok=True)
    output_file_path = complete_out_folder / f"{Path(img_name).name.split('.')[0]}.hdf5"
    
    # Save the DF in a hdf5 file
    with h5py.File(output_file_path, "w") as f:
        f.create_dataset("deeplsd_distance_field", data=distance_field)
        f.create_dataset("deeplsd_angle_field", data=angle_field)


def export_ha(output_folder_path, num_H, n_gpus, image_name_list):
    world_size = n_gpus
    if n_gpus > 1:
        mp.spawn(export_ha_parallel,args=(n_gpus, output_folder_path, num_H,image_name_list,),nprocs=n_gpus,join=True)
    else:
        data_loader = get_dataset_and_loader(args.n_jobs_dataloader,distributed=False)
        device = 'cuda' if torch.cuda.is_available() and n_gpus > 0 else 'cpu'
        export_ha_seq(data_loader, output_folder_path, num_H, start_index, device,image_name_list)

def export_ha_parallel(rank,world_size, output_folder_path, num_H,image_name_list):
    dist.init_process_group("nccl",rank=rank,world_size=world_size)
    print(f"Hello from rank {rank}")
    data_loader = get_dataset_and_loader(world_size,distributed=True)
    device = f"cuda:{rank}"
    with open(image_name_list,"r") as f:
        image_list = f.readlines()
    net = DeepLSD({}).to(device)
    for img_data in data_loader:
        if img_data["name"][0] in image_list:
            print(f"Rank {rank}: Skipping image {img_data['name'][0]} because it already has GT", flush=True)
        process_image(img_data, net, num_H, output_folder_path, device)
        with open(image_name_list,"a") as f:
            f.write(img_data["name"][0] + "\n")
        print(f"Proc {rank} finished gt for image {int(img_data['index'])} with name {img_data['name']}",flush=True)


def export_ha_seq(data_loader, output_folder_path, num_H, device,image_name_list: str):
    net = DeepLSD({}).to(device)
    with open(image_name_list,"r") as f:
        image_list = f.readlines()
    for img_data in tqdm(data_loader, total=len(data_loader)):
        if img_data["name"][0] in image_list:
            print(f"Skipping image {img_data['name'][0]} because it already has GT")
            continue
        process_image(img_data, net, num_H, output_folder_path, device)
        with open(image_name_list,"a") as f:
            f.write(img_data["name"][0] + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, help='Output folder.', default="deeplsd_gt")
    parser.add_argument('--num_H', type=int, default=100, help='Number of homographies used during HA.')
    parser.add_argument('--n_jobs_dataloader', type=int, default=1, help='Number of jobs the dataloader uses to load images')
    parser.add_argument("--n_gpus",type=int,default=0,help="How many gpus we can use")
    parser.add_argument("--image_name_list",type=str,help="File with list of names of images that have been generated, relative to our team folder")
    args = parser.parse_args()
    image_name_list = DATA_PATH / args.image_name_list
    if not os.path.exists(image_name_list):
        with open(image_name_list,"w"): pass
    out_folder_path = EVAL_PATH / args.output_folder
    out_folder_path.mkdir(exist_ok=True, parents=True)
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    print("OUTPUT PATH: ", out_folder_path)
    print("NUMBER OF HOMOGRAPHIES: ", args.num_H)
    print("N DATALOADER JOBS: ", args.n_jobs_dataloader)
    print("N GPUS: ", args.n_gpus)

    export_ha(out_folder_path, args.num_H, args.n_gpus, image_name_list)
    print("Done !")
