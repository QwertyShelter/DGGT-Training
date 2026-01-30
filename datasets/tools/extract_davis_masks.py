"""
@file   extract_masks.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Extract semantic mask

Using SegFormer, 2021. Cityscapes 83.2%
Relies on timm==0.3.2 & pytorch 1.8.1 (buggy on pytorch >= 1.9)

Installation:
    NOTE: mmcv-full==1.2.7 requires another pytorch version & conda env.
        Currently mmcv-full==1.2.7 does not support pytorch>=1.9; 
            will raise AttributeError: 'super' object has no attribute '_specify_ddp_gpu_num'
        Hence, a seperate conda env is needed.

    git clone https://github.com/NVlabs/SegFormer

    conda create -n segformer python=3.8
    conda activate segformer
    # conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

    pip install timm==0.3.2 pylint debugpy opencv-python attrs ipython tqdm imageio scikit-image omegaconf
    pip install mmcv-full==1.2.7 --no-cache-dir
    
    cd SegFormer
    pip install .

Usage:
    Direct run this script in the newly set conda env.
"""

from mmseg.apis import inference_segmentor, init_segmentor
import os
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser

semantic_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]

dataset_classes_in_sematic = {
    'Road': [0],
    'Building': [2],
    'Vegetation': [8],
    'Vehicle': [13, 14, 15],
    'Person': [11],
    'Cyclist': [12, 17, 18],
    'Traffic Sign': [9],
    'Sidewalk': [1],
    'Sky': [10],
    'Other': []
}



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/waymo/processed/training')
    parser.add_argument("--scene_ids", default=None, type=int, nargs="+")
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_scenes", type=int, default=200)
    parser.add_argument('--process_dynamic_mask', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ignore_existing', action='store_true')
    parser.add_argument('--no_compress', action='store_true')
    parser.add_argument('--rgb_dirname', type=str, default="images")
    parser.add_argument('--mask_dirname', type=str, default="fine_dynamic_masks")
    parser.add_argument('--segformer_path', type=str, default='/home/guojianfei/ai_ws/SegFormer')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--palette', default='cityscapes')

    args = parser.parse_args()

    scene_name_list = []
    image_path = os.path.join(args.data_root, 'JPEGImages', '480p')
    for name in os.listdir(image_path):
        scene_name_list.append(name)

    if args.config is None:
        args.config = os.path.join(args.segformer_path, 'local_configs', 'segformer', 'B5', 'segformer.b5.1024x1024.city.160k.py')
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.segformer_path, 'pretrained', 'segformer.b5.1024x1024.city.160k.pth')

    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    for scene_name in tqdm(scene_name_list, desc="Processing Scenes"):

        img_dir = os.path.join(image_path, scene_name)
        flist = sorted(glob(os.path.join(img_dir, '*')))

        sky_mask_dir = os.path.join(args.data_root, "sky_masks", scene_name)
        os.makedirs(sky_mask_dir, exist_ok=True)

        for fpath in tqdm(flist, desc=f'scene[{scene_name}]', leave=False):
            fbase = os.path.splitext(os.path.basename(fpath))[0]

            result = inference_segmentor(model, fpath)
            mask = result[0].astype(np.uint8)

            sky_mask = np.isin(mask, [10])
            imageio.imwrite(os.path.join(sky_mask_dir, f"{fbase}.png"), sky_mask.astype(np.uint8) * 255)