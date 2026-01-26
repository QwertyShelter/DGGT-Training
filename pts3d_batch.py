import argparse
import os
import random
import time
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
import scipy.spatial.transform
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image
import imageio
import matplotlib
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import open3d as o3d
from third_party.difix.infer import process_images_with_difix
from third_party.TAPIP3D.utils.inference_utils import load_model, read_video, inference, get_grid_queries, resize_depth_bilinear
from dggt.models.vggt import VGGT
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import concat_list, get_masked_gs, get_split_gs
from dggt.utils.visual_track import visualize_tracks_on_images
from gsplat.rendering import rasterization
from datasets.dataset import WaymoOpenDataset
from utils.interplation import interp_all
from utils.video_maker import make_comparison_video_quad
def alpha_t(t, t0, alpha, gamma0 = 1, gamma1 = 0.1):
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) / ((gamma0)**2 + 1e-6)
    conf = torch.exp(sigma*(t0-t)**2)
    alpha_ = alpha * conf
    return alpha_.float()

def compute_metrics(img1, img2, loss_fn):
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    psnr_list, ssim_list, lpips_list = [], [], []
    for i in range(img1.shape[0]):
        im1 = img1[i].cpu().permute(1, 2, 0).numpy()
        im2 = img2[i].cpu().permute(1, 2, 0).numpy()
        psnr = peak_signal_noise_ratio(im1, im2, data_range=1.0)
        ssim = structural_similarity(im1, im2, channel_axis=2, data_range=1.0)
        lpips_val = loss_fn(img1[i].unsqueeze(0) * 2 - 1, img2[i].unsqueeze(0) * 2 - 1)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips_val.item())
    return sum(psnr_list) / len(psnr_list), sum(ssim_list) / len(ssim_list), sum(lpips_list) / len(lpips_list)

def calculate_scale_factor(P1, P2):
    distances_P1 = torch.norm(P1[1:], dim=1)  
    distances_P2 = torch.norm(P2[1:], dim=1)  
    avg_distance_P1 = torch.mean(distances_P1)
    if avg_distance_P1 < 0.1: #almost not move
        return 1
    avg_distance_P2 = torch.mean(distances_P2)
    scale_factor = avg_distance_P2 / avg_distance_P1
    return scale_factor

def save_video(images, path, fps=8):
    images = images.detach().cpu()  # Ensure it's on CPU
    if images.max() <= 1.0:
        images = images * 255.0
    images = images.byte().permute(0, 2, 3, 1).numpy()  # [S, H, W, 3]
    
    imageio.mimwrite(path, images, fps=fps, codec='libx264')

def parse_scene_names(scene_names_str):
    scene_names_str = scene_names_str.strip()
    if scene_names_str.startswith("(") and scene_names_str.endswith(")"):
        start, end = scene_names_str[1:-1].split(",")
        return [str(i).zfill(3) for i in range(int(start), int(end)+1)]
    else:
        return [str(int(x)).zfill(3) for x in scene_names_str.split()]

import open3d as o3d
def save_ply_open3d(filename, pts3d, view):
    pts = np.asarray(pts3d).reshape(-1,3)
    colors = np.asarray(view).reshape(-1, view.shape[-1])
    if colors.shape[1] == 4:
        colors = colors[:, :3]
    # open3d expects colors float in [0,1]
    if colors.dtype != np.float32 and colors.dtype != np.float64:
        colors = colors.astype(np.float32) / 255.0
    else:
        if colors.max() > 1.5:
            colors = colors / 255.0
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pc, write_ascii=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the input images')
    parser.add_argument('--scene_names', type=str, nargs='+', required=True, help='Scene names, supports formats like 3 5 7 or (3,7)')
    parser.add_argument('--input_views', type=int, default=1, help='Number of input views')
    parser.add_argument('--sequence_length', type=int, default=4, help='Number of input frames')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting frame index')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory for results')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # scene_names_str = ' '.join(args.scene_names)
    # scene_names = parse_scene_names(scene_names_str)
    dataset = WaymoOpenDataset(
        args.image_dir,
        # scene_names=scene_names,
        scene_names=[str(i).zfill(3) for i in range(0,202)],
        sequence_length=args.sequence_length,
        start_idx=args.start_idx,
        views=args.input_views
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = VGGT().to(device)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    inference_time_list = []
    scene_idx = 1

    with torch.no_grad():
        for batch in tqdm(dataloader):
            pts3d_all, rgbs_all = [], []

            images = batch['images'].to(device)
            bg_mask = np.ones((1, images.shape[1], images.shape[3], images.shape[4]), dtype=bool)         # [B, T, H, W]
            bg_mask = torch.from_numpy(bg_mask).to(device)
            # sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2)
            # gt_dy_map = batch['dynamic_mask'].to(device)
            # gt_depth = batch['gt_depth'].to(device)

            # bg_mask = (sky_mask == 0).any(dim=-1)
            timestamps = batch['timestamps'][0].to(device)

            start_time = time.time()
            dynamic = False
            if 'dynamic_mask' in batch:
                dynamic = True
                dynamic_masks = batch['dynamic_mask'].to(device)[:, :, 0, :, :]

            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
                H, W = images.shape[-2:]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
                extrinsic = extrinsics[0]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                intrinsic = intrinsics[0]

                use_depth = True
                if use_depth:
                    depth_map = predictions["depth"][0]
                    point_map = unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])[None,...]
                    point_map = torch.from_numpy(point_map).to(device).float()
                else:
                    point_map = predictions["world_points"]
                gs_map = predictions["gs_map"]
                gs_conf = predictions["gs_conf"]
                dy_map = predictions["dynamic_conf"].squeeze(-1) #B,H,W,1

                static_mask = (bg_mask & (dy_map < 0.5))
                static_points = point_map[static_mask].reshape(-1, 3)
                gs_dynamic_list = dy_map[static_mask].sigmoid()
                static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
                static_opacity = static_opacity * (1 - gs_dynamic_list)
                static_gs_conf = gs_conf[static_mask]
                frame_idx = torch.nonzero(static_mask, as_tuple=False)[:,1]
                gs_timestamps = timestamps[frame_idx]

                dynamic_points, dynamic_rgbs, dynamic_opacitys, dynamic_scales, dynamic_rotations = [], [], [], [], []
                for i in range(dy_map.shape[1]):
                    point_map_i = point_map[:, i]
                    bg_mask_i = bg_mask[:, i]
                    dy_conf_i = dy_map[:, i].sigmoid()

                    dynamic_point = point_map_i[bg_mask_i].reshape(-1, 3)
                    dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(gs_map[:, i], bg_mask_i)
                    gs_dynamic_list_i = dy_map[:, i][bg_mask_i].sigmoid()
                    dynamic_opacity = dynamic_opacity * gs_dynamic_list_i

                    dynamic_points.append(dynamic_point)
                    dynamic_rgbs.append(dynamic_rgb)
                    dynamic_opacitys.append(dynamic_opacity)
                    dynamic_scales.append(dynamic_scale)
                    dynamic_rotations.append(dynamic_rotation)

                chunked_renders, chunked_alphas = [], [] 
                for idx in range(dy_map.shape[1]):
                    t0 = timestamps[idx]
                    static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0 = static_gs_conf)
                    static_gs_list = [static_points, static_rgbs, static_opacity_, static_scales, static_rotations]
                    '''
                    if dynamic_points:
                        world_points, rgbs, opacity, scales, rotation = concat_list(
                            static_gs_list,
                            [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx], dynamic_scales[idx], dynamic_rotations[idx]]
                        )
                    else:
                        world_points, rgbs, opacity, scales, rotation = static_gs_list
                    '''
                    if dynamic_points is None:
                        continue
                    
                    # world_points, rgbs, opacity, scales, rotation = static_gs_list
                    world_points, rgbs, opacity, scales, rotation = [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx], dynamic_scales[idx], dynamic_rotations[idx]]

                    # 将 world_points 和 rgb 拿出来就好了
                    pts3d_all.append(world_points.cpu().numpy())
                    rgbs_all.append(images[0][idx].permute(1, 2, 0).cpu().numpy().reshape(-1, 3))
                    # rgbs_all.append((rgbs * 255).cpu().numpy())

            scene_name = str(scene_idx).zfill(3)
            inference_time = time.time() - start_time
            inference_time_list.append(inference_time)
            scene_idx += 1

            scene_out_dir = os.path.join(args.output_path, scene_name)
            os.makedirs(scene_out_dir, exist_ok=True)

            for idx in range(len(pts3d_all)):
                filename = os.path.join(scene_out_dir, f'scene_{scene_idx:03d}_frame_{idx:03d}.ply')
                # save_ply(filename, pts3d_all[idx].reshape(-1, 3), rgbs_all[idx].reshape(-1, 3))
                save_ply_open3d(filename, pts3d_all[idx], rgbs_all[idx])


            
if __name__ == "__main__":
    main()