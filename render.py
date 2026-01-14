import argparse
import os
import random
import time
import numpy as np
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
    conf = torch.exp(sigma*torch.tensor(t0-t).to(gamma0.device)**2)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the input images')
    parser.add_argument('--scene_names', type=str, nargs='+', required=True, help='Scene names, supports formats like 3 5 7 or (3,7)')
    parser.add_argument('--input_views', type=int, default=1, help='Number of input views')
    parser.add_argument('--sequence_length', type=int, default=4, help='Number of input frames')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting frame index')
    parser.add_argument('--mode', type=int, choices=[1,2,3], required=True, help='Processing mode')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory for results')
    parser.add_argument('-images', action='store_true', help='Whether to output each frame image')
    parser.add_argument('--intervals', type=int, default=2, help='Interval for mode=3')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    loss_fn = lpips.LPIPS(net='alex').to(device)

    scene_names_str = ' '.join(args.scene_names)
    scene_names = parse_scene_names(scene_names_str)

    from datasets.dataset import load_and_preprocess_images

    image_path_list = []

    for root, dirs, files in os.walk(args.image_dir):
        for file in sorted(files):
            file = os.path.join(root, file)
            image_path_list.append(file)

    images = load_and_preprocess_images(image_path_list).to(device)    # [T, C, H, W]

    bg_mask = np.ones((1, images.shape[0], images.shape[2], images.shape[3]), dtype=bool)   # [B, T, H, W]
    bg_mask = torch.from_numpy(bg_mask).to(device)

    model = VGGT().to(device)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    if args.mode == 3:
        track_ckpt = 'path_to_track_model'
        track_model = load_model(track_ckpt)
        track_model.to(device)
        track_model.seq_len = 2
    model.eval()
    psnr_list, ssim_list, lpips_list = [], [], []
    inference_time_list = []
    scene_idx = 1

    with torch.no_grad():
        start_time = time.time()

        with torch.amp.autocast(dtype=dtype, device_type=device):
            predictions = model(images)
            S, _, H, W = images.shape
            extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
            extrinsic = extrinsics[0]
            bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
            extrinsic = torch.cat([extrinsic, bottom], dim=1)
            intrinsic = intrinsics[0]
            intervals=args.intervals
            views=args.input_views

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
            
            # 让每个有效像素点都能够对应到时间戳上
            frame_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1].cpu().numpy()  # B, T, H, W -> N, 4 -> N,  (取第二维度 T)
        
            start_idx = 0
            indices = [start_idx + i for i in range(S)]
            
            timestamps = np.array(indices) - start_idx
            timestamps = timestamps / timestamps[-1] * (S / 4)
            
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
            if args.mode == 3:
                origin_extrinsic = extrinsic
                origin_intrinsic = intrinsic   
            for idx in range(dy_map.shape[1]):
                if args.mode == 2:
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

                    world_points, rgbs, opacity, scales, rotation = [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx], dynamic_scales[idx], dynamic_rotations[idx]]

                    height_offset = 0
                    renders_chunk, alphas_chunk, _ = rasterization(
                        means=world_points,
                        quats=rotation,
                        scales=scales,
                        opacities=opacity,
                        colors=rgbs,
                        viewmats=extrinsic[idx:idx+1],
                        Ks=intrinsic[idx:idx+1],
                        width=W,
                        height=H,
                        render_mode='RGB+ED',  
                    )
                chunked_renders.append(renders_chunk)
                chunked_alphas.append(alphas_chunk)
            renders = torch.cat(chunked_renders, dim=0)
            depth_maps = renders[..., -1]
            renders = renders[..., :-1]
            alphas = torch.cat(chunked_alphas, dim=0)
            if args.mode == 3:
                bg_render = model.sky_model.forward_with_new_pose(images,origin_extrinsic,origin_intrinsic, extrinsic, intrinsic)
            if args.mode == 2:
                bg_render = model.sky_model(images, extrinsic, intrinsic)
                bg_render = (bg_render - bg_render.min()) / (bg_render.max() - bg_render.min() + 1e-8)  #
            renders = alphas * renders + (1 - alphas) * bg_render
            rendered_image = renders.permute(0, 3, 1, 2)
            target_image = images[0]

        scene_name = str(scene_idx).zfill(3)
        inference_time = time.time() - start_time
        inference_time_list.append(inference_time)

        scene_idx += 1

        scene_out_dir = os.path.join(args.output_path, scene_name)
        os.makedirs(scene_out_dir, exist_ok=True)

        if args.images:
            if args.input_views == 1:
                image_list = []
                for i in range(rendered_image.shape[0]):
                    rendered = rendered_image[i].detach().cpu().clamp(0, 1)
                    image_path = os.path.join(scene_out_dir, f"view_{i}.png")
                    T.ToPILImage()(rendered).save(image_path)
                    image_list.append(rendered.permute(1, 2, 0).numpy())
                video_path = os.path.join(scene_out_dir, "rendered_video.mp4")
                imageio.mimwrite(video_path, (np.array(image_list) * 255).astype(np.uint8), fps=8, codec="libx264")
            if args.input_views == 3:
                T_total = rendered_image.shape[0]
                groups = T_total // 3
                video_list = []
                for g in range(groups):
                    idx_center = 3 * g + 0
                    idx_left = 3 * g + 1
                    idx_right = 3 * g + 2
                    center = rendered_image[idx_center].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                    left = rendered_image[idx_left].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                    right = rendered_image[idx_right].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                    H, W = center.shape[0], center.shape[1]
                    # convert to uint8 HWC
                    def to_uint8(arr):
                        a = (arr * 255.0).astype(np.uint8)
                        if a.ndim == 2:
                            a = np.stack([a] * 3, axis=-1)
                        if a.shape[2] == 4:
                            a = a[:, :, :3]
                        return a
                    left_u = to_uint8(left)
                    center_u = to_uint8(center)
                    right_u = to_uint8(right)                            
                    white = np.ones((H,10, 3), dtype=np.uint8) * 255
                    composed = np.concatenate([left_u, white, center_u, white, right_u], axis=1)
                    # save image
                    image_path = os.path.join(scene_out_dir, f"view_{g:04d}.png")
                    Image.fromarray(composed).save(image_path)
                    video_list.append(composed)
                video_path = os.path.join(scene_out_dir, "rendered_video.mp4")
                imageio.mimwrite(video_path, np.array(video_list), fps=8, codec="libx264")

if __name__ == "__main__":
    main()