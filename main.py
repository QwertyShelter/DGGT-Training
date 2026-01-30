import os

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

import time
import torch
import lpips
import wandb
import random
import imageio
import argparse
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from dggt.models.vggt import VGGT
from dggt.utils.gs import concat_list, get_split_gs
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.geometry import unproject_depth_map_to_point_map
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from gsplat.rendering import rasterization
from datasets.dataset import WaymoOpenDataset
from datasets.davis_dataset import DavisDataset

import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP


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


class RGBLoss(nn.Module):
    def __init__(self, lambda_lpips=0.1, device='cpu'):
        super().__init__()
        self.lambda_lpips = lambda_lpips
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        # 设置为评估模式，禁用梯度计算，便于复现
        self.lpips_fn.eval()

    def forward(self, step, pred, target):
        mse_loss = F.l1_loss(pred, target)
        
        # 修改范围，对于 [0,1] 将映射到 [-1,1]
        pred_lpips = pred * 2 - 1 if pred.min() >= 0 else pred
        target_lpips = target * 2 - 1 if target.min() >= 0 else target
        
        lpips_loss = self.lpips_fn(pred_lpips, target_lpips).mean()

        # 逐步提高 lpips 损失的权重
        lpips_loss = min(step / 1000, 1.0) * lpips_loss
        return mse_loss + self.lambda_lpips * lpips_loss
    

def compute_lifespan_loss(gamma):
    return torch.mean(torch.abs(1 / (gamma + 1e-6)))


class FeedForwardLoss(nn.Module):
    def __init__(self, lambda_o=1, lambda_d=0.05, lambda_l=0.01, lambda_lpips=0.05, device='cpu'):
        super().__init__()

        self.lambda_o, self.lambda_d, self.lambda_l = lambda_o, lambda_d, lambda_l

        self.rgb_loss_fn = RGBLoss(lambda_lpips=lambda_lpips, device=device)
        self.opacity_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.dymask_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.lifespan_reg_fn = compute_lifespan_loss
    
    def forward(self, img, target, skymask, t_skymask, dymask, t_dymask, lifespan_params, step):
        rgb_loss = self.rgb_loss_fn(step, img, target)
        # opacity_loss = self.lambda_o * self.opacity_loss_fn(skymask, t_skymask)
        if t_dymask is not None:
            dymask_loss = self.lambda_d * self.dymask_loss_fn(dymask, t_dymask)
        else:
            dymask_loss = self.lambda_d * self.dymask_loss_fn(dymask, torch.zeros_like(dymask))
        lifespan_loss = self.lambda_l * self.lifespan_reg_fn(lifespan_params)
        
        total_loss =  rgb_loss + dymask_loss + lifespan_loss

        return {
            'total': total_loss,
            'rgb': rgb_loss,
            # 'mask': opacity_loss,
            'dymask': dymask_loss,
            'lifespan': lifespan_loss
        }
    

def print_cuda_memory_stats():
    # stats = torch.cuda.memory_stats()
    # for key in ['active.all.allocated', 'active.all.current', 'active.all.peak']:
    #     print(f"{key}: {stats[key]}")
    return
    print(torch.cuda.memory_summary(device=None, abbreviated=False))


# args includes: output_path, log_dir, save_image, save_ckpt, local_rank, max_epoch
def train(model, dataloader, optimizer, scheduler, loss_fn, step, dtype, device, args):
    
    train_loss_list = []
    training_time_list = []
    # scene_idx = 1
    points, colors = None, None
    
    for batch in tqdm(dataloader, disable=not args.is_main):

        time0 = time.time()
        if args.debug_output:
            print("Before training iteration:") if args.is_main else None    
            print_cuda_memory_stats() if args.is_main else None

        # load data from dataloader
        images = batch['images'].to(device)
        
        if 'masks' in batch:
            sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2)
            bg_mask = (sky_mask == 0).any(dim=-1)
        else:
            bg_mask = torch.ones((images.shape[0], images.shape[1], images.shape[3], images.shape[4]),
                                  dtype=torch.bool, device=device)
            
        timestamps = batch['timestamps'][0].to(device)

        start_time = time.time()
        if 'dynamic_mask' in batch:
            dynamic_masks = batch['dynamic_mask'].to(device)[:, :, 0, :, :]
        else:
            dynamic_masks = None

        optimizer.zero_grad()

        time1 = time.time()
        args.time_list['data'].append(time1 - time0)
        if args.debug_output:
            print("Batch data loaded:") if args.is_main else None    
            print(f"Time cost: {time1 - time0:.4f} seconds") if args.is_main else None
            print_cuda_memory_stats() if args.is_main else None

        with torch.cuda.amp.autocast(dtype=dtype):
            # Get the predictions from the model
            predictions = model(images)

            time2 = time.time()
            args.time_list['forward'].append(time2 - time1)
            if args.debug_output:
                print("Model forward pass done:") if args.is_main else None    
                print(f"Time cost: {time2 - time1:.4f} seconds") if args.is_main else None
                print_cuda_memory_stats() if args.is_main else None

            H, W = images.shape[-2:]
            extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
            extrinsic = extrinsics[0]
            bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
            extrinsic = torch.cat([extrinsic, bottom], dim=1)
            intrinsic = intrinsics[0]

            use_depth = True
            if use_depth:
                depth_map = predictions["depth"][0]
                point_map = unproject_depth_map_to_point_map(depth_map.detach(), extrinsics[0].detach(), intrinsics[0].detach())[None,...]
                point_map = torch.from_numpy(point_map).to(device).float()
            else:
                point_map = predictions["world_points"]
            gs_map = predictions["gs_map"]
            gs_conf = predictions["gs_conf"]
            dy_map = predictions["dynamic_conf"].squeeze(-1) #B,H,W,1

            # TODO: 这里训练的时候不需要考虑 dy_map ?
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

            time3 = time.time()
            if args.debug_output:
                print("Before rasterization:") if args.is_main else None    
                print(f"Time cost: {time3 - time2:.4f} seconds") if args.is_main else None
                print_cuda_memory_stats() if args.is_main else None

            for idx in range(dy_map.shape[1]):
                t0 = timestamps[idx]
                static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0 = static_gs_conf)
                static_gs_list = [static_points, static_rgbs, static_opacity_, static_scales, static_rotations]
                if dynamic_points:
                    world_points, rgbs, opacity, scales, rotation = concat_list(
                        static_gs_list,
                        [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx], dynamic_scales[idx], dynamic_rotations[idx]]
                    )
                # TODO: No this branch in train.py
                else:
                    world_points, rgbs, opacity, scales, rotation = static_gs_list
                # P.S. Render_mode in train.py is 'RGB', which means no depth is needed
                renders_chunk, alphas_chunk, _ = rasterization(
                    means=world_points,
                    quats=rotation,
                    scales=scales,
                    opacities=opacity,
                    colors=rgbs,
                    viewmats=extrinsic[idx][None],
                    Ks=intrinsic[idx][None],
                    width=W,
                    height=H
                )
                chunked_renders.append(renders_chunk)
                chunked_alphas.append(alphas_chunk)


            renders = torch.cat(chunked_renders, dim=0)
            alphas = torch.cat(chunked_alphas, dim=0)
            rendered_image = (alphas * renders).permute(0, 3, 1, 2)
            target_image = images[0]

            training_time = time.time() - start_time
            training_time_list.append(training_time)

            time4 = time.time()
            args.time_list['render'].append(time4 - time3)
            if args.debug_output:
                print("After rasterization:") if args.is_main else None    
                print(f"Time cost: {time4 - time3:.4f} seconds") if args.is_main else None
                print_cuda_memory_stats() if args.is_main else None
            
            # ============================== Loss ==============================

            t_skymask = bg_mask[0][..., None].type(torch.float32)
            loss_dict = loss_fn(rendered_image, target_image, alphas, t_skymask, dy_map, dynamic_masks, gs_conf, step)
            # loss_dict = loss_fn(rendered_image, target_image, dy_map, dynamic_masks, gs_conf, step)

            time5 = time.time()
            args.time_list['loss'].append(time5 - time4)
            if args.debug_output:
                print("Loss computed:") if args.is_main else None    
                print(f"Time cost: {time5 - time4:.4f} seconds") if args.is_main else None
                print_cuda_memory_stats() if args.is_main else None

        if step % args.save_image == 0:
            points = static_points.detach().cpu().numpy()  # (N, 3)
            colors = (static_rgbs * 255).detach().cpu().numpy()    # (N, 3)
        
        # 反向传播和优化器更新
        loss_dict['total'].backward()           # 反向传播
        optimizer.step()                        # 更新参数
        scheduler.step()
        
        # 记录损失
        train_loss_list.append(loss_dict['total'].item())

        time6 = time.time()
        args.time_list['backward'].append(time6 - time5)
        if args.debug_output:
            print("Backward down:") if args.is_main else None    
            print(f"Time cost: {time6 - time5:.4f} seconds") if args.is_main else None
            print_cuda_memory_stats() if args.is_main else None


    # =============================== Record ===============================
    if args.is_main:
        wandb.log({
            **{f"loss/{k}": v.item() for k, v in loss_dict.items()},
            "train/lr": scheduler.get_last_lr()[0],
            "time/iteration": training_time
        }, step=step)

        # if step % 1 == 0:
        #     print(f"[{step}/{args.max_epoch}] Loss: {loss_dict['total'].item():.4f} | LR: {scheduler.get_last_lr()}")

        if step % args.save_image == 0:
            # points = static_points[:100000].detach().cpu().numpy()  # (N, 3)
            # colors = (static_rgbs[:100000] * 255).detach().cpu().numpy()    # (N, 3)
            wandb.log({
                "3D_pointmaps": wandb.Object3D(
                    np.concatenate([points, colors], axis=1)
                )
            }, step=step)
            filename = os.path.join(args.log_dir, "pts3d", f"pts3d_{step}.ply")
            save_ply_open3d(filename, points, colors)
        
        if step > 0 and step % args.save_ckpt == 0:
            ckpt_path = os.path.join(args.log_dir, "ckpts", f"model_{step}.pt")
            torch.save(model.module.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved model at step {step} to {ckpt_path}")


# args includes: output_path, log_dir, save_image, save_ckpt, local_rank, max_epoch
# args includes: image_dir, scene_names, sequence_length, batch_size
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="debug", help='Experiment name')
    parser.add_argument('--image_dir', type=str, default="../../dataset/waymo_processed/validation", help='Path to the input images')
    parser.add_argument('--sequence_length', type=int, default=4, help='Number of input frames')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--log_dir', type=str, default="logs/debug", help='Path to the log directory')
    parser.add_argument('--save_image', type=int, default=5, help='Epoch intervals to save images')
    parser.add_argument('--save_ckpt', type=int, default=5, help='Epoch intervals to save checkpoints')
    parser.add_argument('--ckpt_path', type=str, default="logs/test/ckpts/model_final.pt", help='Path to the pre-trained checkpoint')
    parser.add_argument('--local_rank', type=int, default=6, help='Local rank for distributed training')
    parser.add_argument('--max_epoch', type=int, default=1, help='Maximum number of epochs')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start number of epochs')
    parser.add_argument('--debug_output', action='store_true', help='If to output time and GPU info')
    parser.add_argument('--dataset', type=str, default='waymo', help='Type of dataset to use')
    parser.add_argument('--random', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # ================ Initial ================
    dist.init_process_group(backend='nccl')
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.manual_seed(args.random)
    torch.cuda.manual_seed_all(args.random)
    dtype = torch.float32

    args.is_main = (args.local_rank == 0)

    if not args.is_main:
        import warnings
        warnings.filterwarnings("ignore")

    if args.is_main:
        wandb.init(
            project="dggt-waymo-fintune",
            name=f"{args.exp_name}",
            config=vars(args)
        )

    # ================ Dataset ================
    if args.dataset == 'waymo':
        dataset = WaymoOpenDataset(
            image_dir=args.image_dir,
            # scene_names=[str(i).zfill(3) for i in range(0,2)],
            scene_names=[str(i).zfill(3) for i in range(0,202)],
            sequence_length=args.sequence_length,
            mode=1,
            views=1
        )
    elif args.dataset == 'davis':
        dataset = DavisDataset(
            data_path=args.image_dir,
            seq_len=args.sequence_length,
            partial=False
        )
    else:
        if args.is_main:
            print(f"Unknown dataset type: {args.dataset}")
        wandb.finish()
        return
    
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)

    if args.is_main:
        print("Dataset loaded successfully !")
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, 'ckpts'), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, 'pts3d'), exist_ok=True)

    # ================ Model ================

    model = VGGT().to(device)
    if args.ckpt_path != "":
        if args.is_main:
            print(f"Resume training, loading model from {args.ckpt_path}...")
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)

    model.train()
    if args.is_main:
        print("VGGT model loaded successfully !")

    for param in model.parameters():
        param.requires_grad = False
    for head_name in ["point_head", "depth_head", "gs_head", "instance_head"]: #, "gs_head", "instance_head", "sky_model", "semantic_head"
        for param in getattr(model, head_name).parameters():
            param.requires_grad = True

    model = DDP(
        model,
        device_ids=[args.local_rank], 
        static_graph=False,
        find_unused_parameters=True
    )

    loss_fn = FeedForwardLoss(lambda_o=1, lambda_d=0.05, lambda_l=0.01, lambda_lpips=0.05, device=device)

    # ================ Optimizer & Scheduler ================

    optimizer = AdamW([
        {'params': model.module.point_head.parameters(), 'lr': 1e-4},
        {'params': model.module.depth_head.parameters(), 'lr': 1e-4},
        {'params': model.module.gs_head.parameters(), 'lr': 4e-5},
        {'params': model.module.instance_head.parameters(), 'lr': 4e-5}
    ], weight_decay=1e-4)

    warmup_iterations = 1000
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / warmup_iterations, 1.0) * 0.5 * (
            1 + torch.cos(torch.tensor(torch.pi * step / args.max_epoch)))
    )

    if args.is_main:
        print("Begin the training...")

    time_list = {}
    time_list['data'], time_list['forward'], time_list['render'], time_list['loss'], time_list['backward'] = [], [], [], [], []
    args.time_list = time_list

    for step in tqdm(range(args.start_epoch, args.max_epoch), disable=not args.is_main):
        train(model, dataloader, optimizer, scheduler, loss_fn, step, dtype, device, args)

    if args.is_main:
        ckpt_path = os.path.join(args.log_dir, "ckpts", f"model_final.pt")
        torch.save(model.module.state_dict(), ckpt_path)
        print(f"[Checkpoint] Saved final model to {ckpt_path}")

        for key in time_list:
            times = time_list[key]
            avg_time = sum(times) / len(times) if len(times) > 0 else 0
            print(f"Average time for {key}: {avg_time:.4f} seconds over {len(times)} iterations.")

        wandb.finish()


if __name__ == "__main__":
    main()