from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .visualization.encoder_visualizer_depthsplat_cfg import EncoderVisualizerDepthSplatCfg

import torchvision.transforms as T
import torch.nn.functional as F

from .unimatch.mv_unimatch import MultiViewUniMatch, set_num_views
from .unimatch.ldm_unet.unet import UNetModel
from .unimatch.feature_upsampler import ResizeConvFeatureUpsampler


import os
import torch
import matplotlib.pyplot as plt

from moge.model import MoGeModel

def generate_monocular_depths(mono_depth_model, images_1v3hw, extrinsics=None):
    """
    Generate scale-invariant monocular depth maps for each view in a batch.

    This function assumes the input images have shape [1, V, 3, H, W],
    i.e., a batch of size 1, containing V views, each with 3 color channels.
    It uses a MoGe-based monocular depth model to process each view independently.

    Args:
        mono_depth_model: A MoGeModel (or similar) used for monocular depth inference.
        images_1v3hw (torch.Tensor):
            A float tensor with shape [1, V, 3, H, W].
            Values should typically be normalized to [0,1], e.g. image / 255.
        extrinsics (any, optional):
            Placeholder for extrinsics if needed by the pipeline.
            This function does not currently use them.

    Returns:
        torch.Tensor:
            A float tensor of shape [1, V, H, W], representing scale-invariant depth
            for each view. The 0-th dimension remains 1 to keep consistent with the
            input batch dimension.
    """
    import torch
    device = next(mono_depth_model.parameters()).device
    images_1v3hw = images_1v3hw.to(device)

    # We'll accumulate depth results for each view in a list, then stack
    depth_maps = []

    # images_1v3hw shape is [1, V, 3, H, W]
    # We'll loop over the V dimension
    num_views = images_1v3hw.shape[1]

    for view_idx in range(num_views):
        # Extract the image at index `view_idx`: shape [3, H, W]
        image_3hw = images_1v3hw[0, view_idx]  # Because batch size is 1

        # MoGeModel.infer expects [3, H, W]
        output = mono_depth_model.infer(image_3hw)

        # output contains a dict with keys: "depth", "points", "mask", "intrinsics", etc.
        # We only need "depth" here.
        # depth_map shape: [H, W]
        depth_map = output["depth"]

        depth_maps.append(depth_map)

    # Stack along a new dimension => shape [V, H, W]
    # Then add a leading dimension for the batch => [1, V, H, W]
    depth_1vhw = torch.stack(depth_maps, dim=0)[None, ...]

    return depth_1vhw
import os
import torch
import matplotlib.pyplot as plt
import numpy as np


import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_depth_maps_v3(
    depth_tensor,
    color_tensor=None,
    mono_depth_tensor=None,
    extrinsics_tensor=None,
    intrinsics_tensor=None,
    output_dir="depth_debug",
    ply_filename="fused_pointcloud.ply"
):
    """
    Visualize depth maps, optionally with a color image and a second "mono_depth" map.
    If extrinsics and intrinsics are provided, fuse all views into a single 3D point cloud
    (N x 3) with per-point color and save to a .ply file.

    Args:
        depth_tensor (torch.Tensor):
            Shape = [1, V, H, W] or [V, H, W]. GPU or CPU.
        color_tensor (torch.Tensor, optional):
            Shape = [1, V, 3, H, W] or [V, 3, H, W].
        mono_depth_tensor (torch.Tensor, optional):
            Shape = [1, V, H, W] or [V, H, W]. If provided, we'll also visualize
            it and produce an error map with the primary depth.
        extrinsics_tensor (torch.Tensor, optional):
            Shape = [1, V, 4, 4] or [V, 4, 4]. Each [4,4] transform is presumably from
            the camera's local coordinate to the world frame. If provided, we fuse
            all views into a single point cloud.
        intrinsics_tensor (torch.Tensor, optional):
            Shape = [1, V, 3, 3] or [V, 3, 3]. Each [3,3] is a pinhole camera intrinsics.
            Typically: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]. If provided, we use it
            instead of a simplistic fx=fy=1, center=(W-1)/2, etc.
        output_dir (str):
            Directory to save the resulting images.
        ply_filename (str):
            Name of the final .ply file (saved in output_dir) that fuses all views.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------------------------------
    # 1) Depth
    # --------------------------------------------------------------------------------
    depth_cpu = depth_tensor.detach().cpu()
    if depth_cpu.dim() == 4 and depth_cpu.size(0) == 1:
        # shape [1, V, H, W] => [V, H, W]
        depth_cpu = depth_cpu.squeeze(0)
    depth_cpu = depth_cpu.contiguous()
    num_views = depth_cpu.shape[0]
    H, W = depth_cpu.shape[-2], depth_cpu.shape[-1]

    # --------------------------------------------------------------------------------
    # 2) Color
    # --------------------------------------------------------------------------------
    if color_tensor is not None:
        color_cpu = color_tensor.detach().cpu()
        if color_cpu.dim() == 5 and color_cpu.size(0) == 1:
            # shape [1, V, 3, H, W] => [V, 3, H, W]
            color_cpu = color_cpu.squeeze(0)
        color_cpu = color_cpu.contiguous()
        assert color_cpu.shape[0] == num_views, (
            f"Mismatched views: depth has {num_views}, color has {color_cpu.shape[0]}"
        )
    else:
        color_cpu = None

    # --------------------------------------------------------------------------------
    # 3) Mono Depth
    # --------------------------------------------------------------------------------
    if mono_depth_tensor is not None:
        mono_depth_cpu = mono_depth_tensor.detach().cpu()
        if mono_depth_cpu.dim() == 4 and mono_depth_cpu.size(0) == 1:
            # shape [1, V, H, W] => [V, H, W]
            mono_depth_cpu = mono_depth_cpu.squeeze(0)
        mono_depth_cpu = mono_depth_cpu.contiguous()
        assert mono_depth_cpu.shape[0] == num_views, (
            f"Mismatched views: depth has {num_views}, mono_depth has {mono_depth_cpu.shape[0]}"
        )
    else:
        mono_depth_cpu = None

    # --------------------------------------------------------------------------------
    # 4) Extrinsics
    # --------------------------------------------------------------------------------
    if extrinsics_tensor is not None:
        extrinsics_cpu = extrinsics_tensor.detach().cpu()
        if extrinsics_cpu.dim() == 4 and extrinsics_cpu.size(0) == 1:
            # shape [1, V, 4, 4] => [V, 4, 4]
            extrinsics_cpu = extrinsics_cpu.squeeze(0)
        extrinsics_cpu = extrinsics_cpu.contiguous()
        assert extrinsics_cpu.shape[0] == num_views, (
            f"Mismatched views: depth has {num_views}, extrinsics has {extrinsics_cpu.shape[0]}"
        )
    else:
        extrinsics_cpu = None

    # --------------------------------------------------------------------------------
    # 5) Intrinsics
    # --------------------------------------------------------------------------------
    if intrinsics_tensor is not None:
        intrinsics_cpu = intrinsics_tensor.detach().cpu()
        if intrinsics_cpu.dim() == 4 and intrinsics_cpu.size(0) == 1:
            # shape [1, V, 3, 3] => [V, 3, 3]
            intrinsics_cpu = intrinsics_cpu.squeeze(0)
        intrinsics_cpu = intrinsics_cpu.contiguous()
        assert intrinsics_cpu.shape[0] == num_views, (
            f"Mismatched views: {num_views}, intrinsics has {intrinsics_cpu.shape[0]}"
        )
    else:
        intrinsics_cpu = None

    # --------------------------------------------------------------------------------
    # Prepare arrays for point cloud fusion
    # --------------------------------------------------------------------------------
    all_points = []
    all_colors = []

    # Pre-generate pixel coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij"
    )
    y_coords = y_coords.contiguous()
    x_coords = x_coords.contiguous()

    # --------------------------------------------------------------------------------
    # Process each view
    # --------------------------------------------------------------------------------
    for view_idx in range(1):
        depth_map = depth_cpu[view_idx]  # [H, W]
        depth_map = depth_map.contiguous()

        if color_cpu is not None:
            color_map = color_cpu[view_idx].contiguous()  # [3, H, W]
        else:
            color_map = None

        # Visualization
        if mono_depth_cpu is None:
            # single depth
            d_min, d_max = depth_map.min(), depth_map.max()
            if (d_max - d_min) < 1e-7:
                depth_norm = torch.zeros_like(depth_map)
            else:
                depth_norm = (depth_map - d_min) / (d_max - d_min)

            if color_map is None:
                plt.figure(figsize=(6, 4))
                plt.imshow(depth_norm.numpy(), cmap='magma')
                plt.colorbar(label='Depth')
                plt.title(f"Depth Map - View {view_idx}")
                plt.axis('off')
            else:
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                color_img = color_map.permute(1,2,0).contiguous().numpy()
                ax[0].imshow(color_img.astype('float32'))
                ax[0].set_title(f"Color - View {view_idx}")
                ax[0].axis('off')

                im_d = ax[1].imshow(depth_norm.numpy(), cmap='magma')
                ax[1].set_title("Depth")
                ax[1].axis('off')
                fig.colorbar(im_d, ax=ax[1], fraction=0.046, pad=0.04)

            outpath = os.path.join(output_dir, f"depth_view_{view_idx:02d}.png")
            plt.savefig(outpath, bbox_inches='tight')
            plt.close()
            print(f"Saved depth map for view {view_idx} at: {outpath}")

        else:
            # dual depth => color|depth|mono_depth|error
            mono_map = mono_depth_cpu[view_idx].contiguous()

            global_min = min(depth_map.min(), mono_map.min())
            global_max = max(depth_map.max(), mono_map.max())

            if (global_max - global_min) < 1e-7:
                depth_norm = torch.zeros_like(depth_map)
                mono_norm = torch.zeros_like(mono_map)
            else:
                depth_norm = (depth_map - global_min) / (global_max - global_min)
                mono_norm = (mono_map - global_min) / (global_max - global_min)

            diff_map = (depth_map - mono_map).abs()
            diff_min, diff_max = diff_map.min(), diff_map.max()
            if diff_max < 1e-7:
                diff_norm = torch.zeros_like(diff_map)
            else:
                diff_norm = diff_map / diff_max

            fig, ax = plt.subplots(1, 4, figsize=(16, 4))
            # (1) color
            if color_map is not None:
                color_img = color_map.permute(1,2,0).contiguous().numpy()
                ax[0].imshow(color_img.astype('float32'))
                ax[0].set_title(f"Color - View {view_idx}")
            else:
                ax[0].imshow(depth_norm.numpy(), cmap='magma')
                ax[0].set_title("Depth")
            ax[0].axis('off')

            # (2) depth
            im_d = ax[1].imshow(depth_norm.numpy(), cmap='magma', vmin=0, vmax=1)
            ax[1].set_title("Depth")
            ax[1].axis('off')
            fig.colorbar(im_d, ax=ax[1], fraction=0.046, pad=0.04)

            # (3) mono_depth
            im_m = ax[2].imshow(mono_norm.numpy(), cmap='magma', vmin=0, vmax=1)
            ax[2].set_title("Mono Depth")
            ax[2].axis('off')
            fig.colorbar(im_m, ax=ax[2], fraction=0.046, pad=0.04)

            # (4) diff
            im_diff = ax[3].imshow(diff_norm.numpy(), cmap='magma', vmin=0, vmax=1)
            ax[3].set_title("|Depth - Mono|")
            ax[3].axis('off')
            fig.colorbar(im_diff, ax=ax[3], fraction=0.046, pad=0.04)

            outpath = os.path.join(output_dir, f"depth_mono_view_{view_idx:02d}.png")
            plt.savefig(outpath, bbox_inches='tight')
            plt.close()
            print(f"Saved depth + mono_depth for view {view_idx} at: {outpath}")

        # ----------------------------------------------------------------------------
        # 3D Fusion: If extrinsics & intrinsics => real pinhole
        # Else if only extrinsics => fallback
        # ----------------------------------------------------------------------------
        if extrinsics_cpu is not None and intrinsics_cpu is not None:
            # T shape [4,4], K shape [3,3]
            T = extrinsics_cpu[view_idx]
            K = intrinsics_cpu[view_idx]
            fx = K[0,0].item() * W 
            fy = K[1,1].item() * H
            cx = K[0,2].item() * W
            cy = K[1,2].item() * H

            z_vals = depth_map.reshape(-1)
            x_pix = x_coords.reshape(-1)
            y_pix = y_coords.reshape(-1)



            valid_mask = z_vals > 1e-8
            z_vals = z_vals[valid_mask]
            x_pix = x_pix[valid_mask]
            y_pix = y_pix[valid_mask]

            X_cam = (x_pix - cx)/fx * z_vals
            Y_cam = (y_pix - cy)/fy * z_vals
            Z_cam = z_vals

            ones = torch.ones_like(z_vals)
            pts_cam = torch.stack([X_cam, Y_cam, Z_cam, ones], dim=0)  # [4, N]
            pts_world = T @ pts_cam
            xyz_world = pts_world[:3, :].transpose(0,1).contiguous().numpy()


            # color
            if color_map is not None:
                color_flat = color_map.reshape(3, -1)
                color_flat = color_flat[:, valid_mask]
                color_flat = color_flat.permute(1,0)
                color_flat = (color_flat * 255.0).clamp(0, 255).to(torch.uint8)
                colors_np = color_flat.numpy()
            else:
                colors_np = np.zeros((xyz_world.shape[0], 3), dtype=np.uint8)

            all_points.append(xyz_world)
            all_colors.append(colors_np)

        elif extrinsics_cpu is not None:
            # fallback if no intrinsics
            T = extrinsics_cpu[view_idx]
            cx_fallback = (W - 1) / 2.0
            cy_fallback = (H - 1) / 2.0

            z_vals = depth_map.reshape(-1)
            x_pix = x_coords.reshape(-1)
            y_pix = y_coords.reshape(-1)

            valid_mask = z_vals > 1e-8
            z_vals = z_vals[valid_mask]
            x_pix = x_pix[valid_mask]
            y_pix = y_pix[valid_mask]

            X_cam = (x_pix - cx_fallback) * z_vals
            Y_cam = (y_pix - cy_fallback) * z_vals
            Z_cam = z_vals

            ones = torch.ones_like(z_vals)
            pts_cam = torch.stack([X_cam, Y_cam, Z_cam, ones], dim=0)
            pts_world = T @ pts_cam
            xyz_world = pts_world[:3, :].transpose(0,1).contiguous().numpy()

            if color_map is not None:
                color_flat = color_map.reshape(3, -1)
                color_flat = color_flat[:, valid_mask]
                color_flat = color_flat.permute(1, 0)
                color_flat = (color_flat * 255.0).clamp(0, 255).to(torch.uint8)
                colors_np = color_flat.numpy()
            else:
                colors_np = np.zeros((xyz_world.shape[0], 3), dtype=np.uint8)

            all_points.append(xyz_world)
            all_colors.append(colors_np)

    # --------------------------------------------------------------------------------
    # Finally, if extrinsics was used, we have a fused pointcloud
    # --------------------------------------------------------------------------------
    if extrinsics_cpu is not None:
        pts_concat = np.concatenate(all_points, axis=0)
        clr_concat = np.concatenate(all_colors, axis=0)
        N = pts_concat.shape[0]
        print(f"Fused pointcloud has {N} points")

        ply_path = os.path.join(output_dir, ply_filename)
        with open(ply_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {N}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(N):
                x, y, z = pts_concat[i]
                r, g, b = clr_concat[i]
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
        print(f"Saved fused pointcloud to {ply_path}")





def visualize_depth_maps_v2(depth_tensor, color_tensor=None, mono_depth_tensor=None, output_dir="depth_debug"):
    """
    Visualize depth maps, optionally with a color image and a second "mono_depth" map.

    Args:
        depth_tensor: (torch.Tensor) shape = [1, num_views, H, W]
                      or [num_views, H, W]. GPU or CPU.
        color_tensor: (torch.Tensor) shape = [1, num_views, 3, H, W] (optional).
        mono_depth_tensor: (torch.Tensor) shape = [1, num_views, H, W] (optional).
                           If provided, we also visualize it and compute an error map.
        output_dir: (str) directory to save the resulting images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Move depth to CPU and remove leading batch dims
    depth_cpu = depth_tensor.detach().cpu()
    if depth_cpu.dim() == 4:
        # shape [1, num_views, H, W] => [num_views, H, W]
        if depth_cpu.size(0) == 1:
            depth_cpu = depth_cpu.squeeze(0)
    # Now depth_cpu should be [num_views, H, W]
    num_views = depth_cpu.shape[0]

    # Move color to CPU if present
    if color_tensor is not None:
        color_cpu = color_tensor.detach().cpu()
        # shape [1, num_views, 3, H, W] => [num_views, 3, H, W]
        if color_cpu.dim() == 5 and color_cpu.size(0) == 1:
            color_cpu = color_cpu.squeeze(0)
        # optionally verify matching num_views
        assert color_cpu.shape[0] == num_views, (
            f"Mismatched views: depth has {num_views}, color has {color_cpu.shape[0]}")
    else:
        color_cpu = None

    # Move mono_depth to CPU if present
    if mono_depth_tensor is not None:
        mono_depth_cpu = mono_depth_tensor.detach().cpu()
        if mono_depth_cpu.dim() == 4:
            # shape [1, num_views, H, W] => [num_views, H, W]
            if mono_depth_cpu.size(0) == 1:
                mono_depth_cpu = mono_depth_cpu.squeeze(0)
        # verify matching views
        assert mono_depth_cpu.shape[0] == num_views, (
            f"Mismatched views: depth has {num_views}, mono_depth has {mono_depth_cpu.shape[0]}")
    else:
        mono_depth_cpu = None

    # Iterate over each view
    for view_idx in range(num_views):
        depth_map = depth_cpu[view_idx]  # [H, W]

        # If we only have depth + color => old logic
        if mono_depth_cpu is None:
            # ------------------------
            # Normalization for depth alone
            # ------------------------
            d_min, d_max = depth_map.min(), depth_map.max()
            if (d_max - d_min) < 1e-7:
                depth_norm = torch.zeros_like(depth_map)
            else:
                depth_norm = (depth_map - d_min) / (d_max - d_min)

            # ------------------------
            # Plot
            # ------------------------
            if color_cpu is None:
                # No color => single figure
                plt.figure(figsize=(6, 4))
                plt.imshow(depth_norm, cmap='magma')
                plt.colorbar(label='Depth')
                plt.title(f"Depth Map - View {view_idx}")
                plt.axis('off')
            else:
                # Color + depth => 1x2 subplots
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                # Left: color
                color_img = color_cpu[view_idx].permute(1,2,0).numpy()
                ax[0].imshow(color_img.astype('float32'))
                ax[0].set_title(f"Color - View {view_idx}")
                ax[0].axis('off')
                # Right: depth
                im_d = ax[1].imshow(depth_norm, cmap='magma')
                ax[1].set_title("Depth")
                ax[1].axis('off')
                fig.colorbar(im_d, ax=ax[1], fraction=0.046, pad=0.04)

            # Save
            save_path = os.path.join(output_dir, f"depth_view_{view_idx:02d}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"Saved depth map for view {view_idx} at: {save_path}")

        else:
            # ------------------------
            # We have mono_depth as well => triple visualize with error
            # We'll do 1x4 subplots: color | depth | mono_depth | error
            # same color scale for depth & mono_depth
            mono_map = mono_depth_cpu[view_idx]

            # find global min/max for both
            global_min = min(depth_map.min(), mono_map.min())
            global_max = max(depth_map.max(), mono_map.max())

            # avoid degenerate range
            if (global_max - global_min) < 1e-7:
                depth_norm = torch.zeros_like(depth_map)
                mono_norm = torch.zeros_like(mono_map)
            else:
                depth_norm = (depth_map - global_min) / (global_max - global_min)
                mono_norm = (mono_map - global_min) / (global_max - global_min)

            # difference map (absolute error)
            diff_map = (depth_map - mono_map).abs()
            diff_min, diff_max = diff_map.min(), diff_map.max()
            if diff_max < 1e-7:
                diff_norm = torch.zeros_like(diff_map)
            else:
                diff_norm = diff_map / diff_max

            # Construct figure with 4 subplots horizontally
            fig, ax = plt.subplots(1, 4, figsize=(16, 4))

            # 1) color
            if color_cpu is not None:
                color_img = color_cpu[view_idx].permute(1,2,0).numpy()
                ax[0].imshow(color_img.astype('float32'))
                ax[0].set_title(f"Color - View {view_idx}")
            else:
                # no color => just show empty / or depth
                ax[0].imshow(depth_norm, cmap='magma')
                ax[0].set_title("Depth")

            ax[0].axis('off')

            # 2) depth
            im_d = ax[1].imshow(depth_norm, cmap='magma', vmin=0, vmax=1)
            ax[1].set_title("Depth")
            ax[1].axis('off')
            fig.colorbar(im_d, ax=ax[1], fraction=0.046, pad=0.04)

            # 3) mono_depth
            im_m = ax[2].imshow(mono_norm, cmap='magma', vmin=0, vmax=1)
            ax[2].set_title("Mono Depth")
            ax[2].axis('off')
            fig.colorbar(im_m, ax=ax[2], fraction=0.046, pad=0.04)

            # 4) difference
            im_diff = ax[3].imshow(diff_norm, cmap='magma', vmin=0, vmax=1)
            ax[3].set_title("|Depth - Mono|")
            ax[3].axis('off')
            fig.colorbar(im_diff, ax=ax[3], fraction=0.046, pad=0.04)

            # Save
            save_path = os.path.join(output_dir, f"depth_mono_view_{view_idx:02d}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"Saved depth + mono_depth for view {view_idx} at: {save_path}")

def visualize_depth_maps(depth_tensor, color_tensor=None, output_dir="depth_debug"):
    """
    depth_tensor: (torch.Tensor) shape = [1, num_views, H, W] (또는 유사 형태)
                  GPU 상에 있을 수 있음
    color_tensor: (torch.Tensor) shape = [1, num_views, 3, H, W] (기본 None)
                  RGB 이미지(3채널)가 뎁스와 동일한 뷰 개수/해상도라 가정
                  GPU 상에 있을 수 있음
    output_dir:   (str) 저장할 폴더 경로
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------
    # 뎁스 텐서 CPU로 이동
    # ------------------------
    depth_cpu = depth_tensor.detach().cpu()  # shape = [1, num_views, H, W]
    depth_cpu = depth_cpu.squeeze(0)         # shape = [num_views, H, W]

    # color_tensor가 있을 경우 같이 CPU로 이동
    if color_tensor is not None:
        color_cpu = color_tensor.detach().cpu()  # shape = [1, num_views, 3, H, W] 가정
        color_cpu = color_cpu.squeeze(0)         # shape = [num_views, 3, H, W]

        # 뷰 수가 일치하는지 체크(필요한 경우)
        assert color_cpu.shape[0] == depth_cpu.shape[0], \
            f"depth_tensor와 color_tensor의 뷰 개수가 다릅니다. depth:{depth_cpu.shape[0]}, color:{color_cpu.shape[0]}"

    num_views = depth_cpu.shape[0]

    # ------------------------
    # 뷰별로 시각화
    # ------------------------
    for view_idx in range(num_views):
        depth_map = depth_cpu[view_idx]  # shape = [H, W]

        # 최소값 ~ 최대값으로 정규화
        min_val, max_val = depth_map.min(), depth_map.max()
        if (max_val - min_val) > 1e-7:
            depth_normalized = (depth_map - min_val) / (max_val - min_val)
        else:
            depth_normalized = torch.zeros_like(depth_map)

        # ------------------------
        # Matplotlib 시각화
        # ------------------------
        if color_tensor is None:
            # 컬러가 없는 경우 -> 뎁스맵 단독 시각화
            plt.figure(figsize=(6, 4))
            plt.imshow(depth_normalized, cmap='magma')
            plt.colorbar(label='Depth')
            plt.title(f"Depth Map - View {view_idx}")
            plt.axis('off')

        else:
            # 컬러가 있는 경우 -> subplot(1x2)에 (color / depth) 나란히 표시
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))

            # 왼쪽: color image
            color_image = color_cpu[view_idx]  # shape = [3, H, W]
            # PyTorch -> NumPy 형식으로 변경 (H, W, 3)
            color_image = color_image.permute(1, 2, 0).numpy()

            ax[0].imshow(color_image.astype("float32"))
            ax[0].set_title(f"Color - View {view_idx}")
            ax[0].axis('off')

            # 오른쪽: depth map
            im = ax[1].imshow(depth_normalized, cmap='magma')
            ax[1].set_title(f"Depth Map - View {view_idx}")
            ax[1].axis('off')

            # 컬러바 추가 (우측 subplot 기준)
            fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="Depth")

        # ------------------------
        # 파일 저장
        # ------------------------
        save_path = os.path.join(output_dir, f"depth_view_{view_idx:02d}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        print(f"Saved depth map for view {view_idx} at: {save_path}")


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int
    no_mapping: bool


@dataclass
class EncoderDepthSplatCfg:
    name: Literal["depthsplat"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerDepthSplatCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]

    # mv_unimatch
    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool

    # depthsplat color branch
    large_gaussian_head: bool
    color_large_unet: bool
    init_sh_input_img: bool
    feature_upsampler_channels: int
    gaussian_regressor_channels: int

    # loss config
    supervise_intermediate_depth: bool
    return_depth: bool

    # only depth
    train_depth_only: bool

    # monodepth config
    monodepth_vit_type: str

    # multi-view matching
    costvolume_nearest_n_views: Optional[int] = None
    multiview_trans_nearest_n_views: Optional[int] = None


class EncoderDepthSplat(Encoder[EncoderDepthSplatCfg]):
    def __init__(self, cfg: EncoderDepthSplatCfg) -> None:
        super().__init__(cfg)
        self.mono_depth_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to("cuda")                             
        for param in self.mono_depth_model.parameters():
            param.requires_grad = False

        self.depth_predictor = MultiViewUniMatch(
            num_scales=cfg.num_scales,
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            vit_type=cfg.monodepth_vit_type,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
        )

        if self.cfg.train_depth_only:
            return

        # upsample to the original resolution
        self.feature_upsampler = ResizeConvFeatureUpsampler(num_scales=cfg.num_scales,
                                                            lowest_feature_resolution=cfg.lowest_feature_resolution,
                                                            out_channels=self.cfg.feature_upsampler_channels,
                                                            vit_type=self.cfg.monodepth_vit_type,
                                                            )
        feature_upsampler_channels = self.cfg.feature_upsampler_channels
        
        # gaussians adapter
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # unet
        # concat(img, depth, match_prob, features)
        in_channels = 3 + 1 + 1 + feature_upsampler_channels
        channels = self.cfg.gaussian_regressor_channels

        modules = [
            nn.Conv2d(in_channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        ]

        if self.cfg.color_large_unet or self.cfg.gaussian_regressor_channels == 16:
            unet_channel_mult = [1, 2, 4, 4, 4]
        else:
            unet_channel_mult = [1, 1, 1, 1, 1]
        unet_attn_resolutions = [16]

        modules.append(
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1,
                attention_resolutions=unet_attn_resolutions,
                channel_mult=unet_channel_mult,
                num_head_channels=32 if self.cfg.gaussian_regressor_channels >= 32 else 16,
                dims=2,
                postnorm=False,
                num_frames=2,
                use_cross_view_self_attn=True,
            )
        )

        modules.append(nn.Conv2d(channels, channels, 3, 1, 1))

        self.gaussian_regressor = nn.Sequential(*modules)

        # predict gaussian parameters: scale, q, sh
        num_gaussian_parameters = self.gaussian_adapter.d_in + 2

        # predict opacity
        num_gaussian_parameters += 1

        # concat(img, features, unet_out, match_prob)
        in_channels = 3 + feature_upsampler_channels + channels + 1

        if self.cfg.feature_upsampler_channels != 128:
            self.gaussian_head = nn.Sequential(
                nn.Conv2d(in_channels, num_gaussian_parameters,
                            3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters,
                            num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
            )
        else:
            self.gaussian_head = nn.Sequential(
                nn.Conv2d(
                    in_channels, num_gaussian_parameters * 2, 3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters * 2,
                            num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
            )

        if self.cfg.init_sh_input_img:
            nn.init.zeros_(self.gaussian_head[-1].weight[10:])
            nn.init.zeros_(self.gaussian_head[-1].bias[10:])

        # init scale
        # first 3: opacity, offset_xy
        nn.init.zeros_(self.gaussian_head[-1].weight[3:6])
        nn.init.zeros_(self.gaussian_head[-1].bias[3:6])

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        if (
            self.cfg.costvolume_nearest_n_views is not None
            or self.cfg.multiview_trans_nearest_n_views is not None
        ):
            assert self.cfg.costvolume_nearest_n_views is not None
            with torch.no_grad():
                xyzs = context["extrinsics"][:, :, :3, -1].detach()
                cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                cameras_dist_index = torch.argsort(cameras_dist_matrix)

                cameras_dist_index = cameras_dist_index[:,
                                                        :, :self.cfg.costvolume_nearest_n_views]
        else:
            cameras_dist_index = None

        # depth prediction
        results_dict = self.depth_predictor(
            context["image"],
            attn_splits_list=[2],
            min_depth=1. / context["far"],
            max_depth=1. / context["near"],
            intrinsics=context["intrinsics"],
            extrinsics=context["extrinsics"],
            nn_matrix=cameras_dist_index,
        )

        # list of [B, V, H, W], with all the intermediate depths
        depth_preds = results_dict['depth_preds']

        # [B, V, H, W]
        depth = depth_preds[-1]
        mono_depth = generate_monocular_depths(self.mono_depth_model, 
                                               context["image"],
                                               context["extrinsics"])

        #visualize_depth_maps_v3(depth, context["image"], mono_depth, context["extrinsics"], context["intrinsics"], output_dir="depth_debug")
        # depth = mono_depth
        if self.cfg.train_depth_only:
            # convert format
            # [B, V, H*W, 1, 1]
            depths = rearrange(depth, "b v h w -> b v (h w) () ()")

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
                # supervise all the intermediate depth predictions
                num_depths = len(depth_preds)

                # [B, V, H*W, 1, 1]
                intermediate_depths = torch.cat(
                    depth_preds[:(num_depths - 1)], dim=0)
                intermediate_depths = rearrange(
                    intermediate_depths, "b v h w -> b v (h w) () ()")

                # concat in the batch dim
                depths = torch.cat((intermediate_depths, depths), dim=0)

                b *= num_depths

            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": None,
                "depths": depths
            }

        # update the num_views in unet attention, useful for random input views
        set_num_views(self.gaussian_regressor, v)

        # features [BV, C, H, W]
        features = self.feature_upsampler(results_dict["features_cnn"],
                                            results_dict["features_mv"],
                                            results_dict["features_mono"],
                                            )

        # match prob from softmax
        # [BV, D, H, W] in feature resolution
        match_prob = results_dict['match_probs'][-1]
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[
            0]  # [BV, 1, H, W]
        match_prob = F.interpolate(
            match_prob, size=depth.shape[-2:], mode='nearest')

        # unet input
        concat = torch.cat((
            rearrange(context["image"], "b v c h w -> (b v) c h w"),
            rearrange(depth, "b v h w -> (b v) () h w"),
            match_prob,
            features,
        ), dim=1)


        out = self.gaussian_regressor(concat)

        concat = [out,
                    rearrange(context["image"],
                            "b v c h w -> (b v) c h w"),
                    features,
                    match_prob]

        out = torch.cat(concat, dim=1)

        gaussians = self.gaussian_head(out)  # [BV, C, H, W]

        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)

        depths = rearrange(depth, "b v h w -> b v (h w) () ()")

        # [B, V, H*W, 1, 1]
        densities = rearrange(
            match_prob, "(b v) c h w -> b v (c h w) () ()", b=b, v=v)
        # [B, V, H*W, 84]
        raw_gaussians = rearrange(
            gaussians, "b v c h w -> b v (h w) c")

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:

            # supervise all the intermediate depth predictions
            num_depths = len(depth_preds)

            # [B, V, H*W, 1, 1]
            intermediate_depths = torch.cat(
                depth_preds[:(num_depths - 1)], dim=0)
            
            intermediate_depths = rearrange(
                intermediate_depths, "b v h w -> b v (h w) () ()")

            # concat in the batch dim
            depths = torch.cat((intermediate_depths, depths), dim=0)

            # shared color head
            densities = torch.cat([densities] * num_depths, dim=0)
            raw_gaussians = torch.cat(
                [raw_gaussians] * num_depths, dim=0)

            b *= num_depths

        # [B, V, H*W, 1, 1]
        opacities = raw_gaussians[..., :1].sigmoid().unsqueeze(-1)
        raw_gaussians = raw_gaussians[..., 1:]
        
        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / \
            torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        sh_input_images = context["image"]

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
            context_extrinsics = torch.cat(
                [context["extrinsics"]] * len(depth_preds), dim=0)
            context_intrinsics = torch.cat(
                [context["intrinsics"]] * len(depth_preds), dim=0)

            gaussians = self.gaussian_adapter.forward(
                rearrange(context_extrinsics, "b v i j -> b v () () () i j"),
                rearrange(context_intrinsics, "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
                input_images=sh_input_images.repeat(
                    len(depth_preds), 1, 1, 1, 1) if self.cfg.init_sh_input_img else None,
            )

        else:
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"],
                          "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"],
                          "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
                input_images=sh_input_images if self.cfg.init_sh_input_img else None,
            )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        gaussians = Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

        if self.cfg.return_depth:
            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": gaussians,
                "depths": depths
            }

        return gaussians

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        return None
