#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run SDS optimisation for a video sequence and render a visualization video.

Example:
    python run_video_demo.py \
        --ckpt_dir checkpoints/Multi_Inpaint \
        --video Path to the video to be optimized \
        --save_loc Path to save the optimized results and video demo
"""

import argparse
from pathlib import Path
from typing import Tuple, List

import os
import cv2
import numpy as np
import torch
import lpips
from PIL import Image
from ml_collections import config_dict

from Lighting.dataset import VideoFrameLoader
from Lighting.sds_common import load_models
from SDS.multiball_sds import sds_loop as sds_loop3d
from Lighting.envmap_utils import (
    envmap2ballimg_perspective,
    hdr2ldr,
    imagepoint2worldwithdepth,
)

# ------------------------- CLI -------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D SDS optimisation + visualization (auto device/dtype)")
    parser.add_argument("--ckpt_dir", required=True, type=Path, help="Checkpoint directory")
    parser.add_argument("--video", required=True, type=Path, help="Input video file")
    parser.add_argument("--save_loc", required=True, type=Path, help="Folder to save results")
    return parser.parse_args()

# ------------------------- Utils -------------------------

def _hf_device_index(device) -> int:
    """Map torch device to HF pipeline device index (CUDA idx or -1 for CPU)."""
    if isinstance(device, torch.device) and device.type == "cuda":
        return 0 if device.index is None else device.index
    return -1

def prepare_depth_control_image_withdepth(
    depth_map: torch.Tensor,
    mask: np.ndarray,
    depth_interpolate: float,
    out_size: Tuple[int, int] = (512, 512),
) -> Image.Image:
    W, H = out_size
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-12)
    depth_map = depth_map.squeeze(0).permute(1, 2, 0).repeat(1, 1, 3)

    mask_bool = torch.tensor(mask.astype(bool), device=depth_map.device)
    depth_to_interpolate = float(np.power(max(1e-6, 1.0 / depth_interpolate), 1 / 2.4))
    depth_interpolate_tensor = torch.full((H, W, 3), depth_to_interpolate, device=depth_map.device)
    depth_map[mask_bool] = depth_interpolate_tensor[mask_bool]

    image = Image.fromarray((depth_map.detach().cpu().numpy() * 255).astype(np.uint8))
    return image

# ------------------------- Visualizer -------------------------

class EnvMapVisualizer3D:
    def __init__(
        self,
        envmap,
        args_cfg,
        pipe_sds,
        device: torch.device,
        torch_dtype: torch.dtype,
        dataset: VideoFrameLoader,
    ):
        self.envmap = envmap
        self.args = args_cfg
        self.pipe_sds = pipe_sds
        self.device = device
        self.torch_dtype = torch_dtype
        self.dataset = dataset

        from transformers import pipeline as transformers_pipeline
        self.depth_estimator = transformers_pipeline(
            "depth-estimation",
            model="Intel/dpt-large",
            device=_hf_device_index(device),
        )

        first_img = self.dataset[0]["frames"][0]
        depth_map = self.depth_estimator(first_img)["predicted_depth"]
        W, H = first_img.size
        depth_map_gpu = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
        )
        dmin = torch.amin(depth_map_gpu, dim=[1, 2, 3], keepdim=True)
        dmax = torch.amax(depth_map_gpu, dim=[1, 2, 3], keepdim=True)
        depth_norm = (depth_map_gpu - dmin) / (dmax - dmin + 1e-12)
        large_depth_disparity = float(np.percentile(depth_norm.detach().cpu().numpy(), 15))
        large_depth = 1.0 / max(1e-6, large_depth_disparity)
        self.max_depth = large_depth
        self.depth_ratio = 512.0 / max(1e-6, (self.max_depth - 1.0))
        print(f"[Visualizer] max_depth≈{self.max_depth:.4f}  depth_ratio≈{self.depth_ratio:.4f}")

    def interpolate_envmap(self, x: float, y: float, z: float, t_idx: int):
        env = self.envmap.to_image(
            x, y, z, t=t_idx, device=self.device, depth_ratio=self.depth_ratio
        )
        env = hdr2ldr(env, exposure=0, gamma=2.4)
        env = (env - 0.5) * 2  # [-1,1] for blending
        return env

    def render_ball_image(
        self,
        interpolated_env,
        x: float,
        y: float,
        z: float,
        input_image: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        manual_pose = torch.tensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=torch.float32, device=self.device
        )
        manual_K = torch.tensor(
            [[480,   0, -256],
             [  0, 480, -256],
             [  0,   0,   -1]], dtype=torch.float32, device=self.device
        )

        world_point = imagepoint2worldwithdepth((x, y), manual_pose, manual_K, depth_queried=5)
        ball_param = list(world_point) + [0.80]  # (x,y,z,r)

        depth_map = self.depth_estimator(input_image)["predicted_depth"]
        W, H = input_image.size
        depth_map_gpu = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
        )

        ball_img = envmap2ballimg_perspective(
            interpolated_env[0], ball_param, manual_pose, manual_K,
            image_width=self.args.img_width, image_height=self.args.img_height, interp_mode="bilinear"
        )

        envmapmask = torch.ones(1, 128, 256, device=self.device)
        ball_img_mask = envmap2ballimg_perspective(
            envmapmask, ball_param, manual_pose, manual_K,
            image_width=self.args.img_width, image_height=self.args.img_height, interp_mode="bilinear"
        )
        ball_img_mask = ball_img_mask > 0
        mask_np = ball_img_mask.detach().cpu().numpy()[0] > 0
        mask_img_u8 = Image.fromarray((mask_np * 255).astype(np.uint8))

        depth_vis = prepare_depth_control_image_withdepth(
            depth_map_gpu, mask_np, depth_interpolate=z, out_size=(self.args.img_width, self.args.img_height)
        )

        input_masks_forball = self.pipe_sds.mask_processor.preprocess(
            mask_img_u8, height=self.args.img_height, width=self.args.img_width
        ).to(self.device, dtype=self.torch_dtype)

        input_images_for_ball = self.pipe_sds.image_processor.preprocess(
            [input_image], height=self.args.img_height, width=self.args.img_width
        ).to(self.device, dtype=self.torch_dtype)

        comp = input_images_for_ball[0].float() * (1 - input_masks_forball.float()) \
             + ball_img * input_masks_forball.float()

        rgb_out = self.pipe_sds.image_processor.postprocess(comp.detach())[0].resize(
            (self.args.img_width, self.args.img_height)
        )
        return rgb_out, depth_vis

    def _combine_side_by_side(self, left: Image.Image, right: Image.Image) -> Image.Image:
        canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        return canvas

    def save_envmap_pth(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        pth_path = save_dir / "optimized_envmap.pth"
        torch.save(self.envmap, pth_path)
        print(f"[Visualizer] Saved envmap (pth) -> {pth_path}")

    def render_demo_path_and_save_video(self, save_path: Path):
        frames: List[np.ndarray] = []

        paths = [((100, 100, 1.0), (400, 400, 10.0))]
        total_frames = len(self.dataset[0]["frames"])
        pause_frames = int(total_frames * 0.2)
        half_segment = int(total_frames * 0.5) - pause_frames // 2

        frame_index = 0

        # 1) start -> halfway
        for t in np.linspace(0, 0.5, half_segment, endpoint=False):
            if frame_index >= total_frames: break
            x, y, z = self._lerp_path(paths[0][0], paths[0][1], t)
            input_img = self.dataset[0]["frames"][frame_index]
            with torch.no_grad():
                env = self.interpolate_envmap(x, y, z, t_idx=frame_index + 1)
                rgb, depth_vis = self.render_ball_image(env, x, y, z, input_img)
            frames.append(np.array(self._combine_side_by_side(rgb, depth_vis)))
            frame_index += 1

        # 2) pause at mid
        x_mid, y_mid, z_mid = self._lerp_path(paths[0][0], paths[0][1], 0.5)
        for _ in range(pause_frames):
            if frame_index >= total_frames: break
            input_img = self.dataset[0]["frames"][frame_index]
            with torch.no_grad():
                env = self.interpolate_envmap(x_mid, y_mid, z_mid, t_idx=frame_index + 1)
                rgb, depth_vis = self.render_ball_image(env, x_mid, y_mid, z_mid, input_img)
            frames.append(np.array(self._combine_side_by_side(rgb, depth_vis)))
            frame_index += 1

        # 3) halfway -> end
        for t in np.linspace(0.5, 1.0, half_segment):
            if frame_index >= total_frames: break
            x, y, z = self._lerp_path(paths[0][0], paths[0][1], t)
            input_img = self.dataset[0]["frames"][frame_index]
            with torch.no_grad():
                env = self.interpolate_envmap(x, y, z, t_idx=frame_index + 1)
                rgb, depth_vis = self.render_ball_image(env, x, y, z, input_img)
            frames.append(np.array(self._combine_side_by_side(rgb, depth_vis)))
            frame_index += 1

        if not frames:
            raise RuntimeError("No frames generated for visualization.")

        H, W, _ = frames[0].shape
        save_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (W, H))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[Visualizer] Saved video -> {save_path}")

    @staticmethod
    def _lerp_path(start: Tuple[float, float, float], end: Tuple[float, float, float], t: float):
        x = start[0] * (1 - t) + end[0] * t
        y = start[1] * (1 - t) + end[1] * t
        z = start[2] * (1 - t) + end[2] * t
        return x, y, z

# ------------------------- Main -------------------------

def main() -> None:
    opt = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[Auto] device={device}, dtype={torch_dtype}")

    lpips_loss_fn = lpips.LPIPS(net="vgg").to(device=device)

    # ---------- Config for models / sampler ----------
    args = config_dict.ConfigDict()
    args.ckpt_dir = str(opt.ckpt_dir)
    args.model_option = "sd21"
    args.sampler = "ddim"
    args.offload = False
    args.lora_path = None
    args.use_lora = False
    args.use_torch_compile = False

    # diffusion / control
    args.guidance_scale = 12.5
    args.denoising_step = 15
    args.control_scale = 1.0
    args.img_width = 512
    args.img_height = 512
    args.depth_mode = "nearest"

    # prompt
    args.prompt = "perfect mirrored reflective chrome ball spheres"
    args.prompt_dark = "perfect black dark mirrored reflective chrome ball spheres"
    args.negative_prompt = ""

    # EV & exposure
    args.ev = "0,-2.5,-5"
    args.max_negative_ev = -5

    # SDS hyper-parameters
    args.num_sds_steps = 20
    args.sds_optimizer = "adam"
    args.sds_lr = 2e-3
    args.sds_lr_end = 2e-4
    args.sds_t_min = 0.02
    args.sds_t_max = 0.98
    args.sds_t_strategy = "decay_t_min"
    args.sds_space = "image"
    args.sds_multistep = 10
    args.sds_adaptive = True
    args.sds_weighting = "uniform"
    args.sds_weighted_loss = False
    args.sds_batch_size = 1

    # environment map
    args.envmap_type = "temporal"
    args.envmap_height = 128
    args.envmap_width = 256
    args.envmap_reg_smooth = 0.0
    args.envmap_reg_ev = 0.0
    args.envmap_reg_ref = 0.0
    args.envmap_init = "zero"
    args.envmap = None
    args.round = "first"

    # --------- Load models & data ------------------------------------------
    pipe_sds, control_generator, _ = load_models(args, device, torch_dtype)
    assert pipe_sds.scheduler.prediction_type == "epsilon", "Only epsilon prediction is supported"

    dataset = VideoFrameLoader(video_path=str(opt.video))
    input_images = dataset[0]

    # --------- Run SDS loop (OPTIMIZE ENV MAP) ------------------------------
    _frames_dict, envmap = sds_loop3d(
        args=args,
        pipe_sds=pipe_sds,
        save_loc=str(opt.save_loc), 
        input_images=input_images,
        ev=0,
        ball_radius=0.66,
        envmap=args.envmap,
        max_depth=350,
        envmap_ev_dict=None,
        torch_dtype=torch_dtype,
        device=device,
        lpips=lpips_loss_fn,
        log_dir=None,
        is_ipynb=False,
    )

    # --------- Visualize with the optimized envmap --------------------------
    visualizer = EnvMapVisualizer3D(
        envmap=envmap,
        args_cfg=args,
        pipe_sds=pipe_sds,
        device=device,
        torch_dtype=torch_dtype,
        dataset=dataset,
    )

    visualizer.save_envmap_pth(Path(opt.save_loc))

    video_out = Path(opt.save_loc) / "visualization.mp4"
    visualizer.render_demo_path_and_save_video(video_out)

    print("[Done] Envmap (.pth) and visualization (.mp4) saved under:", opt.save_loc)


if __name__ == "__main__":
    main()
