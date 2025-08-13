from typing import Optional, List, Dict
from functools import partial

from IPython.display import display
import os
import random
import torchvision
import torch
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from typing import List, Optional, Dict
from PIL import Image
from tqdm import tqdm
import mediapy
from PIL import Image
from transformers import pipeline as transformers_pipeline
from Lighting.sds_utils import PipeSDS21
from Lighting.envmap_model import construct_envmap, EnvironmentMap 
from Lighting.envmap_utils import envmap2ballimg_perspective, hdr2ldr, imagepoint2worldwithdepth
from Lighting.inpaint_multi import create_argparser
from Lighting.sds_common import *
try:
    from IPython.display import clear_output
except Exception as e:
    pass


def create_argparser_sds():
    parser = create_argparser()
    
    parser.add_argument("--guidance_scale", type=float, default=7.5)

    parser.add_argument("--envmap_type", type=str, default="grid_3x3")
    parser.add_argument("--envmap_init", type=str, default="black")
    parser.add_argument("--envmap_height", type=int, default=128)
    parser.add_argument("--envmap_width", type=int, default=256)
    parser.add_argument("--envmap_reg_ev", type=float, default=0.0)
    parser.add_argument("--envmap_reg_smooth", type=float, default=0.0)
    parser.add_argument("--envmap_reg_ref", type=float, default=0.0)
    # parser.add_argument("--envmap_proj_factor", type=int, default=2)
    parser.add_argument("--num_sds_steps", type=int, default=300)
    parser.add_argument("--sds_batch_size", type=int, default=1)
    parser.add_argument("--sds_optimizer", type=str, default="adam")
    parser.add_argument("--sds_lr", type=float, default=0.1)
    parser.add_argument("--sds_lr_end", type=float, default=0.01)
    parser.add_argument("--sds_t_min", type=float, default=0.02)
    parser.add_argument("--sds_t_max", type=float, default=0.98)
    parser.add_argument("--sds_t_strategy", type=str, default="decay_t_min")
    parser.add_argument("--sds_use_fixed_noise", action="store_true")
    parser.add_argument("--sds_space", type=str, default="image")
    parser.add_argument("--sds_multistep", type=int, default=10)
    # parser.add_argument("--sds_adaptive_step", action="store_true", default=True)
    parser.add_argument("--sds_weighting", type=str, default="uniform")
    parser.add_argument("--sds_cascade", action="store_true")

    return parser

def project_multiball(pipe_sds, coordinates, ev,time, depth_map, manual_pose, manual_K, ball_radius, MLP, device, depth_ratio, input_image, haj):

    ball_params_list = []
    envmap_mask = torch.ones(1, 128, 256).float().to(device=device)
    balls_list = []
    masks_list = []
    final_mask = torch.zeros(512, 512).float().to(device=device)
    final_control_image = torch.zeros(3, 512, 512).float().to(device=device)
    i=0
    for x, y, z in coordinates:
        world_loc = imagepoint2worldwithdepth((x, y), manual_pose, manual_K, depth_queried=10)
        r = ball_radius
        ball_params = list(world_loc) + [r]
        ball_params_list.append(ball_params)
        interpolated = MLP.to_image(x, y, z, time+1, device, depth_ratio)
        
        exposure_adjusted_envmap = hdr2ldr(interpolated, exposure = ev, gamma=2.4)
        exposure_adjusted_envmap = (exposure_adjusted_envmap-0.5)*2
        ball_img = envmap2ballimg_perspective(
            exposure_adjusted_envmap[0], ball_params, manual_pose, manual_K,
            image_width=512, image_height=512, interp_mode='bilinear'
        )
        ball_img_mask = envmap2ballimg_perspective(
            envmap_mask, ball_params, manual_pose, manual_K,
            image_width=512, image_height=512, interp_mode='bilinear'
        )
        ball_img_mask = ball_img_mask > 0
        masktosave = ball_img_mask.detach().cpu().numpy()[0]
        #another mask
        mask_for_cumulate = ball_img_mask.squeeze(0)
        current_mask = mask_for_cumulate.float()
        final_mask = torch.maximum(final_mask, current_mask)


        masktosave = masktosave>0
        masktosave = Image.fromarray((masktosave * 255).astype(np.uint8))
        control_img = prepare_depth_control_image_withdepth(depth_map, masktosave, index=0, depth_interpolate=z)
        if i==0 :
            final_control_image = np.array(control_img)
        else:
            current_mask = current_mask.cpu().numpy()
            current_masks= np.expand_dims(current_mask, axis=-1) 
            current_masks = np.repeat(current_masks, 3, axis=-1)
            final_control_image = final_control_image*(1-current_masks)+np.array(control_img)*current_masks

        input_mask_forball = pipe_sds.mask_processor.preprocess(masktosave, height=512, width=512).to(device=device, dtype=torch.float16)

        #input_images_forball = input_images_forball.float()
        input_mask_forball = input_mask_forball.float()
        balls_list.append(ball_img)
        masks_list.append(input_mask_forball)
        i+=1
    input_images_forball = pipe_sds.image_processor.preprocess(input_image, height=512, width=512).to(device=device, dtype=torch.float16)
    
    for i in range(len(balls_list)):
        if i==0:
            balls = balls_list[i]
            masks = masks_list[i]
            composite = balls*masks+input_images_forball*(1-masks)
        else:
            balls = balls_list[i]
            masks = masks_list[i]
            composite = balls*masks+composite*(1-masks)

    cul_mask = final_mask > 0
    final_control_image = Image.fromarray(final_control_image.astype(np.uint8))
    cul_mask = cul_mask.cpu().numpy()
    cul_mask = cul_mask*255.0
    cul_mask = Image.fromarray(cul_mask.astype(np.uint8))

    return composite, cul_mask, final_control_image

def get_local_depth(depth_map, image, x, y):

    W, H = image.size

    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)

    
    
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)

    local_depth_disparity = depth_map[0, 0, y, x].item()+1e-6
    local_depth = 1/local_depth_disparity
    local_depth = np.random.uniform(1, local_depth)
    return local_depth
def prepare_depth_control_image_withdepth(depth_map, mask, index=0, depth_interpolate=0.0):

    W, H = 512, 512

    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)

    # Convert depth_map to the correct shape (H, W, 3)
    depth_map = depth_map.squeeze(0).permute(1, 2, 0).repeat(1, 1, 3)

    mask_bool = torch.tensor(np.array(mask).astype(bool))
    depth_to_interpolate = np.power(1.0/depth_interpolate, 1/2.4)
    
    # Create a tensor for the interpolated depth
    depth_interpolate_tensor = torch.full((H, W, 3), depth_to_interpolate, device=depth_map.device)
    
    # Apply the mask
    depth_map[mask_bool] = depth_interpolate_tensor[mask_bool]

    # Convert to image
    image = Image.fromarray((depth_map.cpu().numpy() * 255).astype(np.uint8))
    return image

def sds_loop(
    args,
    pipe_sds: PipeSDS21,
    save_loc: str,
    input_images: List[Image.Image],
    ev: float,
    ball_radius: float,
    max_depth: float,
    torch_dtype: torch.dtype,
    device: torch.device,
    envmap: Optional[EnvironmentMap] = None,
    envmap_ev_dict: Optional[Dict] = None,
    ref_images: Optional[List[Image.Image]] = None,
    seed: Optional[int] = 48,
    lpips: Optional[torch.nn.Module] = None,
    log_dir: Optional[str] = None,
    is_ipynb: Optional[bool] = False
):
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)

    batch_size = min(args.sds_batch_size, len(input_images))
    print(f"Using batch size of {batch_size}")

    # image preprocessing
    do_classifier_free_guidance = args.guidance_scale > 1.0

    #creating depth map
    depth_estimator = transformers_pipeline("depth-estimation", model="Intel/dpt-large", device=device)

    frames = input_images["frames"]
    depth_map_list = []
    depth_map_gpu_list = []
    depth_index = 0
    for frame in frames:
        depth_map = depth_estimator(frame)['predicted_depth']
        depth_map_list.append(depth_map)
        W, H = frame.size
        depth_map_gpu = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        
        depth_map_gpu_list.append(depth_map_gpu)
    depth_min = torch.amin(depth_map_gpu_list[0], dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map_gpu_list[0], dim=[1, 2, 3], keepdim=True)
    depth_normalized = (depth_map_gpu_list[0] - depth_min) / (depth_max - depth_min)
    depth_flattened = depth_normalized.reshape(depth_normalized.shape[0], -1)  # (batch_size, H*W)
    large_depth_disparity = torch.quantile(depth_flattened, 0.1, dim=1)  # (batch_size,)
    large_depth = 1 / large_depth_disparity
    max_depth = large_depth.max().item()
    args.max_depth = max_depth 
    depth_ratio = 512/(max_depth-1)

    W, H = 512,512
    
    if envmap is None:
        grid_pos_start = 70
        grid_pos_end = args.img_width - 70
        args.depth = max_depth
        args.grid_start = grid_pos_start
        args.grid_end = grid_pos_end
        envmap = construct_envmap(args)
    else:
        envmap = torch.load(envmap).to(device=device)
    envmap = envmap.to(device=device)
    for param in envmap.parameters():
        param.requires_grad = True


    optimizer = get_optimizer(args)(envmap.parameters(), lr=args.sds_lr)
    lr_func = partial(
        get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=args.num_sds_steps // 10, num_training_steps=args.num_sds_steps, num_cycles=0.5, min_scale=args.sds_lr_end / args.sds_lr
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_func)

    latent_noise = None
    accumulation_steps = 1
    
    # training loop
    pbar = tqdm(range(args.num_sds_steps))
    frames_dict = {}
    manual_pose = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float().to(device=device)
    manual_K = torch.tensor([[960, 0, -256], [0, 960, -256], [0, 0, -1]]).float().to(device=device)
    frames_num = len(input_images["frames"])
    input_copy = input_images.copy()
    for i in pbar:
        if i < args.num_sds_steps // 2:
            lower_bound = -5.0 * (i / (args.num_sds_steps // 2))
        else:
            lower_bound = -5.0 
        #ev = np.random.uniform(lower_bound, 0)
        ev = 0
        frame = np.random.randint(0, frames_num)
        depth_map_gpu = depth_map_gpu_list[frame]
        input_images = [input_copy["frames"][frame]]
        prompt_embeds = get_interpolated_prompt_embeds(pipe_sds, ev, args.max_negative_ev, device, args)
        pos, negative_prompt_embeds = pipe_sds.encode_prompt(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True  
        )
        # sample t
        alphas_cumprod = pipe_sds.scheduler.alphas_cumprod 
        t = sds_t_sampling(i, args.sds_t_min, args.sds_t_max, args.sds_t_strategy, args.num_sds_steps,alphas_cumprod=alphas_cumprod)
        timestep = int(pipe_sds.scheduler.config.num_train_timesteps * t.item())
        
        seed = seed+i
        np.random.seed(seed)
        

        deviation = np.random.randint(0, 90)
        x_center = random.choice([-1, 1])
        y_center = random.choice([-1, 1])
        x_deviation = deviation*x_center
        y_deviation = deviation*y_center
        x_positions = [118+x_deviation, 256+x_deviation, 394+x_deviation]
        y_positions = [118+y_deviation, 256+y_deviation, 394+y_deviation]
        
        ThreeD_positions = []
        for cx in x_positions:
            for cy in y_positions:
                x = cx
                y = cy
                local_depth = get_local_depth(depth_map_gpu, input_images[0], int(x), int(y))
                z=np.min([local_depth, max_depth])
                z=local_depth
                ThreeD_positions.append((x, y, z))

        
        
        np.random.seed(seed+1)
        queried_image, maskforball, control_images_forball = project_multiball(pipe_sds, ThreeD_positions, ev, frame, depth_map_gpu, manual_pose, manual_K, ball_radius = ball_radius, device=device, MLP=envmap, depth_ratio=depth_ratio, input_image = input_images[0], haj=i)
        control_images_forball = pipe_sds.control_image_processor.preprocess(control_images_forball, height=args.img_height, width=args.img_width).to(device=device, dtype=torch_dtype)
        
        control_image = control_images_forball
            
        predict_kwargs = {
            "prompt_embeds": prompt_embeds,
            "guidance_scale": args.guidance_scale,
            "control_image": control_image,
            "controlnet_conditioning_scale": args.control_scale,
            "control_images_forball": control_images_forball,

        }           
        if args.sds_space == 'image':

            multistep = round(t.item() * args.sds_multistep) # adaptive number of step
            #check device of input tensors
            if do_classifier_free_guidance:
                predict_kwargs["control_image"] = torch.cat([control_images_forball] * 2, dim=0)

           
            mask = pipe_sds.mask_processor.preprocess(maskforball, height=args.img_height, width=args.img_width).to(device=device, dtype=torch_dtype).detach()
            mask = mask.squeeze(0)
            input = input_images[0]
            input = pipe_sds.image_processor.preprocess(input, height=args.img_height, width=args.img_width).to(device=device, dtype=torch_dtype)
            noise = None
            
            _, aux = pipe_sds.loss_sds_image(
                queried_image.to(device=device, dtype=torch_dtype),
                mask,
                input,
                timestep,
                predict_kwargs,
                device,
                generator, 
                multistep=multistep,
                image_mask=mask.to(device=device, dtype=torch_dtype),
                noise=noise,
                lpips=lpips,
                prompts_embed = prompt_embeds,
                negative_prompt_embeds = negative_prompt_embeds,
                return_aux=True,
            )
            mask_bool = mask.bool().expand_as(queried_image)
            image_pred = aux["image_pred"].detach().float()

            mse_loss = 0.5*torch.nn.functional.mse_loss(image_pred[mask_bool], queried_image[mask_bool])
            mask_expanded = mask_bool.expand_as(image_pred)
            pred_masked = image_pred * mask_expanded
            target_masked = queried_image * mask_expanded
            lpips_loss = lpips(pred_masked, target_masked)
            base_loss =lpips_loss+0.1*mse_loss

        else:
            raise ValueError(f"Unknown space: {args.sds_space}")
        # weighting
        if args.sds_weighting == "uniform":
            w = 1.0
        elif args.sds_weighting == "sds":
            w = 1.0 - pipe_sds.scheduler.alphas_cumprod[timestep].to(device=device, dtype=torch_dtype)
        else:
            raise ValueError(f"Unknown weighting: {args.sds_weighting}")
        loss_dict = {"loss_sds": base_loss * w}
        total_loss = sum(loss_dict.values())/accumulation_steps
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(envmap.parameters(), max_norm=1.0)
        scheduler.step()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if i % 20 == 0:
            vis_dict = {}

            image_with_ball = pipe_sds.image_processor.postprocess(queried_image.detach())[0].resize((512, 512))
            vis_dict["image_input"] = image_with_ball
            
            if args.sds_space == 'image':
                image_pred = pipe_sds.image_processor.postprocess(aux["image_pred"].detach())[0].resize((512, 512))
                vis_dict["image_pred"] = image_pred
        
            if is_ipynb:
                clear_output()
                mediapy.show_images(list(vis_dict.values()))
    save_dir = save_loc
    os.makedirs(save_dir, exist_ok=True)
    torch.save(envmap, os.path.join(save_dir, "optimized_envmap.pth"))
    print("Optimized envmap saved to", save_dir)
    return frames_dict, envmap

def main():
    pass


if __name__ == "__main__":
    main()
