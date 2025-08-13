import time
import random
import math
import torch
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
import bisect

from .argument import SD_MODELS, VAE_MODELS, CONTROLNET_MODELS, SAMPLERS
from .sds_utils import PipeSDS21
from .envmap_utils import envmap2ballimg_perspective, hdr2ldr, ldr2hdr, get_saturated_mask


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_models(args, device, torch_dtype):
    #model, controlnet = SD_MODELS[args.model_option], CONTROLNET_MODELS["sd21"]
    controlnet = CONTROLNET_MODELS[args.model_option]
    print("using controlnet: ", controlnet)
    model = args.ckpt_dir
    print("loading model from sd21")
    pipe_sds, control_generator = PipeSDS21.from_sd21(
        model=model, 
        controlnet=controlnet,  
        sampler=args.sampler,
        device=device,
        torch_dtype = torch_dtype,
        offload = args.offload
    )
    print("dtype of unet: ", torch_dtype)

    enabled_lora = False
    

    if args.use_torch_compile:
        try:
            print("compiling unet model")
            start_time = time.time()
            pipe_sds.unet = torch.compile(pipe_sds.unet, mode="reduce-overhead", fullgraph=True)
            print("Model compilation time: ", time.time() - start_time)
        except:
            print("Model compilation failed")

    return pipe_sds, control_generator, enabled_lora

def load_modelsxl(args, device, torch_dtype):
    model, controlnet = SD_MODELS[args.model_option], CONTROLNET_MODELS[args.model_option]
    pipe_sds, control_generator = PipeSDS.from_sdxl(
        model=model, 
        controlnet=controlnet, 
        sampler=args.sampler,
        device=device,
        torch_dtype = torch_dtype,
        offload = args.offload
    )

    enabled_lora = False
    if (args.lora_path is not None) and (args.use_lora):
        print(f"using lora path {args.lora_path}")
        print(f"using lora scale {args.lora_scale}")
        pipe_sds.load_lora_weights(args.lora_path)
        pipe_sds.fuse_lora(lora_scale=args.lora_scale) # fuse lora weight w' = w + \alpha \Delta w
        enabled_lora = True

    if args.use_torch_compile:
        try:
            print("compiling unet model")
            start_time = time.time()
            pipe_sds.unet = torch.compile(pipe_sds.unet, mode="reduce-overhead", fullgraph=True)
            print("Model compilation time: ", time.time() - start_time)
        except:
            print("Model compilation failed")

    return pipe_sds, control_generator, enabled_lora
def get_interpolated_prompt_embeds_sdxl(pipe, ev, max_negative_ev, device, args):
    #print("interpolate embedding...")
    #print("dtype of prompt and prompt dark: ", args.prompt.dtype, args.prompt_dark.dtype)
    prompt_embeds_normal, _, pooled_prompt_embeds_normal, _ = pipe.encode_prompt(args.prompt)
    prompt_embeds_dark, _, pooled_prompt_embeds_dark, _ = pipe.encode_prompt(args.prompt_dark)
    t = ev / max_negative_ev
    negative_prompts, _, pooled_negative_prompts, _ = pipe.encode_prompt(args.negative_prompt)
    #print('exposure facotr: ', t)
    prompts_interpolated = prompt_embeds_normal + t * (prompt_embeds_dark - prompt_embeds_normal)
    pooled_interpolated = pooled_prompt_embeds_normal + t * (pooled_prompt_embeds_dark - pooled_prompt_embeds_normal)
    #print("shape of prompts_interpolated: ", prompts_interpolated.shape)
    #resize to [1, 77, 512] from 1, 77, 1024

    return (prompts_interpolated, pooled_interpolated, negative_prompts, pooled_negative_prompts)



def get_interpolated_prompt_embeds(pipe, ev, max_negative_ev, device, args):
    #print("interpolate embedding...")
    #print("dtype of prompt and prompt dark: ", args.prompt.dtype, args.prompt_dark.dtype)
    prompt_embeds= pipe.encode_prompt(args.prompt, device, num_images_per_prompt=1, do_classifier_free_guidance=True)[0]
    prompt_embeds_dark = pipe.encode_prompt(args.prompt_dark, device, num_images_per_prompt=1, do_classifier_free_guidance=True)[0]

    t = ev / max_negative_ev
    #print('exposure facotr: ', t)
    prompts_interpolated = prompt_embeds + t * (prompt_embeds_dark - prompt_embeds)
    #print("shape of prompts_interpolated: ", prompts_interpolated.shape)
    #resize to [1, 77, 512] from 1, 77, 1024

    return prompts_interpolated


def interpolate_embedding(pipe, args):
    print("interpolate embedding...")

    # get list of all EVs
    ev_list = [float(x) for x in args.ev.split(",")]
    interpolants = [ev / args.max_negative_ev for ev in ev_list]

    print("EV : ", ev_list)
    print("EV : ", interpolants)

    # calculate prompt embeddings
    prompt_normal = args.prompt
    prompt_dark = args.prompt_dark
    prompt_negative = args.negative_prompt
    prompt_embeds_normal, _, pooled_prompt_embeds_normal, _ = pipe.encode_prompt(prompt_normal)
    prompt_embeds_dark, _, pooled_prompt_embeds_dark, _ = pipe.encode_prompt(prompt_dark)
    prompt_embeds_negative, _, pooled_prompt_embeds_negative, _ = pipe.encode_prompt(prompt_negative)

    # interpolate embeddings
    interpolate_embeds = []
    for t in interpolants:
        int_prompt_embeds = prompt_embeds_normal + t * (prompt_embeds_dark - prompt_embeds_normal)
        int_pooled_prompt_embeds = pooled_prompt_embeds_normal + t * (pooled_prompt_embeds_dark - pooled_prompt_embeds_normal)

        interpolate_embeds.append(
            (int_prompt_embeds, int_pooled_prompt_embeds, prompt_embeds_negative, pooled_prompt_embeds_negative)
        )

    return dict(zip(ev_list, interpolate_embeds))


def get_mask_and_control_images_2d(dataset, controlgenerator,args):
    input_masks = []
    control_images = []
    ball_params = args.obj_loc + [args.obj_r]
    #cerate a circle mask for the ball
    mask = np.zeros((args.img_height, args.img_width), dtype=np.uint8)
    cv2.circle(mask, (args.obj_loc[0], args.obj_loc[1]), args.obj_r, 255, -1)
    mask = Image.fromarray(mask)
    return mask

def get_mask_and_control_images(dataset, control_generator, args):
    input_masks = []
    control_images = []
    ball_params = args.obj_loc + [args.obj_r]
    envmap = torch.ones(1, 128, 256).float()
    for i, image_data in enumerate(dataset):
        mask = envmap2ballimg_perspective(
            envmap, ball_params, torch.tensor(image_data["pose"]).float(), torch.tensor(image_data["K"]).float(),
            args.img_height, args.img_width
        ).bool().detach().cpu().numpy()[0]
        mask = mask > 0
        mask = Image.fromarray((mask * 255).astype(np.uint8))

        extra_control_kwargs = {
            'normal_ball': None, 'mask_ball': np.array(mask) > 0,
            'x': 0, 'y': 0, 'r': 0, 'depth_mode': args.depth_mode,
        }
        control_image = control_generator(image_data["image"], **extra_control_kwargs)
        input_masks.append(mask)
        control_images.append(control_image)
    return input_masks, control_images, ball_params

def calculate_dreamtime_weights(num_timesteps, alphas_cumprod, m=500, s=125):
    """Calculate DreamTime weights W(t) = Wd(t) * Wp(t)"""
    # Calculate DDPM weights (Wd)
    ddpm_weights = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    
    # Calculate Gaussian weights (Wp)
    gaussian_weights = []
    for t in range(num_timesteps):
        w = math.exp(-(t - m)**2 / (2 * s**2))
        gaussian_weights.append(w)
    gaussian_weights = torch.tensor(gaussian_weights, device=ddpm_weights.device)
    
    # Combine weights
    weights = ddpm_weights * gaussian_weights
    return weights / weights.sum()

def sds_t_sampling(i, t_min, t_max, t_strategy, n_total_iterations, alphas_cumprod=None):
    """
    Enhanced timestep sampling function supporting multiple strategies including DreamTime
    Returns:
        torch.Tensor: A single value tensor in range [0,1]
    """
    if t_strategy == "uniform":
        pass  # Keep original range
        
    elif t_strategy == "decay_t_min":
        t_min = max(t_min, (1 - (i / (n_total_iterations // 2))) * t_max)
        
    elif t_strategy == "decay_sqrt":
        t = t_max - (t_max - t_min) * math.sqrt(i / n_total_iterations)
        t_min, t_max = t, t + 0.001
        
    elif t_strategy == "dreamtime":
        if alphas_cumprod is None:
            raise ValueError("alphas_cumprod must be provided for dreamtime strategy")
            
        num_train_timesteps = len(alphas_cumprod)
        t_min_step = int(t_min * num_train_timesteps)
        t_max_step = int(t_max * num_train_timesteps)
            
        # Calculate weights using DreamTime method
        weights = calculate_dreamtime_weights(num_train_timesteps, alphas_cumprod)
        
        # Keep only the weights for our timestep range
        weights = weights[t_min_step : t_max_step + 1]
        weights = weights / torch.sum(weights)
        
        # Convert weights to cumsum for sampling
        weights_flip = weights.flip(dims=(0,))
        weights_cumsum = weights_flip.cumsum(dim=0).cpu().numpy()
        
        # Find appropriate timestep using bisect
        delta_timestep = bisect.bisect_left(weights_cumsum, i / n_total_iterations)
        timestep = max(t_max_step - delta_timestep, t_min_step)
        
        # Convert back to [0,1] range and set as new range
        t = timestep / num_train_timesteps
        t_min, t_max = t, t + 0.001

    else:
        raise ValueError(f"Unknown timestep strategy: {t_strategy}")
    
    # Final sampling step
    t = torch.rand(1, device=alphas_cumprod.device if alphas_cumprod is not None else 'cpu') * (t_max - t_min) + t_min
    return t
def get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_scale: float = 0.0
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    scale = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)) # 1 -> 0
    scale = scale * (1.0 - min_scale) + min_scale # 1 -> min_scale
    return scale


def get_optimizer(args):
    d = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
    }
    return d[args.sds_optimizer]


def generate_3d_projected_noise(args, latent_channels, ball_params, poses, Ks, device, torch_dtype):
    # generate noise in envmap space and project it to each view

    noise_envmap = torch.randn(
        1, latent_channels, args.envmap_height, args.envmap_width, device=device
    )
    batch_noise = []
    for idx in range(len(poses)):
        noise = torch.randn(
            1, latent_channels, args.img_height // 8, args.img_width // 8, device=device,
        )
        noise_ballimg = envmap2ballimg_perspective(
            noise_envmap[0], ball_params, poses[idx].float(), Ks[idx].float(),
            image_width=args.img_width // 8, image_height=args.img_height // 8, interp_mode='nearest'
        ).unsqueeze(0)
        # noise_ballimg = torch.nn.functional.interpolate(
        #     noise_ballimg, size=(args.img_height // 8, args.img_width // 8), mode="nearest",
        # )
        mask = (torch.sum(noise_ballimg, dim=1, keepdim=True) > 0).float()
        noise = noise * (1 - mask) + noise_ballimg * mask
        batch_noise.append(noise)
    batch_noise = torch.concat(batch_noise, dim=0).to(torch_dtype)
    return batch_noise

