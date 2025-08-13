# inpaint the ball on an image
# this one is design for general image that does not require special location to place 


import torch
import argparse
import numpy as np
import torch.distributed as dist
import os
from PIL import Image
from tqdm.auto import tqdm
import json


from .inpainter import BallInpainter

from .mask_utils import MaskGenerator
from .ball_processor import (
    get_ideal_normal_ball,
    crop_ball
)
from .dataset import GeneralLoader, PosedImagesLoader
from .utils import name2hash
from .camera_utils import project_points, project_sphere
from .envmap_utils import ballimg2envmap, aggregate_multiview_ballimg_to_envmap
from Lighting import dist_utils as dist_util
import time


# cross import from inpaint_multi-illum.py
from .argument import (
    SD_MODELS, 
    CONTROLNET_MODELS,
    VAE_MODELS
)

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True ,help='directory that contain the image') #dataset name or directory 
    parser.add_argument("--n_img", type=int, default=4, help="")
    parser.add_argument("--depth_mode", type=str, default="nearest" ,help='depth value for the ball')
    parser.add_argument("--obj_loc", type=float, nargs=3, default=None, help="location of the object in the scene")
    parser.add_argument("--obj_r", type=float, default=-1, help="radius of the object in the scene")
    parser.add_argument("--ball_x", type=int, default=-1, help="location x of the ball in pixels")
    parser.add_argument("--ball_y", type=int, default=-1, help="location y of the ball in pixels")
    parser.add_argument("--ball_size", type=int, default=256, help="size of the ball in pixel")
    parser.add_argument("--ball_dilate", type=int, default=20, help="How much pixel to dilate the ball to make a sharper edge")
    parser.add_argument("--prompt", type=str, default="a perfect mirrored reflective chrome ball sphere") 
    parser.add_argument("--prompt_dark", type=str, default="a perfect black dark mirrored reflective chrome ball sphere") 
    parser.add_argument("--negative_prompt", type=str, default="matte, diffuse, flat, dull") 
    parser.add_argument("--model_option", default="sdxl", help='selecting fancy model option (sd15_old, sd15_new, sd21, sdxl)') # [sd15_old, sd15_new, or sd21]
    parser.add_argument("--output_dir", required=True, type=str, help="output directory")
    parser.add_argument("--img_height", type=int, default=1024, help="Dataset Image Height")
    parser.add_argument("--img_width", type=int, default=1024, help="Dataset Image Width")
    # some good seed 0, 37, 71, 125, 140, 196, 307, 434, 485, 575 | 9021, 9166, 9560, 9814, but default auto is for fairness
    parser.add_argument("--seed", default="auto", type=str, help="Seed: right now we use single seed instead to reduce the time, (Auto will use hash file name to generate seed)")
    parser.add_argument("--denoising_step", default=30, type=int, help="number of denoising step of diffusion model")
    parser.add_argument("--control_scale", default=0.5, type=float, help="controlnet conditioning scale")
    
    parser.add_argument('--no_controlnet', dest='use_controlnet', action='store_false', help='by default we using controlnet, we have option to disable to see the different')
    parser.set_defaults(use_controlnet=True)
    
    parser.add_argument('--no_force_square', dest='force_square', action='store_false', help='SDXL is trained for square image, we prefered the square input. but you use this option to disable reshape')
    parser.set_defaults(force_square=True)
    
    parser.add_argument('--no_random_loader', dest='random_loader', action='store_false', help="by default, we random how dataset load. This make us able to peak into the trend of result without waiting entire dataset. but can disable if prefereed")
    parser.set_defaults(random_loader=True)

    parser.add_argument('--cpu', dest='is_cpu', action='store_true', help="using CPU inference instead of GPU inference")
    parser.set_defaults(is_cpu=False)

    parser.add_argument('--offload', dest='offload', action='store_false', help="to enable diffusers cpu offload")
    parser.set_defaults(offload=False)
    
    parser.add_argument("--limit_input", default=0, type=int, help="limit number of image to process to n image (0 = no limit), useful for run smallset")

    parser.add_argument('--no_multifusion', dest='multifusion', action='store_false')
    parser.set_defaults(multifusion=True)
    parser.add_argument("--sampler", default="unipc", type=str)
    parser.add_argument("--merge_varname", default="zt", type=str)
    parser.add_argument("--merge_style", default="weighted_avg", type=str)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=29, type=int)

    # LoRA stuff
    parser.add_argument('--no_lora', dest='use_lora', action='store_false', help='by default we using lora, we have option to disable to see the different')
    parser.set_defaults(use_lora=True)

    parser.add_argument("--lora_path", default="models/ThisIsTheFinal-lora-hdr-continuous-largeT@900/0_-5/checkpoint-2500", type=str, help="LoRA Checkpoint path")
    parser.add_argument("--lora_scale", default=0.75, type=float, help="LoRA scale factor")

    # speed optimization stuff
    parser.add_argument('--no_torch_compile', dest='use_torch_compile', action='store_false', help='by default we using torch compile for faster processing speed. disable it if your environemnt is lower than pytorch2.0')
    parser.set_defaults(use_torch_compile=True)
    
    # algorithm + iterative stuff
    parser.add_argument("--algorithm", type=str, default="iterative", choices=["iterative", "normal"], help="Selecting between iterative or normal (single pass inpaint) algorithm")

    parser.add_argument("--agg_mode", default="median", type=str)
    parser.add_argument("--final_agg_mode", default="median", type=str)
    parser.add_argument("--strength", default=0.8, type=float)
    parser.add_argument("--num_iteration", default=2, type=int)
    parser.add_argument("--ball_per_iteration", default=30, type=int)
    parser.add_argument('--no_save_intermediate', dest='save_intermediate', action='store_false')
    parser.set_defaults(save_intermediate=True)
    parser.add_argument("--cache_dir", default="./temp_inpaint_iterative", type=str, help="cache directory for iterative inpaint")
    
    # pararelle processing
    parser.add_argument("--idx", default=0, type=int, help="index of the current process, useful for running on multiple node")
    parser.add_argument("--total", default=1, type=int, help="total number of process")

    # for HDR stuff
    parser.add_argument("--max_negative_ev", default=-5, type=int, help="maximum negative EV for lora")
    parser.add_argument("--ev", default="0,-2.5,-5", type=str, help="EV: list of EV to generate")

    return parser

def get_ball_location(image_data, args):
    if 'boundary' in image_data:
        # support predefined boundary if need
        x = image_data["boundary"]["x"]
        y = image_data["boundary"]["y"]
        r = image_data["boundary"]["size"]
        
        # support ball dilation
        half_dilate = args.ball_dilate // 2

        # check if not left out-of-bound
        if x - half_dilate < 0: x += half_dilate
        if y - half_dilate < 0: y += half_dilate

        # check if not right out-of-bound
        if x + r + half_dilate > args.img_width: x -= half_dilate
        if y + r + half_dilate > args.img_height: y -= half_dilate   
    elif args.obj_loc is not None:
        assert 'projection' in image_data, "projection matrix is required for obj_loc"
        obj_2d, radius_2d = project_sphere(np.array(args.obj_loc), args.obj_r, image_data['pose'], image_data['K'])
        if radius_2d is None:
            r = args.ball_size
        else:
            r = radius_2d * 2
        x, y = round(obj_2d[0] - (r // 2)), round(obj_2d[1] - (r // 2))
    elif args.ball_x > 0 and args.ball_y > 0:
        # use user defined ball location
        x, y, r = args.ball_x - (args.ball_size // 2), args.ball_y - (args.ball_size // 2), args.ball_size
            
    else:
        # we use top-left corner notation
        x, y, r = ((args.img_width // 2) - (args.ball_size // 2), (args.img_height // 2) - (args.ball_size // 2), args.ball_size)
    return x, y, r

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
    prompt_embeds_normal, _, pooled_prompt_embeds_normal, _ = pipe.pipeline.encode_prompt(prompt_normal)
    prompt_embeds_dark, _, pooled_prompt_embeds_dark, _ = pipe.pipeline.encode_prompt(prompt_dark)

    # interpolate embeddings
    interpolate_embeds = []
    for t in interpolants:
        int_prompt_embeds = prompt_embeds_normal + t * (prompt_embeds_dark - prompt_embeds_normal)
        int_pooled_prompt_embeds = pooled_prompt_embeds_normal + t * (pooled_prompt_embeds_dark - pooled_prompt_embeds_normal)

        interpolate_embeds.append((int_prompt_embeds, int_pooled_prompt_embeds))

    return dict(zip(ev_list, interpolate_embeds))

def main():
    # load arguments
    args = create_argparser().parse_args()
        
    # get local rank
    if args.is_cpu:
        device = torch.device("cpu")
        torch_dtype = torch.float32
    else:
        device = dist_util.dev()
        torch_dtype = torch.float16
    
    # so, we need ball_dilate >= 16 (2*vae_scale_factor) to make our mask shape = (272, 272)
    assert args.ball_dilate % 2 == 0 # ball dilation should be symmetric
    
    # create controlnet pipeline 
    if args.model_option in ["sdxl", "sdxl_fast"] and args.use_controlnet:
        model, controlnet = SD_MODELS[args.model_option], CONTROLNET_MODELS[args.model_option]
        pipe = BallInpainter.from_sdxl(
            model=model, 
            controlnet=controlnet, 
            sampler=args.sampler,
            device=device,
            torch_dtype = torch_dtype,
            offload = args.offload
        )
    elif args.model_option in ["sdxl", "sdxl_fast"] and not args.use_controlnet:
        model = SD_MODELS[args.model_option]
        pipe = BallInpainter.from_sdxl(
            model=model,
            controlnet=None,
            sampler=args.sampler,
            device=device,
            torch_dtype = torch_dtype,
            offload = args.offload
        )
    elif args.use_controlnet:
        model, controlnet = SD_MODELS[args.model_option], CONTROLNET_MODELS[args.model_option]
        pipe = BallInpainter.from_sd(
            model=model,
            controlnet=controlnet,
            device=device,
            torch_dtype = torch_dtype,
            offload = args.offload
        )
    else:
        model = SD_MODELS[args.model_option]
        pipe = BallInpainter.from_sd(
            model=model,
            controlnet=None,
            device=device,
            torch_dtype = torch_dtype,
            offload = args.offload
        )

    
    if args.lora_scale > 0 and args.lora_path is None:
        raise ValueError("lora scale is not 0 but lora path is not set")
    
    if (args.lora_path is not None) and (args.use_lora):
        print(f"using lora path {args.lora_path}")
        print(f"using lora scale {args.lora_scale}")
        pipe.pipeline.load_lora_weights(args.lora_path)
        pipe.pipeline.fuse_lora(lora_scale=args.lora_scale) # fuse lora weight w' = w + \alpha \Delta w
        enabled_lora = True
    else:
        enabled_lora = False

    if args.use_torch_compile:
        try:
            print("compiling unet model")
            start_time = time.time()
            pipe.pipeline.unet = torch.compile(pipe.pipeline.unet, mode="reduce-overhead", fullgraph=True)
            print("Model compilation time: ", time.time() - start_time)
        except:
            pass
                
    # default height for sdxl is 1024, if not set, we set default height.
    if args.model_option == "sdxl" and args.img_height == 0 and args.img_width == 0:
        args.img_height = 1024
        args.img_width = 1024
          
    # load dataset
    dataset = PosedImagesLoader(
        root=args.dataset,
        num_samples=args.n_img,
        resolution=(args.img_width, args.img_height),
        # force_square=args.force_square,
        return_dict=True,
        # random_shuffle=args.random_loader,
        # process_id=args.idx,
        # process_total=args.total,
        # limit_input=args.limit_input,
    )

    # interpolate embedding
    embedding_dict = interpolate_embedding(pipe, args)
    
    # prepare mask and normal ball
    # mask_generator = MaskGenerator()
    # normal_ball, mask_ball = get_ideal_normal_ball(size=args.ball_size+args.ball_dilate)
    # _, mask_ball_for_crop = get_ideal_normal_ball(size=args.ball_size)
    
    xs, ys, rs = [], [], []
    for image_data in dataset:
        x, y, r = get_ball_location(image_data, args)
        xs.append(x)
        ys.append(y)
        rs.append(r)
    depth_modes = [args.depth_mode for _ in range(len(dataset))]
    
    # make output directory if not exist
    raw_output_dir = os.path.join(args.output_dir, "raw")
    control_output_dir = os.path.join(args.output_dir, "control")
    square_output_dir = os.path.join(args.output_dir, "square")
    debug_output_dir = os.path.join(args.output_dir, "debug")
    envmap_output_dir = os.path.join(args.output_dir, "envmap")
    os.makedirs(args.output_dir, exist_ok=True)    
    os.makedirs(raw_output_dir, exist_ok=True)
    os.makedirs(control_output_dir, exist_ok=True)
    os.makedirs(square_output_dir, exist_ok=True)
    os.makedirs(debug_output_dir, exist_ok=True)
    os.makedirs(envmap_output_dir, exist_ok=True)
    
    # create split seed
    # please DO NOT manual replace this line, use --seed option instead
    seeds = args.seed.split(",")
    
    input_images = [image_data["image"] for image_data in dataset]
    image_paths = [image_data["path"] for image_data in dataset]
    input_poses = torch.tensor(
        np.stack([image_data["pose"] for image_data in dataset], axis=0),
        dtype=torch_dtype, device=device
    )
    # if args.n_img > 0:
    #     input_images = input_images[:args.n_img]
    #     image_paths = image_paths[:args.n_img]

    multifusion_kwargs = {
        'poses': input_poses,
        'merge_varname': args.merge_varname,
        'merge_style': args.merge_style,
        'start_iter': args.start_iter,
        'end_iter': args.end_iter,
        'debug_dir': debug_output_dir,
    }
        
    for ev, (prompt_embeds, pooled_prompt_embeds) in embedding_dict.items():
        # create output file name (we always use png to prevent quality loss)
        ev_str = str(ev).replace(".", "") if ev != 0 else "-00"
        outname = os.path.basename(args.dataset).split(".")[0]

        # we use top-left corner notation (which is different from aj.aek's center point notation)
        # x_center = x + args.ball_size // 2
        # y_center = y + args.ball_size // 2
        # outname += f"_x{x_center}_y{y_center}_r{r}_depth-{args.depth_mode}"
        outname += f"_ev{ev_str}"
        
        # create inpaint mask
        masks = []
        for image_data in dataset:
            x, y, r = get_ball_location(image_data, args)
            x = x - (args.ball_dilate // 2)
            y = y - (args.ball_dilate // 2)
            r = r + args.ball_dilate
            mask_generator = MaskGenerator()
            normal_ball, mask_ball = get_ideal_normal_ball(size=r)
            mask = mask_generator.generate_single(
                image_data["image"], mask_ball, x, y, r
            )
            masks.append(mask)
            
        seeds = tqdm(seeds, desc="seeds") if len(seeds) > 10 else seeds   
            
        #replacely create image with differnt seed
        for seed in seeds:
            start_time = time.time()
            # set seed, if seed auto we use file name as seed
            if seed == "auto":
                filename = os.path.basename(image_paths[0]).split(".")[0]
                seed = name2hash(filename) 
                # outpng = f"{outname}.png"
                cache_name = f"{outname}"
            else:
                seed = int(seed)
                # outpng = f"{outname}_seed{seed}.png"
                cache_name = f"{outname}_seed{seed}"
            # skip if file exist, useful for resuming
            # if os.path.exists(os.path.join(square_output_dir, outpng)):
            #     continue
            generator = torch.Generator().manual_seed(seed)
            kwargs = {
                "prompt_embeds": prompt_embeds.expand(len(input_images), -1, -1),
                "pooled_prompt_embeds": pooled_prompt_embeds.expand(len(input_images), -1),
                'negative_prompt': [args.negative_prompt] * len(input_images),
                'num_inference_steps': args.denoising_step,
                'generator': generator,
                'image': input_images,
                'mask_image': masks,
                'strength': 1.0,
                'current_seed': seed, # we still need seed in the pipeline!
                'controlnet_conditioning_scale': args.control_scale,
                'height': args.img_height,
                'width': args.img_width,
                'normal_ball': [normal_ball] * len(input_images),
                'mask_ball': [mask_ball] * len(input_images),
                'x': xs,
                'y': ys,
                'r': rs,
                'depth_mode': depth_modes,
            }
            
            if enabled_lora:
                kwargs["cross_attention_kwargs"] = {"scale": args.lora_scale}
            
            if args.algorithm == "normal":
                kwargs.update({
                    'multifusion': args.multifusion,
                    'multifusion_kwargs': multifusion_kwargs,
                })
                output_images = pipe.inpaint(**kwargs).images # [0]
            elif args.algorithm == "iterative":
                # This is still buggy
                print("using inpainting iterative, this is going to take a while...")
                kwargs.update({
                    "strength": args.strength,
                    "num_iteration": args.num_iteration,
                    "ball_per_iteration": args.ball_per_iteration,
                    "agg_mode": args.agg_mode,
                    "save_intermediate": args.save_intermediate,
                    "cache_dir": debug_output_dir,
                    'multifusion_kwargs': multifusion_kwargs,
                })
                output_images = pipe.inpaint_iterative_multiview(**kwargs)
            else:
                raise NotImplementedError(f"Unknown algorithm {args.algorithm}")
                
            
            square_images = []
            for i in range(len(input_images)):
                output_image = output_images[i]
                square_image = output_image.crop((xs[i], ys[i], xs[i]+rs[i], ys[i]+rs[i]))
                square_images.append(
                    torch.tensor(np.array(square_image) / 255.0, dtype=torch_dtype, device=device).permute(2, 0, 1)
                )

                outpng = outname.replace("_ev", f"_{i:03d}_ev") + ".png"

                # return the most recent control_image for sanity check
                control_image = pipe.get_cache_control_image()[i]
                if control_image is not None:
                    control_image.save(os.path.join(control_output_dir, outpng))
                
                # save image 
                output_image.save(os.path.join(raw_output_dir, outpng))
                square_image.save(os.path.join(square_output_dir, outpng))

            # aggregate all multi-view results
            # square_images = torch.stack(square_images, dim=0)
            envmap_final = aggregate_multiview_ballimg_to_envmap(square_images, input_poses, 256, args.final_agg_mode)
            envmap_final = Image.fromarray((envmap_final.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))
            outpng = outname + ".png"
            envmap_final.save(os.path.join(envmap_output_dir, outpng))

                          
if __name__ == "__main__":
    main()