import torch
from diffusers import ControlNetModel, AutoencoderKL
from PIL import Image
import numpy as np
import os
from tqdm.auto import tqdm
from transformers import pipeline as transformers_pipeline

from .pipeline import CustomStableDiffusionControlNetInpaintPipeline
from .argument import SAMPLERS, VAE_MODELS, DEPTH_ESTIMATOR, get_control_signal_type
from .image_processor import (
    estimate_scene_depth,
    estimate_scene_normal,
    merge_normal_map,
    fill_depth_circular
)
from .ball_processor import get_ideal_normal_ball, crop_ball
import pickle

class NoWaterMark:
    def apply_watermark(self, *args, **kwargs):
        return args[0]

class ControlSignalGenerator():
    def __init__(self, sd_arch, control_signal_type, device):
        self.sd_arch = sd_arch
        self.control_signal_type = control_signal_type
        self.device = device

    def process_sd_depth(self, input_image, normal_ball=None, mask_ball=None, x=None, y=None, r=None, depth_mode="nearest"):
        if getattr(self, 'depth_estimator', None) is None:
            self.depth_estimator = transformers_pipeline("depth-estimation", device=self.device.index)

        control_image = estimate_scene_depth(input_image, depth_estimator=self.depth_estimator)
        xs = [x] if not isinstance(x, list) else x
        ys = [y] if not isinstance(y, list) else y
        rs = [r] if not isinstance(r, list) else r
        mask_balls = [mask_ball] if not isinstance(mask_ball, list) else mask_ball
        
        for x, y, r, mask in zip(xs, ys, rs, mask_balls):
            #print(f"depth at {x}, {y}, {r}")
            control_image = fill_depth_circular(control_image, x, y, r, depth_mode=depth_mode, mask=mask)
        return control_image

    def process_sdxl_depth(self, input_image, normal_ball=None, mask_ball=None, x=None, y=None, r=None, depth=None, depth_mode="nearest"):
        if getattr(self, 'depth_estimator', None) is None:
            self.depth_estimator = transformers_pipeline("depth-estimation", model=DEPTH_ESTIMATOR, device=self.device.index)

        control_image = estimate_scene_depth(input_image, depth_estimator=self.depth_estimator)
        xs = [x] if not isinstance(x, list) else x
        ys = [y] if not isinstance(y, list) else y
        rs = [r] if not isinstance(r, list) else r
        mask_balls = [mask_ball] if not isinstance(mask_ball, list) else mask_ball
        
        for x, y, r, mask in zip(xs, ys, rs, mask_balls):
            #print(f"depth at {x}, {y}, {r}")
            control_image = fill_depth_circular(control_image, x, y, r, depth=depth, depth_mode=depth_mode, mask=mask)
        return control_image

    def process_sd_normal(self, input_image, normal_ball, mask_ball, x, y, r=None, normal_ball_path=None, depth_mode="nearest"):
        if getattr(self, 'depth_estimator', None) is None:
            self.depth_estimator = transformers_pipeline("depth-estimation", model=DEPTH_ESTIMATOR, device=self.device.index)

        normal_scene = estimate_scene_normal(input_image, depth_estimator=self.depth_estimator)
        normal_image = merge_normal_map(normal_scene, normal_ball, mask_ball, x, y)
        normal_image = (normal_image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        control_image = Image.fromarray(normal_image)
        return control_image

    def __call__(self, *args, **kwargs):
        process_fn = getattr(self, f"process_{self.sd_arch}_{self.control_signal_type}", None)
        if process_fn is None:
            raise ValueError
        else:
            return process_fn(*args, **kwargs)
        

class BallInpainter():
    def __init__(self, pipeline, sd_arch, control_generator, disable_water_mask=True):
        self.pipeline = pipeline
        self.sd_arch = sd_arch
        self.control_generator = control_generator
        self.median = {}
        if disable_water_mask:
            self._disable_water_mask()

    def _disable_water_mask(self):
        if hasattr(self.pipeline, "watermark"):
            self.pipeline.watermark = NoWaterMark()
            print("Disabled watermasking")

    @classmethod
    def from_sd(cls, 
                model, 
                controlnet=None, 
                device=0, 
                sampler="unipc", 
                torch_dtype=torch.float16,
                disable_water_mask=True,
                offload=False
    ):
        if controlnet is not None:
            control_signal_type = get_control_signal_type(controlnet)
            controlnet = ControlNetModel.from_pretrained(controlnet, torch_dtype=torch.float16)
            pipe = CustomStableDiffusionControlNetInpaintPipeline.from_pretrained(
                model,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
            ).to(device)
            control_generator = ControlSignalGenerator("sd", control_signal_type, device=device)
        else:
            pipe = CustomStableDiffusionInpaintPipeline.from_pretrained(
                model,
                torch_dtype=torch_dtype,
            ).to(device)
            control_generator = None
        
        try:
            if torch_dtype==torch.float16 and device != torch.device("cpu"):
                pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
        pipe.set_progress_bar_config(disable=True)
        
        pipe.scheduler = SAMPLERS[sampler].from_config(pipe.scheduler.config)
        
        return BallInpainter(pipe, "sd", control_generator, disable_water_mask)
    @classmethod
    def from_sdxl_fullft(cls, 
                model, 
                controlnet=None, 
                device=0, 
                sampler="unipc", 
                torch_dtype=torch.float16,
                disable_water_mask=True,
                use_fixed_vae=True,
                offload=False
    ):
        vae = VAE_MODELS["sdxl"]
        vae = AutoencoderKL.from_pretrained(vae, torch_dtype=torch_dtype).to(device) if use_fixed_vae else None
        extra_kwargs = {"vae": vae} if vae is not None else {}
        
        if controlnet is not None:
            control_signal_type = get_control_signal_type(controlnet)
            controlnet = ControlNetModel.from_pretrained(
                controlnet,
                variant="fp16" if torch_dtype == torch.float16 else None,
                use_safetensors=True,
                torch_dtype=torch_dtype,
            ).to(device)
            pipe = CustomStableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                model,
                controlnet=controlnet,
                variant="fp16" if torch_dtype == torch.float16 else None,
                use_safetensors=True,
                torch_dtype=torch_dtype,
                **extra_kwargs,
            ).to(device)
            control_generator = ControlSignalGenerator("sdxl", control_signal_type, device=device)
        else:
            pipe = CustomStableDiffusionXLInpaintPipeline.from_pretrained(
                model,
                variant="fp16" if torch_dtype == torch.float16 else None,
                use_safetensors=True,
                torch_dtype=torch_dtype,
                **extra_kwargs,
            ).to(device)
            control_generator = None
        
        try:
            if torch_dtype==torch.float16 and device != torch.device("cpu"):
                pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
        
        if offload and device != torch.device("cpu"):
            pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=False)
        pipe.scheduler = SAMPLERS[sampler].from_config(pipe.scheduler.config)
        
        return BallInpainter(pipe, "sdxl", control_generator, disable_water_mask)
    @classmethod
    def from_sdxl(cls, 
                model, 
                controlnet=None, 
                device=0, 
                sampler="unipc", 
                torch_dtype=torch.float16,
                disable_water_mask=True,
                use_fixed_vae=True,
                offload=False
    ):
        vae = VAE_MODELS["sdxl"]
        vae = AutoencoderKL.from_pretrained(vae, torch_dtype=torch_dtype).to(device) if use_fixed_vae else None
        extra_kwargs = {"vae": vae} if vae is not None else {}
        
        if controlnet is not None:
            control_signal_type = get_control_signal_type(controlnet)
            controlnet = ControlNetModel.from_pretrained(
                controlnet,
                variant="fp16" if torch_dtype == torch.float16 else None,
                use_safetensors=True,
                torch_dtype=torch_dtype,
            ).to(device)
            pipe = CustomStableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                model,
                controlnet=controlnet,
                variant="fp16" if torch_dtype == torch.float16 else None,
                use_safetensors=True,
                torch_dtype=torch_dtype,
                **extra_kwargs,
            ).to(device)
            control_generator = ControlSignalGenerator("sdxl", control_signal_type, device=device)
        else:
            pipe = CustomStableDiffusionXLInpaintPipeline.from_pretrained(
                model,
                variant="fp16" if torch_dtype == torch.float16 else None,
                use_safetensors=True,
                torch_dtype=torch_dtype,
                **extra_kwargs,
            ).to(device)
            control_generator = None
        
        try:
            if torch_dtype==torch.float16 and device != torch.device("cpu"):
                pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
        
        if offload and device != torch.device("cpu"):
            pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=False)
        pipe.scheduler = SAMPLERS[sampler].from_config(pipe.scheduler.config)
        
        return BallInpainter(pipe, "sdxl", control_generator, disable_water_mask)

    # TODO: this method should be replaced by inpaint(), but we'll leave it here for now
    # otherwise, the existing experiment code will break down
    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)

    def _default_height_width(self, height=None, width=None):
        if (height is not None) and (width is not None):
            return height, width
        if self.sd_arch == "sd":
            return (512, 512)
        elif self.sd_arch == "sdxl":
            return (1024, 1024)
        else:
            raise NotImplementedError

    # this method is for sanity check only
    def get_cache_control_image(self):
        control_image = getattr(self, "cache_control_image", None)
        return control_image

    def prepare_control_signal(self, image, controlnet_conditioning_scale, extra_kwargs):
        if self.control_generator is not None:
            if isinstance(image, list):
                print("yes")
                control_image = []
                for i in range(len(image)):
                    kwargs = {}
                    for k, v in extra_kwargs.items():
                        if isinstance(v, (list, tuple, np.ndarray, torch.Tensor)):
                            kwargs[k] = v[i] if i < len(v) else v[-1]
                        else:
                            kwargs[k] = v
                    control_out = self.control_generator(image[i], **kwargs)
                    control_image.append(control_out)
            else:
                print("hahahano")
                control_image = self.control_generator(image, **extra_kwargs)
            
            controlnet_kwargs = {
                "control_image": control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale
            }
            self.cache_control_image = control_image
        else:
            print("no kwargs")
            controlnet_kwargs = {}

        return controlnet_kwargs

    def get_cache_median(self, it):
        if it in self.median: return self.median[it]
        else: return None

    def reset_median(self):
        self.median = {}
        print("Reset median")

    def load_median(self, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.median = pickle.load(f)
                print(f"Loaded median from {path}")
        else:
            print(f"Median not found at {path}!")

    def inpaint_iterative(
        self,
        prompt=None,
        negative_prompt="",
        num_inference_steps=150,
        generator=None, # TODO: remove this
        image=None,
        mask_image=None,
        height=None,
        width=None,
        controlnet_conditioning_scale=0.5,
        num_images_per_prompt=1,
        current_seed=0,
        cross_attention_kwargs={},
        strength=0.8,
        num_iteration=2,
        ball_per_iteration=30,
        agg_mode="median",
        save_intermediate=False,
        cache_dir="./temp_inpaint_iterative",
        disable_progress=False,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        use_cache_median=False,
        **extra_kwargs,
    ):

        def computeMedian(ball_images):
            all = np.stack(ball_images, axis=0)
            median = np.median(all, axis=0)
            idx_median = np.argsort(all, axis=0)[all.shape[0]//2]
            # print(all.shape)
            # print(idx_median.shape)
            return median, idx_median

        def generate_balls(avg_image, current_strength, ball_per_iteration, current_iteration):
            print(f"Inpainting balls for {current_iteration} iteration...")
            controlnet_kwargs = self.prepare_control_signal(
                image=avg_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                extra_kwargs=extra_kwargs,
            )

            ball_images = []
            for i in tqdm(range(ball_per_iteration), disable=disable_progress):
                seed = current_seed + i
                new_generator = torch.Generator().manual_seed(seed)
                
                output_image = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    generator=new_generator,
                    image=avg_image,
                    mask_image=mask_image,
                    height=height,
                    width=width,
                    num_images_per_prompt=num_images_per_prompt,
                    strength=current_strength,
                    newx=x,
                    newy=y,
                    newr=r,
                    current_seed=seed,
                    cross_attention_kwargs=cross_attention_kwargs,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    **controlnet_kwargs
                ).images[0]
                
                # ball_image = crop_ball(output_image, mask_ball_for_crop, x, y, r)
                ball_image = np.asarray(output_image)
                ball_images.append(ball_image)

                if save_intermediate:
                    os.makedirs(os.path.join(cache_dir, str(current_iteration)), mode=0o777, exist_ok=True)
                    output_image.save(os.path.join(cache_dir, str(current_iteration), f"raw_{i}.png"))
                    Image.fromarray(ball_image).save(os.path.join(cache_dir, str(current_iteration), f"ball_{i}.png"))
                    # chmod 777
                    os.chmod(os.path.join(cache_dir, str(current_iteration), f"raw_{i}.png"), 0o0777)
                    os.chmod(os.path.join(cache_dir, str(current_iteration), f"ball_{i}.png"), 0o0777)

            
            return ball_images

        if save_intermediate:
            os.makedirs(cache_dir, exist_ok=True)

        height, width = self._default_height_width(height, width)

        x = extra_kwargs["x"]
        y = extra_kwargs["y"]
        r = 256  if "r" not in extra_kwargs else extra_kwargs["r"]
        # _, mask_ball_for_crop = get_ideal_normal_ball(size=r)
        
        # generate initial average ball
        avg_image = image
        ball_images = generate_balls(
            avg_image,
            current_strength=1.0,
            ball_per_iteration=ball_per_iteration,
            current_iteration=0,
        )

        # ball refinement loop
        image = np.array(image)
        for it in range(1, num_iteration+1):
            if use_cache_median and (self.get_cache_median(it) is not None):
                print("Use existing median")
                all = np.stack(ball_images, axis=0)
                idx_median = self.get_cache_median(it)
                avg_ball = all[idx_median, 
                    np.arange(idx_median.shape[0])[:, np.newaxis, np.newaxis],
                    np.arange(idx_median.shape[1])[np.newaxis, :, np.newaxis],
                    np.arange(idx_median.shape[2])[np.newaxis, np.newaxis, :]
                ]
            else:
                avg_ball, idx_median = computeMedian(ball_images)
                print("Add new median")
                self.median[it] = idx_median
            
            # avg_image = merge_normal_map(image, avg_ball, mask_ball_for_crop, x, y)
            avg_image = np.where(np.array(mask_image)[..., None].repeat(3, axis=-1), avg_ball, image)
            avg_image = Image.fromarray(avg_image.astype(np.uint8))
            if save_intermediate:
                avg_image.save(os.path.join(cache_dir, f"average_{it}.png"))
                # chmod777
                os.chmod(os.path.join(cache_dir, f"average_{it}.png"), 0o0777)
            
            ball_images = generate_balls(
                avg_image,
                current_strength=strength,
                ball_per_iteration=ball_per_iteration if it < num_iteration else 1,
                current_iteration=it,
            )

        # TODO: add algorithm for select the best ball
        best_ball = ball_images[0]
        # output_image = merge_normal_map(image, best_ball, mask_ball_for_crop, x, y)
        output_image = np.where(np.array(mask_image)[..., None].repeat(3, axis=-1), best_ball, image)
        return Image.fromarray(output_image.astype(np.uint8))

    def inpaint(
        self,
        prompt=None,
        negative_prompt=None,
        num_inference_steps=100,
        generator=None,
        image=None,
        mask_image=None,
        height=None,
        width=None,
        projection=None,
        controlnet_conditioning_scale=0.5,
        num_images_per_prompt=1,
        strength=1.0,
        current_seed=0,
        cross_attention_kwargs={},
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        multifusion=False,
        posed_image=True,
        control_image=None,
        multifusion_kwargs={},
        **extra_kwargs,
    ):  
        print("the strength i dasdasdass", strength)
        height, width = self._default_height_width(height, width)
        
        if control_image is not None:
            control_image = control_image
            controlnet_kwargs = {
                "control_image": control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale
            }
        if projection == "ortho":
            print("prepare control signal")
            controlnet_kwargs = self.prepare_control_signal(
                image=image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                extra_kwargs=extra_kwargs,
            )
     
        if generator is None:
            generator = torch.Generator().manual_seed(0)

        output_image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            image=image,
            mask_image=mask_image,
            height=height,
            width=width,
            num_images_per_prompt=num_images_per_prompt,
            strength=strength,
            newx = extra_kwargs["x"],
            newy = extra_kwargs["y"],
            newr = getattr(extra_kwargs, "r", 256), # default to ball_size = 256
            current_seed=current_seed,
            cross_attention_kwargs=cross_attention_kwargs,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            # multifusion=multifusion,
            # multifusion_kwargs=multifusion_kwargs,
            **controlnet_kwargs
        )

        return output_image, controlnet_kwargs["control_image"]
    
    def inpaint_iterative_multiview(
        self,
        prompt=None,
        negative_prompt="",
        num_inference_steps=100,
        generator=None, # TODO: remove this
        image=None,
        mask_image=None,
        height=None,
        width=None,
        controlnet_conditioning_scale=0.5,
        num_images_per_prompt=1,
        current_seed=0,
        cross_attention_kwargs={},
        strength=0.8,
        num_iteration=2,
        ball_per_iteration=30,
        agg_mode="median",
        save_intermediate=True,
        cache_dir="./temp_inpaint_iterative",
        disable_progress=False,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        use_cache_median=False,
        multifusion_kwargs={},
        **extra_kwargs,
    ):
        from relighting.envmap_utils import ballimg2envmap, envmap2ballimg

        def computeMedian(ball_images):
            # ball_images [[view1, view2], [view1, view2], ...]
            # return [view1, view2, view3]
            n_balls, n_views = len(ball_images), len(ball_images[0])
            poses = multifusion_kwargs["poses"]
            envmap_size = 256

            envmap_median = []
            for v in range(n_views):
                per_view_median = np.median(
                    np.stack([ball_images[b][v] for b in range(n_balls)], axis=0),
                    axis=0
                )
                per_view_median = torch.tensor(per_view_median).float().unsqueeze(0).permute(0, 3, 1, 2).to(poses)
                view_envmap = ballimg2envmap(
                    per_view_median, envmap_size, pose=poses[v:v+1]
                ).permute(0, 2, 3, 1).detach().cpu().numpy()[0]
                envmap_median.append(view_envmap)
            envmap_median = np.median(np.stack(envmap_median, axis=0), axis=0)
            envmap_median = torch.tensor(envmap_median).float().unsqueeze(0).permute(0, 3, 1, 2).to(poses)

            median_images = []
            for v in range(n_views):
                projected_median = envmap2ballimg(
                    envmap_median, rs[v], poses[v:v+1]
                ).permute(0, 2, 3, 1).detach().cpu().numpy()[0]
                median_images.append(projected_median)
            
            return median_images

        def generate_balls(avg_image, current_strength, ball_per_iteration, current_iteration):
            print(f"Inpainting balls for {current_iteration} iteration...")
            controlnet_kwargs = self.prepare_control_signal(
                image=avg_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                extra_kwargs=extra_kwargs,
            )

            ball_images = []
            for i in tqdm(range(ball_per_iteration), disable=disable_progress):
                seed = current_seed + i
                new_generator = torch.Generator().manual_seed(seed)

                output_image = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    generator=new_generator,
                    image=avg_image,
                    mask_image=mask_image,
                    height=height,
                    width=width,
                    num_images_per_prompt=num_images_per_prompt,
                    strength=current_strength,
                    newx=xs,
                    newy=ys,
                    newr=rs,
                    current_seed=seed,
                    cross_attention_kwargs=cross_attention_kwargs,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    **controlnet_kwargs
                ).images
                
                # ball_image = crop_ball(output_image, mask_ball_for_crop, x, y, r)
                ball_image = [
                    crop_ball(output_image[j], mask_ball_for_crop_list[j], xs[j], ys[j], rs[j]) for j in range(len(output_image))
                ]
                ball_images.append(ball_image)

                if save_intermediate:
                    os.makedirs(os.path.join(cache_dir, str(current_iteration)), exist_ok=True)
                    for j in range(len(output_image)):
                        output_image[j].save(os.path.join(cache_dir, str(current_iteration), f"raw_{i}_{j}.png"))
                        Image.fromarray(ball_image[j]).save(os.path.join(cache_dir, str(current_iteration), f"ball_{i}_{j}.png"))

            # ball_images = np.stack(ball_images, axis=0) # n_balls, n_views, h, w, 3
            
            return ball_images

        if save_intermediate:
            os.makedirs(cache_dir, exist_ok=True)

        height, width = self._default_height_width(height, width)

        xs = extra_kwargs["x"]
        ys = extra_kwargs["y"]
        rs = 256  if "r" not in extra_kwargs else extra_kwargs["r"]
        mask_ball_for_crop_list = [
            get_ideal_normal_ball(size=r)[1] for r in rs
        ]
        # _, mask_ball_for_crop = get_ideal_normal_ball(size=r)
        
        # generate initial average ball
        avg_image = image
        ball_images = generate_balls(
            avg_image,
            current_strength=1.0,
            ball_per_iteration=ball_per_iteration,
            current_iteration=0,
        )

        # ball refinement loop
        # image = np.array(image)
        max_strength = strength
        min_strength = 0.3
        for it in range(1, num_iteration+1):
            if use_cache_median and (self.get_cache_median(it) is not None):
                print("Use existing median")
                avg_ball = self.get_cache_median(it)
            else:
                avg_ball = computeMedian(ball_images)
                print("Add new median")
                self.median[it] = avg_ball

            avg_image = []
            for i in range(len(image)):
                avg_image_i = merge_normal_map(
                    np.array(image[i]), avg_ball[i], mask_ball_for_crop_list[i], xs[i], ys[i]
                )
                avg_image_i = Image.fromarray(avg_image_i.astype(np.uint8))
                avg_image.append(avg_image_i)
            # avg_image = merge_normal_map(image, avg_ball, mask_ball_for_crop, x, y)
            # avg_image = Image.fromarray(avg_image.astype(np.uint8))
            if save_intermediate:
                for i in range(len(avg_image)):
                    avg_image[i].save(os.path.join(cache_dir, f"average_{it}_{i}.png"))
                
            current_strength = max_strength - (max_strength - min_strength) * ((it - 1) / (num_iteration - 1))
            ball_images = generate_balls(
                avg_image,
                current_strength=current_strength,
                ball_per_iteration=ball_per_iteration if it < num_iteration else 1,
                current_iteration=it,
            )

        # TODO: add algorithm for select the best ball
        best_ball = ball_images[0]
        output_images = []
        for i in range(len(best_ball)):
            out_image_i = merge_normal_map(
                np.array(image[i]), best_ball[i], mask_ball_for_crop_list[i], xs[i], ys[i]
            )
            out_image_i = Image.fromarray(out_image_i.astype(np.uint8))
            output_images.append(out_image_i)
        return output_images

        # output_image = merge_normal_map(image, best_ball, mask_ball_for_crop, x, y)
        # return Image.fromarray(output_image.astype(np.uint8))

