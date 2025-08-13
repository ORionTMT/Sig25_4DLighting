import torch
import torchvision
import torch.nn.functional as F
from typing import List, Union, Dict, Any, Callable, Optional, Tuple
import numpy as np
from diffusers import ControlNetModel, AutoencoderKL
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.models import ControlNetModel
from diffusers import DDIMScheduler, DDPMScheduler, UniPCMultistepScheduler
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers import StableDiffusionXLControlNetInpaintPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from .pipeline_utils import custom_prepare_latents, custom_prepare_mask_latents, rescale_noise_cfg
from .envmap_utils import ballimg2envmap, envmap2ballimg
from .pipeline import CustomStableDiffusionControlNetInpaintPipeline
from.inpainter import ControlSignalGenerator
from .argument import VAE_MODELS, get_control_signal_type
from .sds_scheduler import SAMPLERS
import lpips
from torch.cuda.amp import custom_bwd, custom_fwd 

# 初始化 LPIPS 损失模块（通常使用VGG作为特征提取器）


class PipeSDS21(CustomStableDiffusionControlNetInpaintPipeline):
    @staticmethod
    def from_sd21(
        model,
        controlnet,
        device=0,
        sampler="unipc",
        torch_dtype=torch.float16,
        offload=False,
    ):

        control_signal_type = get_control_signal_type(controlnet)
        controlnet = ControlNetModel.from_pretrained(
            controlnet
        ).to(device)
        pipe = PipeSDS21.from_pretrained(
            model,
            controlnet=controlnet,
            torch_dtype=torch_dtype
        ).to(device)
        control_generator = ControlSignalGenerator("sd", control_signal_type, device=device)



        if offload and device != torch.device("cpu"):
            pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=False)
        pipe.scheduler = SAMPLERS[sampler].from_config(pipe.scheduler.config)
        return pipe, control_generator
    class SpecifyGradient(torch.autograd.Function):
        @staticmethod
        @custom_fwd
        def forward(ctx, input_tensor, gt_grad):
            ctx.save_for_backward(gt_grad)
            # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
            return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_scale):
            gt_grad, = ctx.saved_tensors
            gt_grad = gt_grad * grad_scale
            return gt_grad, None

    @torch.no_grad()
    def prepare_sds(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        control_image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.5,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        newx: int = 0,
        newy: int = 0,
        newr: int = 256,
        current_seed=0,
        use_noise_moving=True,
        multifusion: bool = False,
        multifusion_kwargs: Optional[Dict[str, Any]] = None,
    ):
        ret = {}
        # OVERWRITE METHODS
        self.prepare_mask_latents = custom_prepare_mask_latents.__get__(self, CustomStableDiffusionControlNetInpaintPipeline)
        self.prepare_latents = custom_prepare_latents.__get__(self, CustomStableDiffusionControlNetInpaintPipeline)

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 1. Check inputs. Raise error if not correct
        '''self.check_inputs(
            prompt,
            control_image,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )'''

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        ret["guidance_scale"] = guidance_scale

        ret["controlnet_conditioning_scale"] = controlnet_conditioning_scale
        
        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        ret["cross_attention_kwargs"] = cross_attention_kwargs
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            #prompt_embeds = prompt_embeds

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            control_image = self.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []

            for control_image_ in control_image:
                control_image_ = self.prepare_control_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                control_images.append(control_image_)

            control_image = control_images
        else:
            assert False
        
        ret["control_image"] = control_image

        # 4. Preprocess mask and image - resizes image and mask w.r.t height and width
        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=torch.float32)

        mask = self.mask_processor.preprocess(mask_image, height=height, width=width)

        masked_image = init_image * (mask < 0.5)
        _, _, height, width = init_image.shape

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4
        #print("num_channels_unet", num_channels_unet)

        # EDITED HERE
        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
            newx=newx,
            newy=newy,
            newr=newr,
            current_seed=current_seed,
            use_noise_moving=use_noise_moving,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        ret["image_latents"] = image_latents

        # 7. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )
        ret["mask"] = mask

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []

        for i in range(len(timesteps)):
            
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
        
        ret["controlnet_keep"] = controlnet_keep

        return ret
    @torch.no_grad()
    def encode_image(self, image: PipelineImageInput, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None):
        image = self.image_processor.preprocess(image)
        image = image.to(device=self._execution_device, dtype=self.vae.dtype)
        image_latents = self._encode_vae_image(image=image, generator=generator)
        return image_latents

    @torch.no_grad()
    def decode_image(self, latents: torch.Tensor, output_type: str = 'pil'):
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
        return image
    
    @torch.no_grad()
    def encode_mask(self, mask):
        mask = self.mask_processor.preprocess(mask)
        height, width = mask.shape[-2:]
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor),
            mode="bilinear", align_corners=False #PURE: We add this to avoid sharp border of the ball
        )
        mask = mask.to(device=self._execution_device, dtype=self.vae.dtype)
        return mask
    @torch.no_grad()
    def weighted_perpendicular_aggregator(self, delta_noise_preds, weights, batch_size):
        """ 
        Notes: 
        - weights: an array with the weights for combining the noise predictions
        - delta_noise_preds: [B x K, 4, 64, 64], K = max_prompts_per_dir
        """
        delta_noise_preds = delta_noise_preds.split(batch_size, dim=0) # K x [B, 4, 64, 64]
        weights = weights.split(batch_size, dim=0) # K x [B]
        # print(f"{weights[0].shape = } {weights = }")

        assert torch.all(weights[0] == 1.0)

        main_positive = delta_noise_preds[0] # [B, 4, 64, 64]

        accumulated_output = torch.zeros_like(main_positive)
        for i, complementary_noise_pred in enumerate(delta_noise_preds[1:], start=1):
            # print(f"\n{i = }, {weights[i] = }, {weights[i].shape = }\n")

            idx_non_zero = torch.abs(weights[i]) > 1e-4
            
            # print(f"{idx_non_zero.shape = }, {idx_non_zero = }")
            # print(f"{weights[i][idx_non_zero].shape = }, {weights[i][idx_non_zero] = }")
            # print(f"{complementary_noise_pred.shape = }, {complementary_noise_pred[idx_non_zero].shape = }")
            # print(f"{main_positive.shape = }, {main_positive[idx_non_zero].shape = }")
            if sum(idx_non_zero) == 0:
                continue
            accumulated_output[idx_non_zero] += weights[i][idx_non_zero].reshape(-1, 1, 1, 1) * self.batch_get_perpendicular_component(complementary_noise_pred[idx_non_zero], main_positive[idx_non_zero])
        
        #assert accumulated_output.shape == main_positive.shape,# f"{accumulated_output.shape = }, {main_positive.shape = }"


        return accumulated_output + main_positive
    @torch.no_grad()
    def get_perpendicular_component(self, x, y):
        assert x.shape == y.shape
        return x - ((torch.mul(x, y).sum())/max(torch.norm(y)**2, 1e-6)) * y


    @torch.no_grad()
    def batch_get_perpendicular_component(self, x, y):
        assert x.shape == y.shape
        result = []
        for i in range(x.shape[0]):
            result.append(self.get_perpendicular_component(x[i], y[i]))
        return torch.stack(result)
    @torch.no_grad()
    def get_controlnet_cond_kwargs(
        self,
        height: int,
        width: int,
        pooled_prompt_embeds: torch.Tensor,
        negative_pooled_prompt_embeds: torch.Tensor,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        do_classifier_free_guidance: bool = False,
    ):
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(self.controlnet.nets) if isinstance(self.controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]
    
        timesteps = self.scheduler.timesteps
        # 8.2 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            if isinstance(self.controlnet, MultiControlNetModel):
                controlnet_keep.append(keeps)
            else:
                controlnet_keep.append(keeps[0])

        batch_size = pooled_prompt_embeds.shape[0]

        add_text_embeds = pooled_prompt_embeds

        return {"controlnet_keep": controlnet_keep, "add_text_embeds": add_text_embeds}
    
    

    @torch.no_grad()
    def noise_predict(
        self,
        latents,
        mask,
        masked_image_latents,
        t,
        device,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        guidance_scale=5.0,
        control_image=None,
        controlnet_conditioning_scale=1.0,
        controlnet_keep=None,
        add_text_embeds=None,
        add_time_ids=None,
        guess_mode=False,
        cross_attention_kwargs=None,
        **extra_kwargs,
    ):
        do_classifier_free_guidance = guidance_scale > 1.0
        #print("if doing cfg: ", do_classifier_free_guidance)
        #print("doing cfg: ", do_classifier_free_guidance)
        batch_size = latents.shape[0]
        # expand the latents if we are doing classifier free guidance
        #latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        #print("shape of latents: ", latent_model_input.shape)

        # concat latents, mask, masked_image_latents in the channel dimension
        latent_model_input_original = latents
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        if do_classifier_free_guidance:
            #print("doing cfg")
            
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            #prompt_embeds = prompt_embeds
        # controlnet(s) inference
        if guess_mode and do_classifier_free_guidance:
            # Infer ControlNet only for the conditional batch.
            control_model_input = latents
            control_model_input = self.scheduler.scale_model_input(control_model_input, t)
            controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
        else:
            #print("doing no cfg")
            control_model_input = latent_model_input
            #print("if prompt embeds is none: ", prompt_embeds is None)
            controlnet_prompt_embeds = prompt_embeds
        timesteps = self.scheduler.timesteps
        # 8.2 Create tensor stating which controlnets to keep
        controlnet_keep = []
        control_guidance_start = [0.0]
        control_guidance_end = [1.0]
        for i in range(len(timesteps)):
            
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0])
        
        i = (self.scheduler.timesteps == t).nonzero(as_tuple=True)[0].item()
        if isinstance(controlnet_keep[i], list):
            cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
        else:
            controlnet_cond_scale = controlnet_conditioning_scale
            if isinstance(controlnet_cond_scale, list):
                controlnet_cond_scale = controlnet_cond_scale[0]
            cond_scale = controlnet_cond_scale * controlnet_keep[i]
        

        control_model_input = control_model_input.to(torch.float32)
        controlnet_prompt_embeds = controlnet_prompt_embeds.to(torch.float32)
        control_image = control_image.to(torch.float32)
        cond_scale = torch.tensor(cond_scale).to(torch.float32)
        controlnet_prompt_embeds = controlnet_prompt_embeds.expand(control_model_input.shape[0], -1, -1)


        down_block_res_samples, mid_block_res_sample = self.controlnet(
            control_model_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=control_image,
            conditioning_scale=cond_scale,
            guess_mode=guess_mode,
            return_dict=False,
        )
        if guess_mode and do_classifier_free_guidance:
            down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
            mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])


        #mask = mask.unsqueeze(0)
        if do_classifier_free_guidance:
            mask = torch.cat([mask]*2)
        else:
            mask = mask
        
        if do_classifier_free_guidance:
            masked_image_latents = torch.cat([masked_image_latents]*2)
        else:
            masked_image_latents = masked_image_latents
        #if　dimension of these three dont match    
        #print("shape of latent input: ", latent_model_input.shape)
        #print("shape of mask: ", mask.shape)
        #print("shape of masked image latents: ", masked_image_latents.shape)

        latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)



        device = device
        latent_model_input = latent_model_input.to(device=device, dtype=torch.float16)
        t = torch.tensor(t, device=device, dtype=torch.float16)

        prompt_embeds = prompt_embeds.to(device=device, dtype=torch.float16)
        down_block_res_samples = [d.to(device=device, dtype=torch.float16) for d in down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample.to(device=device, dtype=torch.float16)

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            #print("Before rescale:", noise_pred.std())
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=1.0)
            #print("After rescale:", noise_pred.std())  
           

        return noise_pred
    


    def loss_sds_latent(self, image, latents,mask,timestep, predict_kwargs, device, generator, noise=None, latent_mask=None, prompts_embed = None, negative_prompt_embeds=None,return_aux=False):
        batch_size = latents.shape[0]

        # pred noise
        with torch.no_grad():
            if noise is None:
                noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(latents, noise, torch.tensor([timestep]))
            masked_image = image * (mask < 0.5)
            self.prepare_mask_latents = custom_prepare_mask_latents.__get__(self, CustomStableDiffusionControlNetInpaintPipeline)
            masked_image_latents=self.prepare_mask_latents(mask, masked_image, batch_size, 512, 512, latents.dtype, self._execution_device, generator, False)[1]
            if 'prompt_embeds' in predict_kwargs:
                del predict_kwargs['prompt_embeds']
            #print(prompts_embed)
            noise_pred = self.noise_predict(noisy_latents, mask, masked_image_latents, timestep, device, prompts_embed, negative_prompt_embeds=negative_prompt_embeds, **predict_kwargs)
        
        grad = noise_pred - noise
        if latent_mask is not None:
            grad = grad * latent_mask
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        #loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        #l1 loss
        loss = 0.5 * F.l1_loss(latents, target, reduction="sum") / batch_size
        #print("loss: ", loss)

        if return_aux:
            aux = {"noise_pred": noise_pred, "grad": grad}
            return loss, aux
        return loss

    def loss_sds_image(self, image, mask, input, timestep, predict_kwargs, device, generator, multistep=10, image_mask=None, noise=None, lpips=None, prompts_embed = None, negative_prompt_embeds=None, return_aux=False):
        batch_size = image.shape[0]
        
        def get_sub_timesteps(timestep, n_steps):
            # if timestep_min is None:
            timestep_idx = (self.scheduler.timesteps == timestep).nonzero(as_tuple=True)[0].item()
            timestep_min_idx = len(self.scheduler.timesteps) - 1
            assert timestep_min_idx - timestep_idx >= n_steps, f"n_steps is too large: {timestep_min_idx} - {timestep_idx} < {n_steps}"
            timestep_idx_list = torch.linspace(timestep_idx, timestep_min_idx, n_steps + 1).round().int()
            timestep_list = self.scheduler.timesteps[timestep_idx_list]
            return timestep_list

        
        # encode
        image_latents = self._encode_vae_image(image, generator=generator)
        if image_mask is not None:
            latent_mask = self.encode_mask(image_mask)
        else:
            latent_mask = None

        # latent denoising process
        if noise is None:
            noise = torch.randn_like(image_latents)
            #print("shape of noise: ", noise.shape)
        noisy_latents = self.scheduler.add_noise(image_latents, noise, torch.tensor([timestep]))
        if not multistep>1:
            multistep = 5
        #masked_image = 
        masked_image = image * (mask < 0.5)
        if multistep > 1:
            # run denoising untill t=0
            
            #print("timestep: ", timestep)
            timestep_list = get_sub_timesteps(timestep, multistep)
            for i in range(multistep):

                timestep, prev_timestep = timestep_list[i], timestep_list[i + 1]
                #print("timestep: ", timestep)
                
                #print("prev_timestep: ", prev_timestep)
                #fixme
                #masked_image = image * (mask < 0.5)
                #print("image and mask shape: ", image.shape, mask.shape)
                if input is not None:
                    masked_image = input
                #else:
                    #masked_image = image * (mask < 0.5)
                self.prepare_mask_latents = custom_prepare_mask_latents.__get__(self, CustomStableDiffusionControlNetInpaintPipeline)
                #print('mask shape before squeeze: ', mask.shape)
                
                mask = mask.squeeze(1)
                #print("mask after squeeze: ", mask.shape)
                mask, masked_image_latents=self.prepare_mask_latents(mask.unsqueeze(0), masked_image, batch_size, 512, 512, image_latents.dtype, self._execution_device, generator, False)
                #print('mask shape before predicting noise: ', mask.shape)
                mask = mask[:, :1, :, :]
                if 'prompt_embeds' in predict_kwargs:
                    del predict_kwargs['prompt_embeds']
                #print('nega prompts: ', negative_prompt_embeds)
                noise_pred = self.noise_predict(noisy_latents, mask, masked_image_latents, timestep, device, prompts_embed, negative_prompt_embeds=negative_prompt_embeds, **predict_kwargs)
                #print("scheduler: ", self.scheduler)
                noisy_latents = self.scheduler.step(
                    noise_pred, timestep, noisy_latents, prev_timestep=prev_timestep, return_dict=False
                )[0]

                if latent_mask is not None and i < multistep - 1: # do not apply mask at the last step
                    image_latents_noised = self.scheduler.add_noise(image_latents, noise, torch.tensor([prev_timestep]))
                    noisy_latents = (1 - latent_mask) * image_latents_noised + latent_mask * noisy_latents
                #if i==5:
                #    mid_noise = noisy_latents
            total_noise_pred = noisy_latents-  image_latents
            #print("total_noise_pred: ", total_noise_pred.std())
            grad = total_noise_pred-noise
            if latent_mask is not None:
                grad = grad * latent_mask
            target = (image_latents - grad).detach()
            loss_latent = 0.5 * F.mse_loss(image_latents, target, reduction="mean") / batch_size
            latents_pred = noisy_latents
                
        else:

            masked_image = image * (mask < 0.5)
            self.prepare_mask_latents = custom_prepare_mask_latents.__get__(self, CustomStableDiffusionControlNetInpaintPipeline)
            masked_image_latents=self.prepare_mask_latents(mask, masked_image, batch_size, 512, 512, image_latents.dtype, self._execution_device, generator, False)[1]
            if 'prompt_embeds' in predict_kwargs:
                del predict_kwargs['prompt_embeds']

            noise_pred = self.noise_predict(noisy_latents, mask, masked_image_latents, timestep, device, prompts_embed, negative_prompt_embeds=negative_prompt_embeds, **predict_kwargs)
            latents_pred = self.scheduler.get_original_sample(noise_pred, timestep, noisy_latents)

        # decode
        image_pred = self.vae.decode(latents_pred / self.vae.config.scaling_factor, return_dict=False)[0] # range (-1, 1)
        #mid_image = self.vae.decode(mid_noise / self.vae.config.scaling_factor, return_dict=False)[0]
    
        if image_mask is not None:
            # 融合后的预测图像
            image_pred = (1 - image_mask) * image + image_mask * image_pred

            # 计算 mask 区域的 MSE loss
            masked_mse_loss = F.mse_loss(image * image_mask, image_pred * image_mask, reduction="sum") / (image_mask.sum() * batch_size)

            # 计算 LPIPS loss，仅在 mask 区域上
            masked_image = image * image_mask
            masked_image_pred = image_pred * image_mask
            #lpips_loss = lpips(masked_image, masked_image_pred).mean()

            # 合并损失
            #loss = 0.5 * masked_mse_loss + lpips_loss
            loss = 0

        if return_aux:
            aux = {"image_pred": image_pred, "latents_pred": latents_pred, "noise_pred": noise_pred}
            return loss, aux
        return loss
