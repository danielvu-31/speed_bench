import argparse
import json
import time
import torch
import copy
import sys
import numpy as np
import os
from safetensors.torch import load_file
from accelerate import Accelerator, DistributedType, init_empty_weights
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.training_utils import EMAModel

from PIL import Image
from tqdm import tqdm
from utils import *
from torchvision.utils import save_image
from transformers import AutoTokenizer, PretrainedConfig

import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import torch

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler, StableDiffusionXLBrushNetPipeline, BrushNetModel, DPMSolverMultistepScheduler, UniPCMultistepScheduler, AutoencoderKL
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from transformers import CLIPTextModel


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Infer samples from sana sprint."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        required=True,
        help="Num inference steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        required=True,
        help="guidance scale",
    )
    parser.add_argument(
        "--generator_seed",
        type=int,
        default=42,
        help="seed",
    )

    ###### BrushNet ######
    parser.add_argument(
        "--brushnet_mode",
        type=str,
        default="random",
        choices=["segment", "random"],
    )

    ###### Inversion ######
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="lol",
    )
    parser.add_argument(
        "--fill_mask_type",
        type=str,
        default="gaussian",
        choices=["zero", "gaussian", "no"],
    )
    parser.add_argument(
        "--input_mask_type",
        default="image",
        choices=["latent", "image", "zero"],
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="lol",
    )

    return parser.parse_args()


@torch.no_grad()
def process_and_encode(
    src_img, pipe, height=1024, width=1024,
    device="cuda:0", weight_dtype=torch.bfloat16
):
    image_processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor)
    init_image = image_processor.preprocess([src_img], height=height, width=width)
    
    # vae works well for float32
    init_image = init_image.to(device=device, dtype=torch.float32)
    vae = pipe.vae.to(dtype=torch.float32)

    image_latents = vae.encode(init_image).latent_dist
    image_latents = image_latents.sample().to(dtype=weight_dtype) * pipe.vae.config.scaling_factor

    # switch back to bf16
    pipe.vae.to(dtype=weight_dtype)
    return init_image, image_latents


def get_add_time_ids(
    original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids


@torch.no_grad()
def vae_decode(latents, vae):
    latents = latents.to(vae.dtype)
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    return image


@torch.no_grad()
def vae_encode(images, vae, weight_dtype=torch.bfloat16):
    vae = vae.to(torch.float32)
    images = images.to(torch.float32)

    latents = vae.encode(images).latent_dist.sample() *  vae.config.scaling_factor
    vae = vae.to(weight_dtype)

    return latents


if __name__ == "__main__":
    args = parse_args()
    negative_prompt = "blurry, low quality, low resolution, bad anatomy, bad hands, bad feet, poorly drawn, extra limbs, missing limbs, disfigured, deformed, mutated, cloned face, fused fingers, long neck, unrealistic, unnatural, watermark"
    
    weight_dtype = torch.bfloat16
    
    # Define sana inpaint pipeline
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.generator_seed)

    base_data_dir = "checkpoint"
    json_brushnet_data_path = "checkpoint/mapping_file.json"

    sdxl_path = "stabilityai/sdxl-turbo"
    brushnet_path = "checkpoint/brushnet/random_mask_brushnet_ckpt_sdxl_v0"

    brushnet = BrushNetModel.from_pretrained(
        brushnet_path,
        torch_dtype=weight_dtype
    ).to(device)

    pipe = StableDiffusionXLBrushNetPipeline.from_pretrained(
        sdxl_path,
        brushnet=brushnet,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=False,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)

    #### Inversion Net ####
    checkpoint_path = "checkpoint/sdxl/kien_reg-mean+var0.5_fullparams1e-5_maskimage+gaussianblend+losspost_1step_randommask_clsgan+lr1e-6+weight0.5+1critic/checkpoint-3000"
    if args.use_ema:
        print("Loading EMA inversion net from: ", checkpoint_path)
        inversion_network_path = os.path.join(checkpoint_path, "unet_ema")

        config = json.load(open(os.path.join(inversion_network_path, "config.json")))
        inverse_network = UNet2DConditionModel.from_config(config).to(device, dtype=weight_dtype)
        
        state_dict = {}
        for shard_path in [
            "diffusion_pytorch_model-00001-of-00002.safetensors",
            "diffusion_pytorch_model-00002-of-00002.safetensors",
        ]:
            state_dict.update(load_file(os.path.join(inversion_network_path, shard_path)))
        inverse_network.load_state_dict(state_dict, strict=False)
    else:
        print("Loading inversion net from: ", checkpoint_path)
        inverse_network = UNet2DConditionModel.from_pretrained(
            sdxl_path, 
            subfolder="unet",
            revision=None,
            variant=None,
        ).to(device, dtype=weight_dtype)
        try:
            inversion_network_path = os.path.join(checkpoint_path, "unet", "diffusion_pytorch_model.safetensors")
            inverse_network.load_state_dict(load_file(inversion_network_path), strict=True)
        except:
            inversion_network_path = os.path.join(checkpoint_path, "unet", "model.safetensors")
            inverse_network.load_state_dict(load_file(inversion_network_path), strict=True)

    text_encoding_pipeline = StableDiffusionXLPipeline.from_pretrained(
        sdxl_path,
        torch_dtype=weight_dtype,
    ).to(device, dtype=weight_dtype)

    # read data json
    with open(json_brushnet_data_path, "r") as fp:
        mapping_file = json.load(fp)
    
    speed_list = []
    brushnet_conditioning_scale = 1.0
    with torch.no_grad():
        for it, (key, sample) in tqdm(enumerate(mapping_file.items())):
            if it >= 200:
                break

            img_path = osp.join(base_data_dir, sample["image"])
            id_img = sample["image"].split("/")[-1].split(".")[0]
            mask_read = sample["inpainting_mask"]

            edit_p = sample["caption"]
            print("Normal: ", edit_p)
            
            src_img = cv2.imread(img_path)[:, :, ::-1]
            src_img = cv2.resize(src_img, (1024, 1024))

            # process mask for blending
            mask = rle2mask(mask_read, (1024, 1024))

            # Downsample mask to blend noise
            latent_size = 128
            mask_blend = mask.astype(np.float32)
            mask_blend_to_save = torch.from_numpy(
                mask_blend
            ).unsqueeze(0).unsqueeze(0).to(device)
            mask_blend = F.interpolate(
                mask_blend_to_save, 
                size=(latent_size, latent_size), 
                mode='bilinear', align_corners=False
            )

            tmp_img = Image.open(img_path).convert("RGB").resize(
                (1024, 1024), Image.Resampling.LANCZOS
            )
            gt_img, gt_img_latents = process_and_encode(tmp_img, pipe)

            mask = mask[:, :, np.newaxis]
            init_image = src_img * (1 - mask)
            init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
            mask_image = Image.fromarray(mask.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")

            if args.input_mask_type == "image":
                masks_tensor = torch.from_numpy(mask.astype(np.uint8)).permute(2, 0, 1).unsqueeze(0).repeat(1, 3, 1, 1).to(device, weight_dtype)
                masked_gt_img = (1 - masks_tensor) * ((torch.clamp(gt_img, min=-1, max=+1) + 1) / 2)
                masked_gt_img_latents = vae_encode(
                    (masked_gt_img - 0.5) * 2.0, pipe.vae, weight_dtype
                ).cuda()
            elif args.input_mask_type == "latent":
                fg_noise = randn_tensor(gt_img_latents.shape, generator=generator, device=device, dtype=weight_dtype)
                masked_gt_img_latents = mask_blend * fg_noise + (1 - mask_blend) * gt_img_latents
            elif args.input_mask_type == "zero":
                masked_gt_img_latents = (1 - mask_blend) * gt_img_latents
            masked_gt_img_latents = masked_gt_img_latents.to(weight_dtype)
            
            # Encode prompt
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = text_encoding_pipeline.encode_prompt(
                edit_p, 
                negative_prompt=negative_prompt, 
                do_classifier_free_guidance=True
            )
            add_time_ids = get_add_time_ids(
                original_size=(1024, 1024),
                crops_coords_top_left=(0.0, 0.0),
                target_size=(1024, 1024),
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=int(pooled_prompt_embeds.shape[-1]),
            ).to(device).repeat(1, 1)
            add_text_embeds = pooled_prompt_embeds
            unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

            # Timestep
            t = (torch.ones((1, )) * 500).to(device)
            random_noise = randn_tensor(masked_gt_img_latents.shape, generator=generator, device=device, dtype=weight_dtype)

            #### Start ####
            start_time = time.time()
            pred_inverted_noise = inverse_network(
                masked_gt_img_latents,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=unet_added_cond_kwargs,
                return_dict=False,
            )[0]

            if args.fill_mask_type == "gaussian":
                blend_noise = mask_blend * random_noise + (1 - mask_blend) * pred_inverted_noise
            elif args.fill_mask_type == "zero":
                blend_noise = (1 - mask_blend) * pred_inverted_noise
            elif args.fill_mask_type == "no":
                blend_noise = pred_inverted_noise
            blend_noise = blend_noise.to(dtype=weight_dtype)

            image = pipe(
                latents=blend_noise,
                image=init_image,
                mask=mask_image,
                height=1024,
                width=1024,
                brushnet_conditioning_scale=brushnet_conditioning_scale,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=0.0,
                generator=generator,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt=None,
                negative_prompt_embeds=None,
                negative_pooled_prompt_embeds=None
            ).images[0]
            run_time = time.time() - start_time

            if it >= 100 and it < 200:
                speed_list.append(run_time)
            print(f"Finish inpainting within {run_time}")
    
    mean_runtime = np.mean(speed_list)
    print(f"AVERAGE: ", np.mean(speed_list))
    with open(f"{args.exp_name}.txt", "w") as f:
        f.write(f"AVERAGE: {mean_runtime:.16f}\n")
        f.close()


#### python speed/inference_kien_sdxl_turbo_brushnet.py --num_inference_steps 2 --guidance_scale 0.0 --checkpoint_path "/lustre/scratch/client/movian/research/users/ducvh5/sana_lora_inpainting/sdxl_exps/kien_reg-mean+var0.5_fullparams1e-5_maskimage+gaussianblend+losspost_1step_randommask_clsgan+lr1e-6+weight0.5+1critic/checkpoint-3000" --fill_mask_type "gaussian" --input_mask_type "image" --use_ema