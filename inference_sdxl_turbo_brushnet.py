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
    pipe = pipe.to(device)

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
            mask = mask[:, :, np.newaxis]

            init_image = src_img * (1 - mask)
            init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
            mask_image = Image.fromarray(mask.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")

            #### Start ####
            start_time = time.time()
            image = pipe(
                image=init_image,
                mask=mask_image,
                height=1024,
                width=1024,
                brushnet_conditioning_scale=brushnet_conditioning_scale,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=0.0,
                prompt=edit_p,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_prompt=None,
                negative_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                generator=generator
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